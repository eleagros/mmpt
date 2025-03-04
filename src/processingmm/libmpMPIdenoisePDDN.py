# *** libmpMPIdenoisePDDN ***
#
# Use this library to perform Denoising on Mueller Polarimetric Imaging (HORAO project)
#
# * Version: Multi-threaded CPUs + GPU 
#
#
# * The library builds on:
#
# - Denoising Diffusion Networks 
#   source: https://github.com/lucidrains/denoising-diffusion-pytorch
#
# - Mueller Polarimetric Decompositions
#   source: https://github.com/stefanomoriconi/libmpMuelMat
#
#
# and it provides a *full* framework for:
#
# - Estimating (training) new Models from Input Polarimetric Intensities
#
# - Denoising (inference) a set of Polarimetric Intensities
#
# - Synthesising (inference) a set of Polarimetric Intensities with a Generative Approach
#
#
# * Usage:
#
#		>> from libmpMPIdenoisePDDN import MPI_PDDN
#       >> PDDN = MPI_PDDN([Path_to_PDDN_model.pt])
#
#
# * Help:
#
#		>> PDDN.Help()
# 
# * Requirements:
#
#		libmpMPIdenoisePDDN_requirements.txt
#
#
# * References and Credits:
#
# [1] Moriconi, S., et al. "Near-real-time Mueller polarimetric image processing for neurosurgical intervention", Int J CARS (2024). https://doi.org/10.1007/s11548-024-03090-6
# [2] Moriconi, S., et al. "Denoising Diffusion Network for Real-Time Neurosurgical Mueller Polarimetric Imaging", Medical Image Analysis (2024) - Under Review
# [3] Moriconi, S., "libmpMuelMat Computational tools for MPI", Technical report (2022). https://github.com/stefanomoriconi/libmpMuelMat
#
# Author (libmpMPIdenoisePDDN): Stefano Moriconi, April 2024, at Inselspital, for HORAO project.
#                               email: stefano.nicola.moriconi@gmail.com
#                               website: https://stefanomoriconi.github.io/mypage/
#
# This Codebase is a polarimetric adaptation of the original package:
#	Phil Wang, aka 'lucidrains' - Denoising Diffusion Probabilistic Model, in Pytorch
# 	source: https://github.com/lucidrains/denoising-diffusion-pytorch


# Importing Required Libraries
import math
import warnings
import copy
import numpy as np


from inspect import isfunction
from functools import partial

from torch.utils import data
from multiprocessing import cpu_count

from pathlib import Path

import torch
import torchsummary
from torch import nn, einsum
import torch.nn.functional as F
from torch.amp import GradScaler # autocast, 
from torch.optim import Adam
from torchvision import transforms, utils
    
from PIL import Image, ImageDraw, ImageFont

from tqdm import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from ema_pytorch import EMA

## Added components (Stefano)
# Training Monitoring
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

    

import webbrowser

# library to export PNGs and GIFs
import imageio
# Performance
import time
# Windowing Signals
import scipy.signal

# Default FileNames and Files Management
from datetime import datetime
import shutil

try:
    torch.cuda.empty_cache()
except:
    pass
    
### IMPLEMENTATION ###

### Helper Functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

### Small Helper Modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

### Sinusoidal Positional Embedding

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    '''
    following @crowsonkb 's lead with learned sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8
    '''

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

### Building-Block Modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

### Polarimetric U-Net Model

class MPI_PDDN_Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3, # Original RGB
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out * 2, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

### Polarimetric Gaussian Diffusion Class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # ORIGINAL

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    '''
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    '''

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class MPI_PDDN_GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 3, # Original RGB
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1
    ):
        super().__init__()
        assert not (type(self) == MPI_PDDN_GaussianDiffusion and denoise_fn.channels != denoise_fn.out_dim)

        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.objective = objective

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise): # ORIG
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t): # ORIG
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool): # ORIG
        model_output = self.denoise_fn(x, t)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True): # ORIG
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, gifNumFrames = 100, gifOutputFileName=None): # ORIG + GIF ADD-ONs
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        # GIF OPTION (Stefano ADD)
        if not (gifOutputFileName == None):# Exporting GIF
            if gifNumFrames < 100:
                gifNumFrames = 100
                print(' [wrn] Min Gif frames set to: 100 - Log-spaced')

            if gifNumFrames == self.num_timesteps:  # ALL the time-points in the Markov-chain
                gifTseq = list(range(0, self.num_timesteps))
            else:  # A subset (>=100) of time-points in the Markov-chain (logarithmically spaced)
                gifTseq = myUtils_genLogDynamic(0, self.num_timesteps - 1, gifNumFrames)

            print(' >> Exporting Gif to: ' + gifOutputFileName + '-- This may take some time...')

            gifOutputSize = 512  # 512x512 FIXED for memory and resampling (Change at your own risk)
            gifResizeFactor = gifOutputSize/(shape[-1]*np.sqrt(shape[1])*np.ceil(np.sqrt(shape[0])))
            gifTransform = transforms.Compose([transforms.Resize(int(shape[-1]*gifResizeFactor))])
            with imageio.get_writer(gifOutputFileName, mode='I') as writer:
                for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',total=self.num_timesteps):
                    img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
                    if np.isin(i,gifTseq): # logarithmic time sequence
                        gifFrame = myUtils_catImg4DToImg2D( unnormalize_to_zero_to_one(gifTransform(img))) # (re-sizing the 2D montage image)
                        writer.append_data((gifFrame * 255).round().astype(np.uint8))
                    if i==0: # !!! put 10 extra frames at convergence (just for better visualisation of the final synthesis)
                        for jj in range(0,10):
                            gifFrame = myUtils_catImg4DToImg2D(unnormalize_to_zero_to_one(gifTransform(img))) # (re-sizing the 2D montage image)
                            writer.append_data((gifFrame * 255).round().astype(np.uint8))

        else: # Original code without exporting GIF
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16, gifOutputFileName=None): # ORIG
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), gifOutputFileName)

    @torch.no_grad()
    def sampleGIF(self, batch_size = 4, gifNumFrames = 100, gifOutputSize = 512): # REF (Stefano ADD)
        image_size = self.image_size
        channels = self.channels

        device = self.betas.device
        img = torch.randn((batch_size,channels,image_size,image_size), device=device)

        gifData = []
        gifExpectedChannels = 16
        if gifNumFrames == self.num_timesteps: # ALL the time-points in the Markov-chain
            gifTseq = list(range(0,self.num_timesteps))
        else: # A subset (>=100) of time-points in the Markov-chain (logarithmically spaced)
            gifTseq = myUtils_genLogDynamic(0, self.num_timesteps - 1, gifNumFrames)
        gifResizeFactor = gifOutputSize / (image_size * np.sqrt(gifExpectedChannels) * np.ceil(np.sqrt(batch_size)))
        gifTransform = transforms.Compose([transforms.Resize(int(image_size * gifResizeFactor))])

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            if np.isin(i, gifTseq):  # logarithmic time sequence
                gifFrame = myUtils_catImg4DToImg2D(unnormalize_to_zero_to_one(gifTransform(img)))  # (re-sizing the 2D montage image)
                gifFrame = (gifFrame-gifFrame.min())/(gifFrame.max()-gifFrame.min()) # [0,1]
                labelFrame = myUtils_genTextPhantom('T: {:d}/{:d}'.format(i,self.num_timesteps), gifOutputSize)
                gifFrame = np.multiply(gifFrame, labelFrame)
                gifFrame = (gifFrame * 255).round().astype(np.uint8)
                gifFrame = np.reshape(gifFrame,[1,1,1,gifOutputSize,gifOutputSize]) # N T C H W (Batches, Timepoint, Channels, Height, Width)
                if len(gifData) == 0:
                    gifData = gifFrame
                else:
                    gifData = np.concatenate((gifData,gifFrame),axis=1)

        return gifData

    @torch.no_grad()
    def img_denoise_OLA(self, imgIN, tstr=1, tend=0, hop=0.25, verboseFlag=False): # REF (NUMPY + TORCH)
        '''
        Use this function to denoise a polarimetric image with the trained Dissufion Network.
        This function embeds an optimised Overlap-and-Add (OLA) routine to filter the input image which may vary in size
        (H,W) with respect to the reference patch-size the network is trained on.
        To optimise performance the OLA is performed on GPU tensors and ultimately converted back to np.ndarray.

        NB: the channels in the 3rd dimension of the input tensor are assumed constant.

        Inputs:

        :param imgIN: input image to denoise as np.ndarray [H,W,C]
        :param tstr: scalar integer indicating the diffusion starting time-step , any ]0,1000]
        :param tend: scalar integer indicating the diffusion ending time-step, usually 0
        :param hop: scalar for the normalised overlap size: usually 0.5 or [0.25] for faster performance
        :param verboseFlag: scalar boolean to enable performance printout

        Outputs:

        :return: imgOUT: output filtered (denoised) image as np.ndarray of size [H,W,C]
        :return: elapsTime: performance analysis [step-wise and total]
        '''

        # imgIN expected of shape: [H,W,C]
        if not (type(imgIN) is np.ndarray):
            raise Exception(
                ' <!> sample_denoise: input "imgIN" expected to be a numpy ndarray -- Found: {}'.format(type(imgIN)))
        imgIN_shp = np.shape(imgIN)
        if not (len(imgIN_shp) == 3):
            raise Exception(
                ' <!> sample_denoise: input "imgIN" expected to have 3D shape [H,W,C] -- Found: {}'.format(imgIN_shp))
        if not (imgIN_shp[2] == self.channels):
            raise Exception(' <!> sample_denoise: input "imgIN" expected to have', self.channels,
                            'channels -- Found: {}'.format(imgIN_shp[2]))
        if not (tstr <= self.num_timesteps):
            raise Exception(' <!> sample_denoise: input "tstr" expected to be <=', self.num_timesteps,
                            '-- Found: {}'.format(tstr))
        if not (tend >= 0):
            raise Exception(' <!> sample_denoise: input "tend" expected to be >=', 0, '-- Found: {}'.format(tend))

        # Step-wise Performance Analysis
        proc_t = time.time()
        STEPsTime_elaps = []
        preproc_t = time.time()

        wlen = self.image_size
        # New (Faster)
        hwin2, ov, _ = myUtils_getHannWin2Dov(wlen, hop=hop)  # the ORIGINAL has hop=0.5 (ok, but slower)
        h3D = np.tile(hwin2.reshape([wlen, wlen, 1]), [1, 1, imgIN.shape[-1]])

        imgIN_minVal = np.min(imgIN)
        imgIN_maxVal = np.max(imgIN)
        imgIN = (imgIN - imgIN_minVal) / (imgIN_maxVal - imgIN_minVal)  # Rescaling [0,1]

        # Padding Input Intensities (Noisy) and getting the final cropping-sizes
        imgIN_pad, Rpadsmpls, Cpadsmpls = myUtils_padImg3D(imgIN, wlen)

        # Necessary Rescale to set the range of Values within [-1,1]
        imgIN_pad = normalize_to_neg_one_to_one(imgIN_pad)

        # NEW (FASTER)
        hopLen = wlen - ov
        Rmax = int(np.ceil(imgIN_pad.shape[0] / hopLen) - 1)
        Cmax = int(np.ceil(imgIN_pad.shape[1] / hopLen) - 1)

        # Step-wise Performance Analysis
        STEPsTime_elaps.append(time.time() - preproc_t)
        img2btcTensor_t = time.time()

        # Converting to Tensor as is
        tform = transforms.Compose([transforms.ToTensor()])
        imgIN_pad_T = tform(imgIN_pad)  # Initialising output as NOISY input (as tensor)

        # image2multibatchTensor conversion:
        for blk in range(0, Rmax * Cmax):
            r, c = np.unravel_index(blk, (Rmax, Cmax))
            if blk == 0:
                imgBTC_T = imgIN_pad_T[:, int(r * hopLen): int(wlen + r * hopLen),
                           int(c * hopLen): int(wlen + c * hopLen)].clone().detach().reshape([1, imgIN.shape[-1], wlen, wlen])
            else:
                imgBTC_T = torch.cat((imgBTC_T,
                                      imgIN_pad_T[:, int(r * hopLen): int(wlen + r * hopLen),
                                      int(c * hopLen): int(wlen + c * hopLen)].clone().detach().reshape(
                                          [1, imgIN.shape[-1], wlen, wlen])),
                                     dim=0)
        # Pre-processing ENDS here

        imgBTC_T = imgBTC_T.cuda().type(torch.float32) # allocation to gpu memory as float32

        # Step-wise Performance Analysis
        STEPsTime_elaps.append(time.time() - img2btcTensor_t)
        filtering_t = time.time()

        # ACTUAL DENOISING FILTERING: the image is denoised as a multiBatchTensor in just one go
        for i in reversed(range(tend, tstr)):  # PDDN filtering -- NOTE: for a single step denoising: tstr = 1; tend = 0
            imgBTC_T = self.p_sample(imgBTC_T,
                                     torch.full((int(Rmax * Cmax),), # creating a similar multiBatchTensor for the Time point
                                                i,
                                                device=self.betas.device,
                                                dtype=torch.long))

        # Necessary Re-Scaling after De-Noising
        imgBTC_T = unnormalize_to_zero_to_one(imgBTC_T)

        # Step-wise Performance Analysis
        STEPsTime_elaps.append(time.time() - filtering_t)
        inioutput_t = time.time()

        # Initialising output
        imgOUT_pad = np.zeros(imgIN_pad.shape)

        STEPsTime_elaps.append(time.time() - inioutput_t)
        overlapadd_t = time.time()

        # Integration of the denoised filtered responses into a polarimetric image:
        # This block-wise process compactly implements the sliding window overlap-add.
        for blk in range(0, Rmax * Cmax):  # OverLap-Add (OLA)
            r, c = np.unravel_index(blk, (Rmax, Cmax))
            imgOUT_sel = np.moveaxis(np.array(imgBTC_T[blk, :, :, :].cpu().squeeze(),dtype=np.double), 0, 2)
            imgOUT_pad[int(r * hopLen): int(wlen + r * hopLen),
                       int(c * hopLen): int(wlen + c * hopLen), :] = \
                imgOUT_pad[int(r * hopLen): int(wlen + r * hopLen),
                           int(c * hopLen): int(wlen + c * hopLen), :] + np.multiply(h3D, imgOUT_sel)

        # Step-wise Performance Analysis
        STEPsTime_elaps.append(time.time() - overlapadd_t)
        postprocess_t = time.time()

        # Removing padding
        imgOUT = imgOUT_pad[int(Rpadsmpls):-int(Rpadsmpls), int(Cpadsmpls):-int(Cpadsmpls), :]

        # Restoring Original Input Values Range (min,max)
        imgOUT = (imgOUT * (imgIN_maxVal - imgIN_minVal)) + imgIN_minVal

        # Step-wise Performance Analysis
        STEPsTime_elaps.append(time.time() - postprocess_t)

        ## Final Time (Total Performance)
        proc_t_elaps = time.time() - proc_t

        if verboseFlag:
            print(' >> MPI Denoising PDDN Performance: Elapsed time = {:.3f} s'.format(proc_t_elaps))

        return imgOUT, [STEPsTime_elaps, proc_t_elaps]

    @torch.no_grad()
    def img_synth_OLA(self, imgH=256, imgW=256, hop=0.25): #REF (NUMPY + TORCH)
        # Synthesising Arbitrary-size Polarimetric Intensities: Full Generative Trajectory

        tstr = self.num_timesteps
        tend = 0
        channels = self.channels
        device = self.betas.device
        extraPad = [int(self.image_size/2), int(self.image_size/2)]
        ini_T = torch.randn((1, channels, imgH + (2*extraPad[0]), imgW + (2*extraPad[1])), device=device)
        imgIN = np.moveaxis(np.array(ini_T.cpu().squeeze(), dtype=np.double), 0, 2).astype(np.float)
        imgIN_batch = 1

        wlen = self.image_size

        # Padding Input Intensities (Noisy) and getting the final cropping-sizes
        imgIN_pad, Rpadsmpls, Cpadsmpls = myUtils_padImg3D(imgIN, wlen)

        # Converting to Tensor as is
        tform = transforms.Compose([transforms.ToTensor()])
        imgIN_pad_T = tform(imgIN_pad)  # Initialising output as NOISY input (as tensor)
        imgIN_pad_T = imgIN_pad_T.clone().detach().reshape(
            [imgIN_batch, imgIN_pad.shape[2], imgIN_pad.shape[0], imgIN_pad.shape[1]])
        imgIN_pad_T = imgIN_pad_T.cuda().type(torch.float32)

        # New (Faster)
        hwin2, ov, _ = myUtils_getHannWin2Dov(wlen, hop)

        # Getting Hann Window -- for sliding window OverLap-and-Add (OLA)
        # h3D = np.tile(myUtils_getHannWin2D(wlen).reshape([wlen, wlen, 1]), [1, 1, imgIN_pad.shape[-1]])
        h3D = np.tile(hwin2.reshape([wlen, wlen, 1]), [1, 1, imgIN_pad.shape[-1]])
        h3D_T = tform(h3D)
        h3D_T = h3D_T.clone().detach().reshape([imgIN_batch, imgIN_pad.shape[-1], wlen, wlen])
        h3D_T = h3D_T.cuda().type(torch.float32)

        # NEW (FASTER)
        hopLen = wlen - ov
        Rmax = int(np.ceil(imgIN_pad.shape[0] / hopLen) - 1)
        Cmax = int(np.ceil(imgIN_pad.shape[1] / hopLen) - 1)

        # Overlap-and-Add Filtering
        for i in tqdm(reversed(range(tend, tstr)), desc='Generating Synthetic Image',total=np.abs(tend - tstr)):
            # Initialising OUTPUT
            imgOUT_pad_T = torch.zeros(imgIN_pad_T.shape, dtype=torch.float32).cuda()

            for blk in range(0, Rmax * Cmax):
                r, c = np.unravel_index(blk, (Rmax, Cmax))
                imgSEL_T = imgIN_pad_T[:, :, int(r * hopLen): int(wlen + r * hopLen),
                           int(c * hopLen): int(wlen + c * hopLen)].clone().detach()

                imgSEL_T = self.p_sample(imgSEL_T,
                                         torch.full((imgIN_batch,), i, device=self.betas.device, dtype=torch.long))

                imgOUT_pad_T[:, :, int(r * hopLen): int(wlen + r * hopLen), int(c * hopLen): int(wlen + c * hopLen)] = \
                imgOUT_pad_T[:, :, int(r * hopLen): int(wlen + r * hopLen), int(c * hopLen): int(wlen + c * hopLen)] + \
                h3D_T * imgSEL_T

            if i > tend:
                # Flattening and Re-arranging
                imgOUT_pad_tmp = np.array(imgOUT_pad_T.cpu().squeeze())
                imgOUT_pad_tmp = np.moveaxis(imgOUT_pad_tmp.astype(np.double), 0, 2)
                # Removing padding
                imgOUT_tmp = imgOUT_pad_tmp[int(Rpadsmpls):-int(Rpadsmpls), int(Cpadsmpls):-int(Cpadsmpls), :]
                # Re-Padding
                imgIN_pad, _, _ = myUtils_padImg3D(imgOUT_tmp, wlen)
                # To Tensor
                imgIN_pad_T = tform(imgIN_pad)  # Initialising output as NOISY input (as tensor)
                imgIN_pad_T = imgIN_pad_T.clone().detach().reshape(
                    [imgIN_batch, imgIN_pad.shape[2], imgIN_pad.shape[0], imgIN_pad.shape[1]])
                imgIN_pad_T = imgIN_pad_T.cuda().type(torch.float32)

        # Necessary Re-Scaling after De-Noising
        imgOUT_pad_T = unnormalize_to_zero_to_one(imgOUT_pad_T)

        # Reshaping the Denoised Output Image as original input shape and type=double
        imgOUT_pad = np.array(imgOUT_pad_T.cpu().squeeze())
        imgOUT_pad = np.moveaxis(imgOUT_pad.astype(np.double), 0, 2)

        # Removing padding
        imgOUT = imgOUT_pad[int(Rpadsmpls):-int(Rpadsmpls), int(Cpadsmpls):-int(Cpadsmpls), :]
        imgOUT = imgOUT[int(extraPad[0]):-int(extraPad[0]), int(extraPad[1]):-int(extraPad[1]), :]

        return imgOUT


    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5): #ORIG
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None): # ORIG
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self): # ORIG
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise = None): # ORIG
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.denoise_fn(x, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs): # ORIG
        # Patch-based output (used for training)
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)

### Polarimetric Dataset Class: specific data type [see COD file format in myUtils_read_cod_data_X3D or libmpMuelMat]

class MPI_PDDN_Dataset(data.Dataset): #REF
    # Dataset loader for Polarimetric INTENSITY data (output: 16 channels)
    def __init__(self, folder, image_size, exts = ['cod'], augment_Flag = False):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # The Dataset is loaded in randomised patches, concatenated along the batch dimension.
        # NOTE: the IMAGE-to-PATCH conversion, augmentation, and cropping is performed in the followin lines
        if augment_Flag:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(int(image_size*1.5),
                                      pad_if_needed=True,
                                      padding_mode='symmetric'),
                transforms.RandomRotation((-180, 180)),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(image_size), # alternatively for ramdom sampling: RandomCrop(image_size)
                nn.Identity(),
                transforms.Resize(image_size),
                ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        
        path = self.paths[index]
        
        # Load Image
        img = myUtils_read_cod_data_X3D( path, 0 ) # << Flag to remove file Header while reading ... see help. 
        # FLAG to remove header (1 = True) if the images are original from the MPI instrumentation
        # FLAG to remove header (0 = False) if images have been previously manipulated and exported, eg. Reflections Removed << RECOMMENDED!!!
        
        # Rescaling img Data [0,1]
        img = (img-np.min(img))/(np.max(img)-np.min(img)) # NOTE: Rescaling * THE ENTIRE IMAGE * [0,1] (not the patch!)
        return self.transform(img)

### Polarimetric Trainer Class (Loading Pre-trained Models and Estimating New Ones)

class MPI_PDDN_Trainer(object): #REF
    # This is the baseline Trainer for POLARIMETRIC INTENSITIES!
    def __init__(
            self,
            diffusion_model,
            PDDNdatasetFolderPath=None,
            PDDNmodelFileName=None, # final model path at convergence
            *,
            ema_decay=0.995,
            train_batch_size=16,
            train_lr=1e-4,
            train_num_steps=100001,
            gradient_accumulate_every=2,
            amp=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            temp_results_folder='./results',
            augment_Flag=True,  # Augmentation FLAG -- [Recommended]
            delete_temp_results_Flag=False,
            label=None,
    ):
        super().__init__()
        self.image_size = diffusion_model.image_size

        self.model = diffusion_model
        self.ema_decay = ema_decay
        self.ema = EMA(self.model, beta=self.ema_decay)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.disable_training_Flag = True
        self.augment_Flag = augment_Flag

        # Define the dataset
        self.PDDNdatasetFolderPath = PDDNdatasetFolderPath
        self.ds = None
        self.dl = None

        # Initialise the Adam optimiser
        self.train_lr = train_lr
        self.opt = None

        # Initialisation of training steps
        self.step = 0

        # Other initialisations
        self.amp = amp
        self.scaler = GradScaler(enabled=self.amp)

        # Initialise the output folder for the trained model to be exported
        self.results_folder = Path(temp_results_folder)

        self.delete_temp_results_Flag = delete_temp_results_Flag

        self.PDDNmodelFileName = PDDNmodelFileName
        # Note: The routine to export the final model into PDDNmodelFileName MUST be written!!!

        # User-defined Label for New Models to be Trained
        self.PDDNlabel = label
        # Other Info for Model Summary
        self.PDDNmodelSummary = {'PDDNmodelFileName':self.PDDNmodelFileName,
                                 'PDDNmodelLabel':self.PDDNlabel,
                                 'PDDNmodelDataset':self.PDDNdatasetFolderPath,
                                 'PDDNmodelTrainingSession':None,
                                 'PDDNmodelTrainingSteps':None,
                                 'PDDNmodelTrainingConvergenceSteps':None,
                                 'PDDNmodelLossType':None,
                                 'PDDNmodelLossValue':None,
                                 'PDDNmodelSource':self.PDDNmodelFileName,
                                 'PDDNmodelPatchSize':[self.model.image_size, self.model.image_size, self.model.channels],
                                 'PDDNmodelTotTimePoints':self.model.num_timesteps}

    def passParameters(self, PDDNdatasetFolderPath, PDDNmodelFileName, ema_decay, train_batch_size, train_lr, train_num_steps, gradient_accumulate_every, amp, step_start_ema, update_ema_every, save_and_sample_every, TempResultsFolderPath, augment_Flag, delete_temp_results_Flag, label):

        self.PDDNdatasetFolderPath = PDDNdatasetFolderPath
        self.batch_size = train_batch_size
        self.augment_Flag = augment_Flag

        if self.PDDNdatasetFolderPath is None:
            print(' [wrn] PDDN: <TRAINING MODE> input dataset folder path NOT Given/Found!')
        else:
            self.ds = MPI_PDDN_Dataset(self.PDDNdatasetFolderPath, self.image_size, augment_Flag=self.augment_Flag)

            # Load the dataset with the dataloader
            self.dl = cycle(data.DataLoader(self.ds,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            pin_memory=False,
                                            num_workers=cpu_count()))
            self.disable_training_Flag = False

        self.ema_decay = ema_decay
        self.ema = EMA(self.model, beta=self.ema_decay)
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every

        self.train_lr = train_lr
        self.opt = Adam(self.model.parameters(), lr=self.train_lr)

        self.amp = amp
        self.scaler = GradScaler(enabled=self.amp)
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        self.results_folder = Path(TempResultsFolderPath)
        self.save_and_sample_every = save_and_sample_every
        self.delete_temp_results_Flag = delete_temp_results_Flag

        self.PDDNmodelSummary['PDDNmodelSource'] = self.PDDNmodelFileName

        self.PDDNmodelFileName = PDDNmodelFileName
        # Routine to export the final model at convergence to self.PDDNmodelFileName
        datetime_now = datetime.now()
        datetime_now_str = datetime_now.strftime('%Y_%m_%d_%H%M')
        if self.PDDNmodelFileName is None:
            print(' [wrn] PDDN (Output) Model File Name NOT provided! - Default Applied')
            # Set Default PDDNmodelFileName
            DefaultPDDNFileName = 'PDDN_model_' + datetime_now_str + '.pt'
            self.PDDNmodelFileName = Path.cwd().joinpath(DefaultPDDNFileName)

        self.PDDNlabel = label

        ## Routine to Update Model Summary (to append)
        if self.PDDNmodelSummary['PDDNmodelDataset'] is None:
            self.PDDNmodelSummary['PDDNmodelDataset'] = [self.PDDNdatasetFolderPath]
        else:
            self.PDDNmodelSummary['PDDNmodelDataset'].append(self.PDDNdatasetFolderPath)

        if self.PDDNmodelSummary['PDDNmodelTrainingSession'] is None:
            self.PDDNmodelSummary['PDDNmodelTrainingSession'] = [datetime_now_str]
        else:
            self.PDDNmodelSummary['PDDNmodelTrainingSession'].append(datetime_now_str)

        ## Routine to Update Model Summary Global
        self.PDDNmodelSummary['PDDNmodelLabel'] = self.PDDNlabel
        self.PDDNmodelSummary['PDDNmodelFileName'] = self.PDDNmodelFileName
        self.PDDNmodelSummary['PDDNmodelTrainingSteps'] = self.step
        self.PDDNmodelSummary['PDDNmodelTrainingConvergenceSteps'] = self.train_num_steps
        self.PDDNmodelSummary['PDDNmodelLossType'] = self.model.loss_type
        self.PDDNmodelSummary['PDDNmodelPatchSize'] = [self.model.image_size, self.model.image_size, self.model.channels]
        self.PDDNmodelSummary['PDDNmodelTotTimePoints'] = self.model.num_timesteps

    def save(self, milestone): #REF
        # Routine to save and export the trained model (weights)
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict(),
            'summary': self.PDDNmodelSummary
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load_milestone(self, milestone): #REF
        # Routine to load and import the trained model (weights) previously saved & exported from specific milestone
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema.load_state_dict(data['ema'])
        self.scaler.load_state_dict(data['scaler'])
        try:
            self.PDDNmodelSummary = data['summary']
        except:
            # print(' [wrn] PDDN Model Summary NOT Available!')
            self.PDDNmodelSummary['PDDNmodelTrainingSteps'] = self.step

    def load(self):
        if self.PDDNmodelFileName is None:
            print(' <!> PDDN Model Not Found or NONE provided!')
        else:
            # Routine to load and import the trained model at convergence
            data = torch.load(self.PDDNmodelFileName)

            self.step = data['step']
            self.model.load_state_dict(data['model'])
            self.ema.load_state_dict(data['ema'])
            self.scaler.load_state_dict(data['scaler'])

            try:
                self.PDDNmodelSummary = data['summary']
            except:
                # print(' [wrn] PDDN Model Summary NOT Available!')
                self.PDDNmodelSummary['PDDNmodelTrainingSteps'] = self.step

            print(' >> PDDN Loading Model: ' + str(self.PDDNmodelFileName) + ' -- Complete')

    def printsummary(self): # DEV
        # Function to printout the Model Summary
        print(' >> Model FileName: ' + str(self.PDDNmodelSummary['PDDNmodelFileName']))
        print(' >> Label: ' + str(self.PDDNmodelSummary['PDDNmodelLabel']))
        print(' >> Dataset: ' + str(self.PDDNmodelSummary['PDDNmodelDataset']))
        print(' >> Training Session: ' + str(self.PDDNmodelSummary['PDDNmodelTrainingSession']))
        print(' >> Training Steps: ' + str(self.PDDNmodelSummary['PDDNmodelTrainingSteps']) + '/' + str(self.PDDNmodelSummary['PDDNmodelTrainingConvergenceSteps']))
        print(' >> Convergence Loss: [' + str(self.PDDNmodelSummary['PDDNmodelLossType']) + '] val: ' + str(self.PDDNmodelSummary['PDDNmodelLossValue']))
        print(' >> Source Model: ' + str(self.PDDNmodelSummary['PDDNmodelSource']))
        print(' >> Patch Size: ' + str(self.PDDNmodelSummary['PDDNmodelPatchSize']) + ' <FIXED>')
        print(' >> Total Time-Points: ' + str(self.PDDNmodelSummary['PDDNmodelTotTimePoints']) + ' <FIXED>')
        print(' >> Network Architecture: ')
        print(' ')
        torchsummary.summary(self.model, input_size=(self.model.channels, self.model.image_size, self.model.image_size))
        print(' ')


    """ def train(self): #REF
        # Routine to TRAIN a * NEW MODEL * or 
        # to keep training an existing model (after loading it)

        if self.disable_training_Flag:
            print(' <!> PDDN TRAINING: MISSING Required Inputs! (e.g. Dataset Folder,...) - Abort!')
            return None

        if not self.augment_Flag:
            print(' [wrn] PDDN TRAINING: Data Augmentation is OFF <NOT Recommended>')
            print(' ')

        print(' [TIP] PDDN TRAINING: Please consider removing possible Specular Reflections and other Artifacts')
        print('                      from the Training Polarimetric Dataset PRIOR to training new models!')
        print(' ')

        # Creating the Temporary Results Folder
        self.results_folder.mkdir(exist_ok=True)

        # Clearing Cuda Cache
        torch.cuda.empty_cache()

        # Initialisation of interface for live visualisation of TRAINING
        writer = SummaryWriter()

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir',
                           writer.log_dir,
                           '--samples_per_plugin',
                           "images=0",
                           '--samples_per_plugin',
                           "scalars="+str(self.train_num_steps),
                           "--load_fast=false"])
        url = tb.launch()
        webbrowser.open(url, new=2)

        with tqdm(initial=self.step, total=self.train_num_steps, desc='PDDN Training Model - Loss') as pbar: # ITERATIVE TRAINING

            while self.step < self.train_num_steps:
                for i in range(self.gradient_accumulate_every):
                    # Load a new batch of data for training
                    data = next(self.dl).type(torch.float32).cuda()

                    # TRAIN
                    with autocast(enabled=self.amp):
                        loss = self.model(data)
                        self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    writer.add_scalar("Training Loss", loss.item(), self.step)
                    pbar.set_description('PDDN Training Model - Loss: {:.4f}'.format(loss.item()))

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                self.ema.update()

                # Output intermediate TRAINING Results (after every pre-defined training step interval)
                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.ema.ema_model.eval()

                    # Update Model Summary Local
                    self.PDDNmodelSummary['PDDNmodelLossValue'] = loss.item()
                    self.PDDNmodelSummary['PDDNmodelTrainingSteps'] = self.step

                    with torch.no_grad():
                        milestone = self.step // self.save_and_sample_every # label of the milestone
                        # Generating GIF for fully generative inferential path
                        pbar.set_description('PDDN Training Model - [Save/Sample]')
                        gifData = self.ema.ema_model.sampleGIF()


                    # Visualising in the interactive layout (tensorboard)
                    writer.add_image("Synthetically Generated Batch:",
                                     gifData[:, -1, :, :, :].squeeze(),
                                     self.step,
                                     dataformats="HW")
                    writer.add_video("Batch Diffusion: Full Inferential Trajectory",
                                     gifData,
                                     self.step)
                    # Exporting the SYNTHETIC PATCHES as a PNG file in the results folder (final GIF FRAME)
                    imageio.imwrite(str(self.results_folder / f'batch_sample-{milestone}.png'),
                                    gifData[:, -1, :, :, :].squeeze()) # LAST FRAME - Note: GIF DATA DIMENSION = (B,T,C,H,W)
                    # Export and save the trained model (as per the milestone)
                    self.save(milestone)

                self.step += 1
                pbar.update(1)

        # TRAINING Complete!

        # Print Export PDDNmodelFileName
        print(' ')
        print(' >> PDDN Model (Convergence) Export to: ' + str(self.PDDNmodelFileName))
        # Export (Copy) last Milestone Model (Convergence) to PDDNmodelFileName
        src = str(self.results_folder / f'model-{milestone}.pt')
        trg = str(self.PDDNmodelFileName)
        shutil.copyfile(src, trg)

        # Removal of temp_results_folder 
        if self.delete_temp_results_Flag:
            print(' ')
            print(' >> [REMOVING] Intermediate results folder: ' + str(self.results_folder))
            shutil.rmtree(self.results_folder)

        writer.close()
        print(' ')
        print(' >> PDDN Training: Complete')
        print('**** ***** ***** ***** ****') """


### UTILITY FUNCTIONS (Other HELPERS) by STEFANO

def myUtils_read_cod_data_X3D(input_cod_Filename, isRawFlag=0):
    '''# Function to read (load) stacked 3D data from the camera acquisition or other exported dataset in '.cod' binary format.
	# The data can be calibration data, intensity data (and background noise), or processed data with the libmpMuelMat library.
	#
	# Call: X3D = myUtils_read_cod_data_X3D( input_cod_Filename, [isRawFlag] )
	#
	# * Inputs *
	# input_cod_Filename: string with Global (or local) path to the '.cod' file to import.
	#
	# [isRawFlag]: scalar boolean (0,1) as flag for Raw data (e.g. raw intensities and calibration data) -- default: 0
	#              if 1 -> '.cod' interpreted *with Header* and *Fortran-like ordering*
	#              if 0 -> '.cod' interpreted *without Header* and native *C-like ordering*
 	#
	# * Outputs *
	# X3D: 3D stack of Polarimetric Components of shape shp3 = [dim[0],dim[1],16].
	# 	   NB: the dimensions dim[0] and dim[1] are given by the camera parameters from CamType.
	#
	# NB: CamType is FIXED to 'Stingray IPM2' 
	# NB: shp2 is FIXED to [388, 516]
	#
	'''

    CamType = 'Stingray IPM2' # Default -- Reference for Polarimetric Images
    shp2 = [388, 516] # Default -- Reference for Polarimetric Images

    header_size = 140  # header size of the '.cod' file
    m = 16  # components of the polarimetric data in the '.cod' file

    # Reading the Data (loading) NB: binary data is float, but the output array is *cast* to double
    with open(input_cod_Filename, "rb") as f:
        X3D = np.fromfile(f, dtype=np.single).astype(np.double)
        f.close()

    if isRawFlag:
        # Calibration files WITH HEADER (to be discarded)
        # Reshaping and transposing the shape of the imported array (from Fortran- -> C-like)
        X3D = np.moveaxis(X3D[header_size:].reshape([shp2[1], shp2[0], m]), 0, 1)
    else:
        # Other Files from libmpMuelMat processing WITHOUT HEADER
        # Reshaping the imported array (already C-like)
        X3D = X3D.reshape([shp2[0], shp2[1], m])

    return X3D


def myUtils_padImg3D(Img3D, wlen): # REF
    # Padding image to be denoised/processed in order to avoid border effects
    Rmax = np.ceil(Img3D.shape[0]/wlen) + np.any([(np.remainder(Img3D.shape[0],wlen)/wlen)>0.5,
                                                  (np.remainder(Img3D.shape[0],wlen)/wlen) == 0])
    Cmax = np.ceil(Img3D.shape[1]/wlen) + np.any([(np.remainder(Img3D.shape[1],wlen)/wlen)>0.5,
                                                  (np.remainder(Img3D.shape[1],wlen)/wlen) == 0])

    Rpadsmpls = (Rmax*wlen - Img3D.shape[0])/2
    Cpadsmpls = (Cmax*wlen - Img3D.shape[1])/2

    if int(Rpadsmpls)!=Rpadsmpls:
        raise Exception(' <!> padImg: ODD samples to pad (Rows)! - WIP')
    if int(Cpadsmpls)!=Cpadsmpls:
        raise Exception(' <!> padImg: ODD samples to pad (Cols)! - WIP')

    Img3D_pad = np.pad(Img3D, ((int(Rpadsmpls), int(Rpadsmpls)),
                               (int(Cpadsmpls), int(Cpadsmpls)),
                               (0, 0)), 'reflect')

    return Img3D_pad, Rpadsmpls, Cpadsmpls

def myUtils_getHannWin2D(wlen): #REF
    # Generation of a 2D Hann (weighting) Window for the perfect-reconstruction overlap-add
    hwin = scipy.signal.windows.hann(int(wlen), sym=False).reshape([int(wlen), 1])
    hwin2 = np.kron(hwin, np.transpose(hwin))
    return hwin2

def myUtils_getHannWin2Dov(wlen, hop=0.5): #REF
    # Generation of a 1D Hann (weighting) Window for the perfect-reconstruction overlap-add
    # NOTE: the hop parameter can be changed, but it is recommended: 0.5 (slower processing) or 0.25 (faster processing)
    # NOTE: experiment OTHER values of hop at your OWN RISK
    wlen = np.round(wlen)
    if hop==0.5:
        if wlen % 2 == 0:
            hwin = scipy.signal.windows.hann(int(wlen),
                                             sym=False).reshape([int(wlen), 1])
            ov = wlen/2
            strd = 1
        else:
            hwin = scipy.signal.windows.hann(int(wlen),
                                             sym=True).reshape([int(wlen), 1])
            ov = np.floor(wlen/2)
            strd = 2

    elif hop<0.5:
        if wlen % 2 == 0:
            hsmpls = np.round(2*wlen*hop)
            if hsmpls % 2 == 1:
                hsmpls = hsmpls + 1
            h_tmp = scipy.signal.windows.hann(int(hsmpls),
                                              sym=False).reshape([int(hsmpls), 1])
            hwin = np.concatenate((h_tmp[0:int(hsmpls/2)],
                                   np.ones([int(wlen-hsmpls),1]),
                                   h_tmp[int(hsmpls/2):]), axis=0)
            ov = int(hsmpls/2)
            strd = wlen/2 - ov + 1
        else:
            print(' [wrn] Wlen is ODD and Overlap is <0.5! WIP')
    else:
        print(' [wrn] Overlap is >0.5! WIP')

    hwin2 = np.kron(hwin, np.transpose(hwin))

    return hwin2, ov, strd

def myUtils_genTextPhantom(text_string, sqr_size): #REF
    # This function generates an image layer to add the time-step number to the animated GIF (square patch)
    pil_font = ImageFont.truetype('./Arial.ttf', size=24, encoding='unic')
    canvas = Image.new('I', [sqr_size, sqr_size])
    draw = ImageDraw.Draw(canvas)
    offset = (int(0.025 * sqr_size), int(0.025 * sqr_size))
    draw.text(offset, text_string, font=pil_font)
    return (1 - np.asarray(canvas).astype(np.single))

def myUtils_genLogDynamic(s0, s1, n, roundFlag=True, flipFlag=False): #REF
    # Checking Inputs
    if s0 < 0:
        raise Exception(' <!> Input: "s0" MUST be a scalar non-negative value. The parsed value is: {}'.format(s0))
    if s1 < 0:
        raise Exception(' <!> Input: "s1" MUST be a scalar non-negative value. The parsed value is: {}'.format(s1))
    if n < 1:
        raise Exception(' <!> Input: "n" MUST be a scalar positive value. The parsed value is: {}'.format(n))
    n = np.round(n)

    # Regularising sequence end-points (default: t0<t1 -- reversing if t0>t1)
    if s0 > s1:
        s_temp = s0
        s0 = s1
        s1 = s_temp
        s_temp = True
    else:
        s_temp = False

    # Checking for 0-value s0
    if s0 == 0:
        s0chk = 1
    else:
        s0chk = s0
    # Checking for 0-value s1
    if s1 == 0:
        s1chk = 1
    else:
        s1chk = s1

    # Enabling/Disabling Flip for the Logarithmic Dynamic
    if flipFlag:
        seq = np.cumsum(np.flip(np.diff(np.logspace(np.log10(s0chk), np.log10(s1chk), n + 1))))
    else:
        seq = np.cumsum(np.diff(np.logspace(np.log10(s0chk), np.log10(s1chk), n + 1)))

    # Removing unwanted samples
    seq = np.array([i for i in seq if i > s0])
    seq = np.array([i for i in seq if i < s1])

    # Restoring end-points in the sequence
    if not (seq[0] == s0):
        seq[0] = s0
    if not (seq[-1] == s1):
        seq[-1] = s1

    # Rounding-up the sequence values as per Flag
    if roundFlag:
        seq = np.unique(np.round(seq))
    else:
        seq = np.unique(seq)

    # Filling the gaps
    while seq.size < n:
        if roundFlag:
            seq = np.unique(np.round(np.interp(np.linspace(0, 1, n),
                                               np.linspace(0, 1, seq.size),
                                               seq)))
        else:
            seq = np.unique(np.interp(np.linspace(0, 1, n),
                                      np.linspace(0, 1, seq.size),
                                      seq))

    # Reversing output sequence if input t0 > t1
    if s_temp:
        seq = np.flip(seq)

    return seq

def myUtils_catImg4DToImg2D(img4D): #REF

    if img4D.device.type == 'cuda':
        img4D = np.array(img4D.cpu())
    else:
        img4D = np.array(img4D)

    (img2Dbatch, gridShape) = myUtils_getBatchGrid(img4D)

    for bb in range(0, img4D.shape[0]):  # for each batch
        (r, c) = np.unravel_index(bb, gridShape)
        img3D = np.squeeze(img4D[bb, :, :, :])
        img3D = np.moveaxis(img3D, 0, 2)

        img2Dbatch[r * 4 * img3D.shape[0]:(r + 1) * 4 * img3D.shape[0],
        c * 4 * img3D.shape[1]:(c + 1) * 4 * img3D.shape[1]] = myUtils_catImg3DToImg2D(img3D)

    return img2Dbatch


def myUtils_getBatchGrid(img4D): #REF
    if not (np.sqrt(img4D.shape[1]) == 4):
        raise Exception(
            ' <!> Input: "img4D" MUST have 16 channels in dim[1]. The channels found are: {}'.format(img4D.shape[1]))

    gridShape = np.tile(np.ceil(np.sqrt(img4D.shape[0])), 2).astype(np.uint8)
    img2Dbatch = np.zeros((gridShape[0] * 4 * img4D.shape[2], gridShape[1] * 4 * img4D.shape[3]))

    return img2Dbatch, gridShape


def myUtils_catImg3DToImg2D(img3D, gridCol=4, colWiseFlag=True): #REF
    gridRow = int(np.ceil(img3D.shape[-1]/gridCol))
    img2D = np.zeros((int(img3D.shape[0] * gridRow), int(img3D.shape[1] * gridCol)))
    for i in range(0, img3D.shape[-1]):
        if colWiseFlag:
            (r, c) = np.unravel_index(i, [gridRow, gridCol])
        else:
            (c, r) = np.unravel_index(i, [gridRow, gridCol])

        img2D[r * img3D.shape[0]:(r + 1) * img3D.shape[0],
              c * img3D.shape[1]:(c + 1) * img3D.shape[1]] = img3D[:, :, i]
    return img2D


### >>> MAIN CLASS to IMPORT <<<
class MPI_PDDN():
    '''
    This class instantiates the Polarimetric Denoising Diffusion Network (PDDN) for Mueller Polarimetric Imaging (MPI).
    This version includes classes and methods for:
    
	* Training NEW models from images of polarisation states intensities (Full Mueller Polarimetry)

    * Denoising a polarisation states intensity image of size [H,W,C], with arbitrary 2D size (Height, Width),
      and fixed channels C = 16, as per the full formalism of polarisation states

    * Synthesising Arbitrary-size Polarimetric Images by following a Generative Inferential Trajectory of the PDDN (based on the learned patterns).

	This Codebase has been developed and configured as per [1,2]. Downstream decomposition may leverage tools as in [3].

	References and Credits:
	[1] Moriconi, S., et al. "Near-real-time Mueller polarimetric image processing for neurosurgical intervention", Int J CARS (2024). https://doi.org/10.1007/s11548-024-03090-6
	[2] Moriconi, S., et al. "Denoising Diffusion Network for Real-Time Neurosurgical Mueller Polarimetric Imaging", Medical Image Analysis (2024) - Under Review
	[3] Moriconi, S., "libmpMuelMat Computational tools for MPI", Technical report (2022). https://github.com/stefanomoriconi/libmpMuelMat

    Author (libmpMPIdenoisePDDN): Stefano Moriconi, April 2024, at Inselspital, for HORAO project.
                                  email: stefano.nicola.moriconi@gmail.com
                                  website: https://stefanomoriconi.github.io/mypage/


	USE:
        >> from libmpMPIdenoisePDDN import MPI_PDDN
        >> PDDN = MPI_PDDN([Path_to_PDDN_model.pt])  

	* Training:

		>> PDDN.Train(PDDNdatasetFolderPath, PDDNmodelFileName)

    * Denoising:

        >> imgOUT, time_Elaps = PDDN.Denoise(imgIN, [tstr = 1], [tend = 0], [recursiveFlag = 0])

    * Synthesising:

        >> imgOUT = PDDN.Synthesise([imgH = 256], [imgW = 256])

    * HELP: 

		>> PDDN.Help()
    '''
    def __init__(self,
                 PDDNmodelFileName=None, # CHANGE/INPUT THE PATH TO THE MODEL
                 ):
        super().__init__()
        # Assigning path to Trained PDDN Model
        self.PDDNmodelFileName = PDDNmodelFileName
        # Instantiating UNet Model
        self.PDDN_model = MPI_PDDN_Unet(
            dim=64, # default number dimensions of the UNet model - This is specific of the TRAINED MODEL!
            channels=16, # default number of MPI channels - This is specific of the TRAINED MODEL!
            dim_mults=(1, 2, 4, 8) # default scheme of down/up-sampling - This is specific of the TRAINED MODEL!
        ).cuda()
        # Instantiating Gaussian Diffusion Probabilistic Model
        self.PDDN_diffusion = MPI_PDDN_GaussianDiffusion(
            self.PDDN_model,
            channels=self.PDDN_model.channels, # default number of MPI channels - This is specific of the TRAINED MODEL!
            image_size=128, # default size of patch - This is specific of the TRAINED MODEL!
            timesteps=1000, # default number of Markov-Chain time-steps - This is specific of the TRAINED MODEL!
            loss_type='l1' # default loss type - This is specific of the TRAINED MODEL!
        ).cuda()
        # Instantiating PDDN Trainer (i.e. LOADER)
        self.PDDN_trainer = MPI_PDDN_Trainer(
            self.PDDN_diffusion,
            PDDNdatasetFolderPath=None,
            PDDNmodelFileName=self.PDDNmodelFileName,
        )
        if self.PDDNmodelFileName is None:
            print(' [wrn] PDDN Model: None -- NOT Provided')
        else:
            if Path(self.PDDNmodelFileName).is_file():
                # Load Estimated Model
                print(' >> PDDN Loading Model: ' + str(self.PDDNmodelFileName) + ' -- ...')
                self.PDDN_trainer.load()
            else:
                print(' >> PDDN [Provided Model NOT Found/Loaded]: New Model * TRAINING MODE *')

    def Help(self):
        # Printing the Help! here
        print('*** MPI_PDDN (libmpMPIdenoisePDDN.py) ***')
        print('* Help *')
        print('  This class instantiates the Polarimetric Denoising Diffusion Network (PDDN) for Mueller Polarimetric Imaging (MPI).')
        print('  This version includes methods for:')
        print('  ')
        print('  - Training NEW models from images of polarisation states intensities (Full Mueller Polarimetry)')
        print('  ')
        print('  - Denoising a polarisation states intensity image of size [H,W,C], with arbitrary 2D size (Height, Width),')
        print('    and fixed channels C = 16, as per the full formalism of polarisation states')
        print('  ')
        print('  - Synthesising Arbitrary-size Polarimetric Images by following a Generative Inferential Trajectory of the PDDN (based on the learned patterns).')
        print('  ')
        print('  ')
        print('  This Codebase has been developed and configured as per [1,2]. Downstream decomposition may leverage tools as in [3].')
        print('  ')
        print('  References and Credits:')
        print('  [1] Moriconi, S., et al. "Near-real-time Mueller polarimetric image processing for neurosurgical intervention", Int J CARS (2024). https://doi.org/10.1007/s11548-024-03090-6')
        print('  [2] Moriconi, S., et al. "Denoising Diffusion Network for Real-Time Neurosurgical Mueller Polarimetric Imaging", Medical Image Analysis (2024) - Under Review')
        print('  [3] Moriconi, S., "libmpMuelMat Computational tools for MPI", Technical report (2022). https://github.com/stefanomoriconi/libmpMuelMat')
        print('  ')
        print('  Author (libmpMPIdenoisePDDN): Stefano Moriconi, April 2024, at Inselspital, for HORAO project. ')
        print('                                email: stefano.nicola.moriconi@gmail.com')
        print('                                website: https://stefanomoriconi.github.io/mypage/')
        print('  ')
        print('  ')
        print('  ****************************************')
        print('  USE (Instance Class and Initialisation):')
        print('      >> from libmpMPIdenoisePDDN import MPI_PDDN')
        print('      >> PDDN = MPI_PDDN([Path_to_PDDN_model.pt])')
        print('  ')
        print('  - TRAINING: ')
        print('      >> PDDN.Train(PDDNdatasetFolderPath, PDDNmodelFileName)')
        print('  ')
        print('  - DENOISING: ')
        print('      >> imgOUT, time_Elaps = PDDN.Denoise(imgIN, [tstr = 1], [tend = 0], [recursiveFlag = 0])')
        print('  ')
        print('  - SINTHESYSING: ')
        print('      >> imgOUT = PDDN.Synthesise([imgH = 256], [imgW = 256])')
        print('  ')
        print('  - HELP: ')
        print('      >> PDDN.Help()')
        print('  ')


    def Denoise(self, imgIN, tstr=1, tend=0, hop=0.25, recursiveFlag=False, verboseFlag=True):
        '''Use this function to denoise an intensity image with the trained Polarimetric Denoising Diffusion Network.
        This function integrates an OverLap-and-Add (OLA) routine to filter the input image of arbitrary 2D size (H,W)
        with respect to the reference patch-size used to train the network.

        The image denosing operated by PDDN is configured based on [1,2]. Downstream decomposition may leverage tools as in [3].

        References:
        [1] Moriconi, S., et al. "Near-real-time Mueller polarimetric image processing for neurosurgical intervention", Int J CARS (2024). https://doi.org/10.1007/s11548-024-03090-6
        [2] Moriconi, S., et al. "Denoising Diffusion Network for Real-Time Neurosurgical Mueller Polarimetric Imaging", Medical Image Analysis (2024) - Under Review
        [3] Moriconi, S., "libmpMuelMat Computational tools for MPI", Technical report (2022). https://github.com/stefanomoriconi/libmpMuelMat

        Inputs:
        :param imgIN: input polarimetric intensity image to denoise as np.ndarray [H,W,C=16] C=polarisation states
        :param tstr: scalar integer of the diffusion starting time-step: any ]0,1000] (default: 1 single-time-point)
        :param tend: scalar integer of the diffusion ending time-step (default: 0 aka uncorrupted image)
        :param recursiveFlag: boolean flag to enable the recursive denoising (default: False)
        :param verboseFlag: boolean flag to enable the performance time-stamp (default: True)

        Outputs:

        :return: imgOUT: denoised output of polarimetric intensity image as np.ndarray of size [H,W,C=16]
        :return: time_elaps: list of step-wise computational times and total performance as time_elaps[-1]

        Call:
            imgOUT, time_Elaps = MPI_PDDN.Denoise(imgIN)'''

        # Checking Inputs...
        # imgIN expected of shape: [H,W,C]
        if self.PDDNmodelFileName is None:
            print(' [wrn] PDNN Model NOT Found/Loaded! Denoising with *RANDOM* Network Initialisation!')

        if not (type(imgIN) is np.ndarray):
            raise Exception(
                ' <!> sample_denoise: input "imgIN" expected to be a numpy ndarray -- Found: {}'.format(type(imgIN)))
        imgIN_shp = np.shape(imgIN)
        if not (len(imgIN_shp) == 3):
            raise Exception(
                ' <!> sample_denoise: input "imgIN" expected to have 3D shape [H,W,C] -- Found: {}'.format(imgIN_shp))
        if not (imgIN_shp[2] == self.PDDN_diffusion.channels):
            raise Exception(' <!> sample_denoise: input "imgIN" expected to have', self.PDDN_diffusion.channels,
                            'channels -- Found: {}'.format(imgIN_shp[2]))
        if not (tstr <= self.PDDN_diffusion.num_timesteps):
            raise Exception(' <!> sample_denoise: input "tstr" expected to be <=', self.PDDN_diffusion.num_timesteps,
                            '-- Found: {}'.format(tstr))
        if not (tend >= 0):
            raise Exception(' <!> sample_denoise: input "tend" expected to be >=', 0, '-- Found: {}'.format(tend))

        # printout other Denoising parameters (tstr, tend, recursiveFlag)
        if verboseFlag:
            print('*** PDDN DENOISING FILTER *** ')
            print(' >> PDDN Denoising: total time-points: {:d}'.format(self.PDDN_diffusion.num_timesteps))
            print(' >> PDDN Denoising: start time-point: {:d}'.format(tstr))
            print(' >> PDDN Denoising: final time-point: {:d}'.format(tend))
            print(' >> PDDN Denoising: recursive filtering enabled (2 x cascade): ' + str(recursiveFlag))

        # Denoising
        imgOUT, time_Elaps = self.PDDN_diffusion.img_denoise_OLA(imgIN,
                                                                 tstr=tstr,
                                                                 tend=tend,
                                                                 hop=hop,
                                                                 verboseFlag=verboseFlag)

        if recursiveFlag:
            imgOUT, r_time_Elaps = self.PDDN_diffusion.img_denoise_OLA(imgOUT,
                                                                       tstr=tstr,
                                                                       tend=tend,
                                                                       hop=hop,
                                                                       verboseFlag=verboseFlag)
            time_Elaps = (time_Elaps, r_time_Elaps)


        return imgOUT, time_Elaps

    def Train(self, PDDNdatasetFolderPath=None, PDDNmodelFileName=None, TempResultsFolderPath='./results', train_num_steps=100001, augment_Flag=True, save_and_sample_every=1000, delete_temp_results_Flag=False, train_batch_size=16, train_lr=1e-4, gradient_accumulate_every=2, amp=False, step_start_ema=2000, update_ema_every=10, ema_decay=0.995, label=None):
        '''Use this function to *TRAIN* a new Polarimetric Denoising Diffusion Network.
        This may be wavelength- and tissue-specific, according to the considered Dataset.
        This step requires NVIDIA GPU and Cuda Drivers.

        The training of new PDDN models is configured based on [1,2].
        Note that the training is patch-wise performed by sampling the images in the input Dataset.
        NB: the patch-size is fixed at 128x128xC (square) and it is a property of the Gaussian Diffusion Module Class (MPI_PDDN_GaussianDiffusion)
        NB: the number of time-steps in the Markov chain is fixed at 1000 and it is a property of the Gaussian Diffusion Module Class (MPI_PDDN_GaussianDiffusion)
        In order to change these (and other) parameters, a more in-depth revision of the source code is required.

        References:
        [1] Moriconi, S., et al. "Near-real-time Mueller polarimetric image processing for neurosurgical intervention", Int J CARS (2024). https://doi.org/10.1007/s11548-024-03090-6
        [2] Moriconi, S., et al. "Denoising Diffusion Network for Real-Time Neurosurgical Mueller Polarimetric Imaging", Medical Image Analysis (2024) - Under Review

        Inputs:

        :param PDDNdatasetFolderPath: path to the *DATASET* folder [REQUIRED]
        :param PDDNmodelFileName: filename (full path) of the *OUTPUT* model at convergence [RECOMMENDED]
        :param TempResultsFolderPath: folder path to temporary and intermediate results, i.e. saving/sampling models/patches (default='./results')
        :param train_num_steps: max number of training steps/iterations to converge (default: 100001)
        :param augment_Flag: boolean scalar flag to enable Dataset Augmentation (default: True)
        :param save_and_sample_every: scalar integer for every iteration interval to save/sample intermediate models/patches (default: 1000)
        :param delete_temp_results_Flag: scalar boolean flag to enable/disable the removal of temporary and intermediate results at convergence (default: False)
        :param train_batch_size: scalar integer indicating the training batch size (default: 16)
        :param train_lr: (default:1e-4)
        :param gradient_accumulate_every: (default:2)
        :param amp: (default:False)
        :param step_start_ema: (default:2000)
        :param update_ema_every: (default:10)
        :param ema_decay (default:0.995)
        :param label: User-defined string to identify the Model (default: None)

        Call:

            MPI_PDDN.Train(PDDNdatasetFolderPath, PDDNmodelFileName)
    	'''

        if PDDNdatasetFolderPath is None:
            print(' <!> PDDN Training: MISSING Dataset FolderPath - Abort!')
        else:
            # Passing parameters for Training
            self.PDDN_trainer.passParameters(PDDNdatasetFolderPath,
                                             PDDNmodelFileName,
                                             ema_decay,
                                             train_batch_size,
                                             train_lr,
                                             train_num_steps,
                                             gradient_accumulate_every,
                                             amp,
                                             step_start_ema,
                                             update_ema_every,
                                             save_and_sample_every,
                                             TempResultsFolderPath,
                                             augment_Flag,
                                             delete_temp_results_Flag,
                                             label)

            # Printout Training configurations -- at the init
            print('*** PDDN TRAINING *** ')
            print(' >> Dataset: ' + str(self.PDDN_trainer.PDDNdatasetFolderPath))
            print(' >> Output Model FileName: ' + str(self.PDDN_trainer.PDDNmodelFileName))
            print(' >> Temporary Results Folder: ' + str(self.PDDN_trainer.results_folder.absolute()))
            print(' >> Model Label: ' + str(self.PDDN_trainer.PDDNlabel))
            print(' >> Training Max Iterations: ' + str(self.PDDN_trainer.train_num_steps))
            print(' >> Data Augmentation: ' + str(self.PDDN_trainer.augment_Flag))
            print(' >> Model Save/Sample Every N Iterations: ' + str(self.PDDN_trainer.save_and_sample_every))
            print(' >> Delete Intermediate Results at Convergence: ' + str(self.PDDN_trainer.delete_temp_results_Flag))
            print(' >> Training Batch Size: ' + str(self.PDDN_trainer.batch_size))
            print(' >> Training Learning Rate: ' + str(self.PDDN_trainer.train_lr))
            print(' >> Gradient Accum. Every N Iteration: ' + str(self.PDDN_trainer.gradient_accumulate_every))
            print(' >> Amp Grad Scaler: ' + str(self.PDDN_trainer.amp))
            print(' >> Step Start EMA: ' + str(self.PDDN_trainer.step_start_ema))
            print(' >> Update EMA Every N Iteration: ' + str(self.PDDN_trainer.update_ema_every))
            print(' >> EMA Decay: ' + str(self.PDDN_trainer.ema_decay))
            print(' ')

            # Running Training
            self.PDDN_trainer.train()


    def Synthesise(self, imgH=256, imgW=256, hop=0.25):
        '''Use this function to synthetically generate a polarisation states intensity image of arbitrary size,
        based on the trained Polarimetric Denoising Diffusion Network (PDDN) model.
        The function generates a synthetic tensor of size [imgH x imgW x C] , with C=polarisation states.
        
        NB: The full inferential trajectory from pure Gaussian noise to clean image may take some time.

        The image synthesis operated by PDDN is configured based on [1,2]. Downstream decomposition may leverage tools as in [3].

        References:
        [1] Moriconi, S., et al. "Near-real-time Mueller polarimetric image processing for neurosurgical intervention", Int J CARS (2024). https://doi.org/10.1007/s11548-024-03090-6
        [2] Moriconi, S., et al. "Denoising Diffusion Network for Real-Time Neurosurgical Mueller Polarimetric Imaging", Medical Image Analysis (2024) - Under Review
        [3] Moriconi, S., "libmpMuelMat Computational tools for MPI", Technical report (2022). https://github.com/stefanomoriconi/libmpMuelMat

        Inputs:

        :param imgH: image height in pixels (default: 256)
        :param imgW: image width in pixels (default: 256)
        :param hop: overlap and add ratio among neighbouring patches (default: 0.25)

        Outputs:

        :return: imgOUT: synthetically generated image of polarisation states intensities as np.ndarray

        Call:

            imgOUT = MPI_PDDN.Synthesise(imgH, imgW)
        '''

        if self.PDDNmodelFileName is None:
            print('[wrn] PDNN Model NOT Found/Loaded! Generative Synthesis with *RANDOM* Network Initialisation!')

        print('*** PDDN GENERATIVE SYNTHESIS *** ')
        print(' >> Output Synthetic Polarimetric Image (Polarisation States): [{:d}, {:d}, {:d}]'.format(imgH, imgW, self.PDDN_diffusion.channels))
        imgOUT = self.PDDN_diffusion.img_synth_OLA(imgH=imgH, imgW=imgW, hop=hop)
        return imgOUT


    def ModelSummary(self):
        '''
        Use this function to printout Model-specific information stored at Training.
        Note: This is a relatively new feature - Previous Models may NOT have stored other information.

        Call:

            MPI_PDDN.ModelSummary()

        '''
        print('*** PDDN MODEL SUMMARY ***')
        self.PDDN_trainer.printsummary()