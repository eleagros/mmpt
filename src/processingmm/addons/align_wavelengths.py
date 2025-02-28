import numpy as np
from PIL import Image
import shutil, os
import pickle
import cv2
from processingmm import libmpMuelMat, utils
import matplotlib.pyplot as plt
import imageio
import sys
import SimpleITK as sitk
from tqdm import tqdm

def align_wavelenghts(directories, PDDN, run_all, wlen_to_align = [600]):
    
    wl_to_align = []
    for val in wlen_to_align:
        if val == 550:
            pass
        else:
            wl_to_align.append(val)
    
    if type(wl_to_align) == int:
        wl_to_align = str(wl_to_align)
        align_wavelenght(directories, PDDN, run_all, wl_to_align, imgj_processing = False)
    else:
        assert type(wl_to_align) == list
        for wl in wl_to_align:
            print('Aligning wavelenght: ' + str(wl) + 'nm')
            align_wavelenght(directories, PDDN, run_all, str(wl), imgj_processing = False)
    
def align_wavelenght(directories, PDDN, run_all, wl_to_align, imgj_processing = False):
    data_folder, _ = utils.getAllFolders(directories)

    wl_to_align_with_nm = wl_to_align + 'nm'
    
    for folder in tqdm(data_folder):

        paths = []
        
        if PDDN in ['both', 'no']:
            paths.append(os.path.join(folder, 'raw_data', wl_to_align_with_nm, wl_to_align + '_Intensite_aligned.cod'))
        if PDDN in ['both', 'pddn']:
            if os.path.exists(os.path.join(folder, 'raw_data', wl_to_align_with_nm, wl_to_align + '_Intensite_PDDN.cod')):
                paths.append(os.path.join(folder, 'raw_data', wl_to_align_with_nm, wl_to_align + '_Intensite_PDDN_aligned.cod'))
        
        moving_intensities_path = os.path.join(folder, 'raw_data', wl_to_align_with_nm, wl_to_align + '_Intensite.cod')
        
        exists = 0

        for path in paths:
            exists += os.path.exists(path)
            
        condition_process = (exists < len(paths) or run_all) and os.path.exists(moving_intensities_path)
        
        if condition_process:
            
            fixed_intensities = libmpMuelMat.read_cod_data_X3D(os.path.join(folder, 'raw_data', '550nm', '550_Intensite.cod'), isRawFlag = 1)
            moving_intensities = libmpMuelMat.read_cod_data_X3D(moving_intensities_path, isRawFlag = 1)
            img = (fixed_intensities[:,:,0] / np.max(fixed_intensities[:,:,0]) * 255).astype(np.uint8)
            moving = (moving_intensities[:,:,0] / np.max(moving_intensities[:,:,0]) * 255).astype(np.uint8)
            
            try:
                shutil.rmtree('temp')
            except FileNotFoundError:
                pass
            os.mkdir('temp')

            # copy the images into a temporary folder
            # source = image_550nm_path
            # dst = os.path.join('temp', '550nm_intensity.png')            
            # shutil.copy(source, dst)
            Image.fromarray(img).save(os.path.join('temp', '550nm_intensity.png'))

            # source = image_600nm_path
            # dst = os.path.join('temp', '600nm_intensity.png')
            # shutil.copy(source, dst)
            Image.fromarray(moving).save(os.path.join('temp', wl_to_align_with_nm + '_intensity.png'))

            # run superglue to get matching points between 550nm and 600nm grayscale image
            path_superglue = os.path.join(utils.getSupergluePath(), 'demo_superglue.py')
            try:
                shutil.rmtree('temp_output')
            except FileNotFoundError:
                pass
            os.mkdir('temp_output')
            cmd = r'python3 ' + path_superglue + r' --input temp/ --output_dir temp_output --no_display --resize -1'
            os.system(cmd)

            # recover the matching points and save them into a text file
            try:
                with open(os.path.join('temp_output', 'matches_000000_000001.pickle'), 'rb') as handle:
                    matching_points = pickle.load(handle)
                    points_folder = matching_points[0]
                    points_global = matching_points[1]
            except FileNotFoundError:
                raise ValueError('No matching points found. Please check the superglue installation - it should be located in ./third_party/superglue.')
                
            text = write_mp_fp_txt_format([points_folder, points_global])
            f = open(os.path.join('temp', 'coordinates.txt').replace("\\","/"), "w")
            f.write(text)
            f.close()

            # create and save the indexes image to propagate and run the imagej macro
            to_propagate = create_propagation_img(img)
            if imgj_processing:
                cv2.imwrite(os.path.join('temp', 'to_align_x.tif'), to_propagate[0])
                cv2.imwrite(os.path.join('temp', 'to_align_y.tif'), to_propagate[1])
                os.system('python processing_ij.py ' + os.path.abspath(""))
            else:
                # processing with python wrapper for simple elastix
                resampled_imgs = align_with_sitk_rigid(img, moving, to_propagate, matching_points)
            
            intensities = {}
            
            if PDDN in ['both', 'no']:
                path_intensity = os.path.join(folder, 'raw_data', wl_to_align_with_nm, wl_to_align + '_Intensite.cod')
                if os.path.exists(path_intensity):
                    intensities[path_intensity] = libmpMuelMat.read_cod_data_X3D(path_intensity, isRawFlag = 1)
            if PDDN in ['both', 'pddn']:
                path_intensity = os.path.join(folder, 'raw_data', wl_to_align_with_nm, wl_to_align + '_Intensite_PDDN.cod')
                if os.path.exists(path_intensity):
                    intensities[path_intensity] = libmpMuelMat.read_cod_data_X3D(path_intensity, isRawFlag = 0)
                            
            # load the remapping matrices
            if imgj_processing:
                remapping_x = np.array(Image.open(os.path.join('temp', 'registered_img_brightfield_x.tif')))
                remapping_y = np.array(Image.open(os.path.join('temp', 'registered_img_brightfield_y.tif')))
            else:
                remapping_x = resampled_imgs[0]
                remapping_y = resampled_imgs[1]
                
            intensities_remapped = {}
            for key, val in intensities.items():
                shape_intensities = val.shape
                intensities_remapped[key] = np.zeros(shape_intensities)

            # fill in the refilled matrices
            for idx, x in enumerate(range(shape_intensities[0])):
                for idy, y in enumerate(range(shape_intensities[1])):
                    for id_intensity in range(shape_intensities[2]):
                        idx_remapped = int(remapping_x[idx, idy])
                        idy_remapped = int(remapping_y[idx, idy])
                        if idx_remapped <= 0 or idy_remapped <= 0:
                            pass
                        else:
                            for key, val in intensities.items():
                                intensities_remapped[key][idx, idy, id_intensity] = val[idx_remapped, idy_remapped, id_intensity]

            for key, val in intensities_remapped.items():
                libmpMuelMat.write_cod_data_X3D(val, key.replace('.cod', '_aligned.cod'))

            for key, val in intensities_remapped.items():
                if 'PDDN' in key:
                    pass
                else:
                    initial = libmpMuelMat.read_cod_data_X3D(key, isRawFlag = 1)
                    target = libmpMuelMat.read_cod_data_X3D(key.replace(wl_to_align_with_nm, '550nm').replace(f"{wl_to_align}_", '550_'), isRawFlag = 1)
                    final = libmpMuelMat.read_cod_data_X3D(key.replace('.cod', '_aligned.cod'), isRawFlag = 0)

                    show_output(initial, final, target, folder, wl_to_align, imgj_processing = imgj_processing)

    try:
        shutil.rmtree('temp')
    except FileNotFoundError:
        pass

    try:
        shutil.rmtree('temp_output')
    except FileNotFoundError:
        pass
    
def align_with_sitk(fixed_arr, moving_arr, to_propagate, matching_points):
    fixed_image = sitk.GetImageFromArray(fixed_arr)
    
    # set up the matching points
    fixed_points = matching_points[0]
    moving_points = matching_points[1]
    fixed_landmarks = fixed_points.astype(np.uint16).flatten().tolist()
    moving_landmarks = moving_points.astype(np.uint16).flatten().tolist()
    

    # set up the bspline transform
    transform = sitk.BSplineTransformInitializer(fixed_image, (2, 2), 3)
    landmark_initializer = sitk.LandmarkBasedTransformInitializerFilter()
    landmark_initializer.SetFixedLandmarks(fixed_landmarks)
    landmark_initializer.SetMovingLandmarks(moving_landmarks)
    landmark_initializer.SetBSplineNumberOfControlPoints(8)
    landmark_initializer.SetReferenceImage(fixed_image)
    landmark_initializer.Execute(transform)
    output_transform = landmark_initializer.Execute(transform)

    # resample the moving images
    interpolator = sitk.sitkNearestNeighbor
    moving_images = [sitk.GetImageFromArray(to_propagate[0]), sitk.GetImageFromArray(to_propagate[1]),
                     sitk.GetImageFromArray(moving_arr)]
    resampled_images = []
    for moving_img in moving_images:
        resampled_images.append(sitk.GetArrayFromImage(sitk.Resample(moving_img, fixed_image, output_transform, interpolator, 0)))
                                
    return resampled_images
                                
import SimpleITK as sitk
import numpy as np

def align_with_sitk_rigid(fixed_arr, moving_arr, to_propagate, matching_points):
    fixed_image = sitk.GetImageFromArray(fixed_arr)

    # set up the matching points
    fixed_points = matching_points[0]
    moving_points = matching_points[1]
    fixed_landmarks = fixed_points.astype(np.uint16).flatten().tolist()
    moving_landmarks = moving_points.astype(np.uint16).flatten().tolist()

    # Select the appropriate transform
    transform = sitk.Euler2DTransform()

    # Initialize the transform using the landmarks
    landmark_initializer = sitk.LandmarkBasedTransformInitializerFilter()

    landmark_initializer.SetFixedLandmarks(fixed_landmarks)
    landmark_initializer.SetMovingLandmarks(moving_landmarks)

    output_transform = landmark_initializer.Execute(transform)

    # Resample the moving images
    moving_images = [sitk.GetImageFromArray(to_propagate[0]), sitk.GetImageFromArray(to_propagate[1]), 
                     sitk.GetImageFromArray(moving_arr)]
    resampled_images = []
    
    for moving_img in moving_images:
        resampled_images.append(sitk.GetArrayFromImage(sitk.Resample(moving_img, fixed_image, transform = output_transform)))
        
    # Apply the transform to the moving landmarks
    transformed_moving_points = np.array([
        np.array(output_transform.TransformPoint(tuple(p.astype(np.float64)))) for p in moving_points
    ])
    displacements = np.linalg.norm(moving_points - transformed_moving_points, axis=1)
    
    if np.mean(displacements) > 15:
        Warning('The mean displacement is higher than 15 pixels. The alignment may not be accurate.')
        
    return resampled_images


def write_mp_fp_txt_format(mp_fp):

    # write the header
    text = 'Index\txSource\tySource\txTarget\tyTarget\n'
        
    # write each index, that must be splitted by a tab
    gen = enumerate(zip(mp_fp[0], mp_fp[1]))
        
    for index, (idx, idy) in gen:
        line = str(index) + '\t' + str(round(idx[0])) + '\t' + str(round(idx[1])) + '\t' + str(round(idy[0])) + '\t' + str(round(idy[1])) +'\n' 
        text += line
    return text

def create_propagation_img(img):

    # initializing the map of coordinates in the x and y direction
    to_propagate = [np.zeros((img.shape[0], img.shape[1])).astype('uint16'), 
                    np.zeros((img.shape[0], img.shape[1])).astype('uint16')]

    # filling the map with the coordinates
    for idx, x in enumerate(to_propagate[0]):
        for idy, _ in enumerate(x):
            to_propagate[0][idx, idy] = idx
            to_propagate[1][idx, idy] = idy
    
    return to_propagate


def generate_gif_reconstruction(img1 = None, img2 = None, gif_save_path = None):
    
    # 1. Create a temp folder and save the different blended images inside
    try:
        shutil.rmtree('./tmp_gif')
    except FileNotFoundError:
        pass
    try:
        os.mkdir('./tmp_gif')
    except FileExistsError:
        pass

    gen = enumerate(range(0, 101, 4))
            
    for idx, alpha in gen:
        blended = Image.blend(img1.convert('RGB'), img2.convert('RGB'), alpha=alpha/100)
        blended.save(os.path.join('./tmp_gif', str(idx) + '.png'))
        
        
    # 2. Organize the images in the order in which they should appear
    filenames = os.listdir('./tmp_gif')

    idxs = []
    for f in filenames:
        try:
            idxs.append(int(f.split('.png')[0]))
        except:
            pass
    filenames = [filename for _, filename in sorted(zip(idxs, filenames))]
    filenames = [filenames[0]] * 10 + filenames + [filenames[-1]] * 10 + filenames[::-1]
    
    # 3. Load the images and save the animated gif, remove the temp folder at the end
    images = []
    for filename in filenames:
        if filename == 'final.png':
            pass
        else:
            images.append(imageio.imread(os.path.join('./tmp_gif', filename)))

    duration = 60

    imageio.mimsave(gif_save_path, images, format='GIF', duration=duration, loop = 0)
    try:
        shutil.rmtree('./tmp_gif')
    except PermissionError:
        print('The folder tmp_gif could not be removed. Please remove it manually.')
        

def show_output(initial, final, target, folder, wl_to_align, imgj_processing = False):
    
    os.makedirs(os.path.join(folder, 'annotation'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'annotation', 'alignment'), exist_ok=True)
    
    folder_save_alignment = os.path.join(folder, 'annotation', 'alignment')
    try:
        os.mkdir(folder_save_alignment)
    except:
        pass

    plt.imshow(initial[:,:,0])
    plt.savefig(os.path.join(folder_save_alignment, wl_to_align + '_initial.png'))
    plt.close()
    
    plt.imshow(final[:,:,0])
    plt.savefig(os.path.join(folder_save_alignment, wl_to_align + '_aligned.png'))
    plt.close()
    
    plt.imshow(target[:,:,0])
    plt.savefig(os.path.join(folder_save_alignment, wl_to_align + '_target.png'))
    plt.close()
    
    initial = (initial[:,:,0] / np.max(initial[:,:,0]) * 255).astype(np.uint8)
    final = (final[:,:,0] / np.max(final[:,:,0]) * 255).astype(np.uint8)
    target = (target[:,:,0] / np.max(target[:,:,0]) * 255).astype(np.uint8)
    
    if imgj_processing:
        generate_gif_reconstruction(Image.fromarray(initial), Image.fromarray(target), 
                                    os.path.join(folder_save_alignment, wl_to_align + '_initial_vs_target_imgj.gif'))
        generate_gif_reconstruction(Image.fromarray(final), Image.fromarray(target), 
                                    os.path.join(folder_save_alignment, wl_to_align + '_final_vs_target_imgj.gif'))
    else:
        generate_gif_reconstruction(Image.fromarray(initial), Image.fromarray(target), 
                                    os.path.join(folder_save_alignment, wl_to_align + '_initial_vs_target_python.gif'))
        generate_gif_reconstruction(Image.fromarray(final), Image.fromarray(target), 
                                    os.path.join(folder_save_alignment, wl_to_align + '_final_vs_target_python.gif'))