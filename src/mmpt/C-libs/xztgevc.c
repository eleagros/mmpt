/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: xztgevc.c
 *
 * MATLAB Coder version            : 5.2
 * C/C++ source code generated on  : 19-Jul-2022 14:47:38
 */

/* Include Files */
#include "xztgevc.h"
//#include "eig4x4cmplx_data.h"
#include "eigL4x4real_data.h"
#include "rt_nonfinite.h"
#include <math.h>
#include <string.h>

/* Function Definitions */
/*
 * Arguments    : const creal_T A[16]
 *                creal_T V[16]
 * Return Type  : void
 */
void xztgevc(const creal_T A[16], creal_T V[16])
{
  creal_T work1[4];
  creal_T work2[4];
  double rworka[4];
  double acoeff;
  double ai;
  double anorm;
  double ascale;
  double brm;
  double d_im;
  double d_re;
  double dmin;
  double salpha_im;
  double salpha_re;
  double scale;
  double temp;
  double xmx;
  double y;
  int b_i;
  int d_re_tmp_tmp;
  int i;
  int j;
  int je;
  int jr;
  int re_tmp;
  int x_tmp_tmp_tmp;
  boolean_T lscalea;
  boolean_T lscaleb;
  rworka[0] = 0.0;
  rworka[1] = 0.0;
  rworka[2] = 0.0;
  rworka[3] = 0.0;
  anorm = fabs(A[0].re) + fabs(A[0].im);
  for (j = 0; j < 3; j++) {
    for (i = 0; i <= j; i++) {
      re_tmp = i + ((j + 1) << 2);
      rworka[j + 1] += fabs(A[re_tmp].re) + fabs(A[re_tmp].im);
    }
    re_tmp = (j + ((j + 1) << 2)) + 1;
    y = rworka[j + 1] + (fabs(A[re_tmp].re) + fabs(A[re_tmp].im));
    if (y > anorm) {
      anorm = y;
    }
  }
  y = anorm;
  if (2.2250738585072014E-308 > anorm) {
    y = 2.2250738585072014E-308;
  }
  ascale = 1.0 / y;
  for (je = 0; je < 4; je++) {
    x_tmp_tmp_tmp = (3 - je) << 2;
    re_tmp = (x_tmp_tmp_tmp - je) + 3;
    xmx = A[re_tmp].re;
    scale = A[re_tmp].im;
    y = (fabs(xmx) + fabs(scale)) * ascale;
    if (1.0 > y) {
      y = 1.0;
    }
    temp = 1.0 / y;
    salpha_re = ascale * (temp * xmx);
    salpha_im = ascale * (temp * scale);
    acoeff = temp * ascale;
    if ((temp >= 2.2250738585072014E-308) &&
        (acoeff < 4.0083367200179456E-292)) {
      lscalea = true;
    } else {
      lscalea = false;
    }
    xmx = fabs(salpha_re) + fabs(salpha_im);
    if ((xmx >= 2.2250738585072014E-308) && (xmx < 4.0083367200179456E-292)) {
      lscaleb = true;
    } else {
      lscaleb = false;
    }
    scale = 1.0;
    if (lscalea) {
      y = anorm;
      if (2.4948003869184E+291 < anorm) {
        y = 2.4948003869184E+291;
      }
      scale = 4.0083367200179456E-292 / temp * y;
    }
    if (lscaleb) {
      y = 4.0083367200179456E-292 / xmx;
      if (y > scale) {
        scale = y;
      }
    }
    if (lscalea || lscaleb) {
      y = acoeff;
      if (1.0 > acoeff) {
        y = 1.0;
      }
      if (xmx > y) {
        y = xmx;
      }
      y = 1.0 / (2.2250738585072014E-308 * y);
      if (y < scale) {
        scale = y;
      }
      if (lscalea) {
        acoeff = ascale * (scale * temp);
      } else {
        acoeff *= scale;
      }
      salpha_re *= scale;
      salpha_im *= scale;
    }
    memset(&work1[0], 0, 4U * sizeof(creal_T));
    work1[3 - je].re = 1.0;
    work1[3 - je].im = 0.0;
    dmin = 2.2204460492503131E-16 * acoeff * anorm;
    y = 2.2204460492503131E-16 * (fabs(salpha_re) + fabs(salpha_im));
    if (y > dmin) {
      dmin = y;
    }
    if (2.2250738585072014E-308 > dmin) {
      dmin = 2.2250738585072014E-308;
    }
    b_i = 2 - je;
    for (jr = 0; jr <= b_i; jr++) {
      re_tmp = jr + x_tmp_tmp_tmp;
      work1[jr].re = acoeff * A[re_tmp].re;
      work1[jr].im = acoeff * A[re_tmp].im;
    }
    work1[3 - je].re = 1.0;
    work1[3 - je].im = 0.0;
    b_i = (int)(((-1.0 - ((-(double)je + 4.0) - 1.0)) + 1.0) / -1.0);
    for (j = 0; j < b_i; j++) {
      i = 2 - (je + j);
      d_re_tmp_tmp = i << 2;
      re_tmp = i + d_re_tmp_tmp;
      d_re = acoeff * A[re_tmp].re - salpha_re;
      d_im = acoeff * A[re_tmp].im - salpha_im;
      if (fabs(d_re) + fabs(d_im) <= dmin) {
        d_re = dmin;
        d_im = 0.0;
      }
      brm = fabs(d_re);
      y = fabs(d_im);
      xmx = brm + y;
      if (xmx < 1.0) {
        scale = fabs(work1[i].re) + fabs(work1[i].im);
        if (scale >= 1.1235582092889474E+307 * xmx) {
          temp = 1.0 / scale;
          re_tmp = 3 - je;
          for (jr = 0; jr <= re_tmp; jr++) {
            work1[jr].re *= temp;
            work1[jr].im *= temp;
          }
        }
      }
      temp = -work1[i].re;
      ai = -work1[i].im;
      if (d_im == 0.0) {
        if (ai == 0.0) {
          y = temp / d_re;
          xmx = 0.0;
        } else if (temp == 0.0) {
          y = 0.0;
          xmx = ai / d_re;
        } else {
          y = temp / d_re;
          xmx = ai / d_re;
        }
      } else if (d_re == 0.0) {
        if (temp == 0.0) {
          y = ai / d_im;
          xmx = 0.0;
        } else if (ai == 0.0) {
          y = 0.0;
          xmx = -(temp / d_im);
        } else {
          y = ai / d_im;
          xmx = -(temp / d_im);
        }
      } else if (brm > y) {
        scale = d_im / d_re;
        xmx = d_re + scale * d_im;
        y = (temp + scale * ai) / xmx;
        xmx = (ai - scale * temp) / xmx;
      } else if (y == brm) {
        if (d_re > 0.0) {
          scale = 0.5;
        } else {
          scale = -0.5;
        }
        if (d_im > 0.0) {
          xmx = 0.5;
        } else {
          xmx = -0.5;
        }
        y = (temp * scale + ai * xmx) / brm;
        xmx = (ai * scale - temp * xmx) / brm;
      } else {
        scale = d_re / d_im;
        xmx = d_im + scale * d_re;
        y = (scale * temp + ai) / xmx;
        xmx = (scale * ai - temp) / xmx;
      }
      work1[i].re = y;
      work1[i].im = xmx;
      if (i + 1 > 1) {
        if (fabs(work1[i].re) + fabs(work1[i].im) > 1.0) {
          temp = 1.0 / (fabs(work1[i].re) + fabs(work1[i].im));
          if (acoeff * rworka[i] >= 1.1235582092889474E+307 * temp) {
            re_tmp = 3 - je;
            for (jr = 0; jr <= re_tmp; jr++) {
              work1[jr].re *= temp;
              work1[jr].im *= temp;
            }
          }
        }
        d_re = acoeff * work1[i].re;
        d_im = acoeff * work1[i].im;
        for (jr = 0; jr < i; jr++) {
          re_tmp = jr + d_re_tmp_tmp;
          work1[jr].re += d_re * A[re_tmp].re - d_im * A[re_tmp].im;
          work1[jr].im += d_re * A[re_tmp].im + d_im * A[re_tmp].re;
        }
      }
    }
    memset(&work2[0], 0, 4U * sizeof(creal_T));
    b_i = 3 - je;
    for (i = 0; i <= b_i; i++) {
      re_tmp = i << 2;
      xmx = work1[i].re;
      scale = work1[i].im;
      work2[0].re += V[re_tmp].re * xmx - V[re_tmp].im * scale;
      work2[0].im += V[re_tmp].re * scale + V[re_tmp].im * xmx;
      work2[1].re += V[re_tmp + 1].re * xmx - V[re_tmp + 1].im * scale;
      work2[1].im += V[re_tmp + 1].re * scale + V[re_tmp + 1].im * xmx;
      work2[2].re += V[re_tmp + 2].re * xmx - V[re_tmp + 2].im * scale;
      work2[2].im += V[re_tmp + 2].re * scale + V[re_tmp + 2].im * xmx;
      work2[3].re += V[re_tmp + 3].re * xmx - V[re_tmp + 3].im * scale;
      work2[3].im += V[re_tmp + 3].re * scale + V[re_tmp + 3].im * xmx;
    }
    xmx = fabs(work2[0].re) + fabs(work2[0].im);
    y = fabs(work2[1].re) + fabs(work2[1].im);
    if (y > xmx) {
      xmx = y;
    }
    y = fabs(work2[2].re) + fabs(work2[2].im);
    if (y > xmx) {
      xmx = y;
    }
    y = fabs(work2[3].re) + fabs(work2[3].im);
    if (y > xmx) {
      xmx = y;
    }
    if (xmx > 2.2250738585072014E-308) {
      temp = 1.0 / xmx;
      V[x_tmp_tmp_tmp].re = temp * work2[0].re;
      V[x_tmp_tmp_tmp].im = temp * work2[0].im;
      V[x_tmp_tmp_tmp + 1].re = temp * work2[1].re;
      V[x_tmp_tmp_tmp + 1].im = temp * work2[1].im;
      V[x_tmp_tmp_tmp + 2].re = temp * work2[2].re;
      V[x_tmp_tmp_tmp + 2].im = temp * work2[2].im;
      V[x_tmp_tmp_tmp + 3].re = temp * work2[3].re;
      V[x_tmp_tmp_tmp + 3].im = temp * work2[3].im;
    } else {
      V[x_tmp_tmp_tmp].re = 0.0;
      V[x_tmp_tmp_tmp].im = 0.0;
      V[x_tmp_tmp_tmp + 1].re = 0.0;
      V[x_tmp_tmp_tmp + 1].im = 0.0;
      V[x_tmp_tmp_tmp + 2].re = 0.0;
      V[x_tmp_tmp_tmp + 2].im = 0.0;
      b_i = ((3 - je) << 2) + 3;
      V[b_i].re = 0.0;
      V[b_i].im = 0.0;
    }
  }
}

/*
 * File trailer for xztgevc.c
 *
 * [EOF]
 */
