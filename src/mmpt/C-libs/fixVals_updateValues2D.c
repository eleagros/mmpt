#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "fixVals_ancillary.h"
#include "fixVals_updateValues2D.h"

// Functions Implementations
void fixVals_updateValues2D(double* img2D, double* dims2, double idx2)
{
	double r, c, rr, cc, nn2idx;
	double acc = 0;
	double val = 0;

	double d = sqrt(2);
	double nnWgt[] = {d, 1.0, d, 1.0, 0.0, 1.0, d, 1.0, d};	

	int lin3x3 = 9;
	int nr[] = {-1,-1,-1,0,0,0,1,1,1};
	int nc[] = {-1,0,1,-1,0,1,-1,0,1};

	ind2subs2D(idx2, dims2, &r, &c); // Recover the subs indices for the indexed pixel
	for(int l=0;l<lin3x3;l++) // For each pixel in its 3x3 neighbourhood (2D)
	{
		rr = r + nr[l]; // Row of the neighbouring pixel
		cc = c + nc[l]; // Column of the neighbouring pixel
		if((rr >= 0) && (rr < dims2[0]) && (cc >= 0) && (cc < dims2[1])) // Check the neighbouring pixel is in the image domain (2D)
		{
			nn2idx = subs2ind2D(rr, cc, dims2); // Recover the indexed position of the neighbouring pixel (2D)
			if(!isinf(img2D[(int)nn2idx]) && !isnan(img2D[(int)nn2idx])) // If the neighbouring pixel has finite value in the image 2D 
			{
				val = val + (img2D[(int)nn2idx] * nnWgt[l]);
               	acc = acc + nnWgt[l];
			}
		}
	}
	// Assign Values: imgage 3D
	img2D[(int)idx2] = val/acc;
}