#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "fixVals_ancillary.h"
#include "fixVals_updateValues3D.h"

// Functions Implementations
void fixVals_updateValues3D(double* img3D, double* dims3, double idx2)
{
	double dims2[2] = {dims3[0],dims3[1]};
	double r, c, rr, cc, nn3idx, idx3;

	double d = sqrt(2);
	double nnWgt[] = {d, 1.0, d, 1.0, 0.0, 1.0, d, 1.0, d};
	
	double acc = 0;
	double val[(int)dims3[2]];
	for (int j=0;j<dims3[2];j++) // Resetting for each channel
	{val[j] = 0;}

	int lin3x3 = 9;
	int pp;
	int nr[] = {-1,-1,-1,0,0,0,1,1,1};
	int nc[] = {-1,0,1,-1,0,1,-1,0,1};

	ind2subs2D(idx2, dims2, &r, &c); // Recover the subs indices for the indexed pixel
	for(int l=0;l<lin3x3;l++) // For each pixel in its 3x3 neighbourhood (2D)
	{
		rr = r + nr[l]; // Row of the neighbouring pixel
		cc = c + nc[l]; // Column of the neighbouring pixel
		if((rr >= 0) && (rr < dims2[0]) && (cc >= 0) && (cc < dims2[1])) // Check the neighbouring pixel is in the image domain (2D)
		{
			for(pp=0;pp<dims3[2];pp++) // For each channel (3D)
			{
				nn3idx = subs2ind3D(rr, cc, (double)pp, dims3); // Recover the indexed position of the neighbouring pixel in 3D (channel-wise)
				if(!isinf(img3D[(int)nn3idx]) && !isnan(img3D[(int)nn3idx])) // If the neighbouring pixel has finite value in the image 3D (channel-wise)
				{
					val[pp] = val[pp] + (img3D[(int)nn3idx] * nnWgt[l]);
               		if (pp == 0) // only once for all
               		{acc = acc + nnWgt[l];}
				}
			}
		}
	}
	// Assign Values: imgage 3D
	for(pp=0;pp<dims3[2];pp++) // For each channel (3D)
	{
		idx3 = subs2ind3D(r, c, (double)pp, dims3); // Recover the indexed position of the neighbouring pixel in 3D (channel-wise)
		img3D[(int)idx3] = val[pp]/acc;
	}
}
