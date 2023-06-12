#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "fixVals_ancillary.h"
#include "fixVals_updateValues2D.h"
#include "fixVals_restoreValues2D.h"

struct str
{
    double value;
    double index;
};

void fixVals_restoreValues2D(double* img2D, double* wgt2D, double* dims2, double* idxList, int lenList)
{
	// Sorting indexes accorting to weights
	struct str objects[lenList];
	for (int i=0;i<lenList;i++)
    {
        objects[i].value = wgt2D[(int)idxList[i]];
        objects[i].index = idxList[i];
    }
    qsort(objects, lenList, sizeof(objects[0]), cmpDouble);

    // Main for cycle
    /*STRICTLY SEQUENTIAL! -- NOT PARALLEL: the sequence of indices to be updated must follow the sorted list in objects!*/
    for (int i=0;i<lenList;i++) 
    {
    	fixVals_updateValues2D(img2D, dims2, objects[i].index);
    }
}