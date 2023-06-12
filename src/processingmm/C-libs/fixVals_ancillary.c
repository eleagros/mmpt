#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "fixVals_ancillary.h"

// Ancillary Function for indexing (ind2subs2D) in 2D -- C-like ordering for indexing
void ind2subs2D(double idx, double* dims, double* r, double* c)
{
	double fracPart;
	fracPart = modf( idx/dims[1] , r );
	*c = round(fracPart*dims[1]);
}

// Ancillary Function for indexing (ind2subs3D) in 3D -- C-like ordering for indexing
void ind2subs3D(double idx, double* dims, double* r, double* c, double* p)
{
	double fracPart;
	fracPart = modf( idx/(dims[1]*dims[2]) , r );
	fracPart = fracPart*dims[1];
	fracPart = modf( fracPart, c);
	*p = round(fracPart*dims[2]);
}

// Ancillary Function for indexing (subs2ind2D) in 2D -- C-like ordering for indexing
double subs2ind2D(double r, double c, double* dims2)
{
	return c + r*dims2[1];
}

// Ancillary Function for indexing (subs2ind3D) in 3D -- C-like ordering for indexing
double subs2ind3D(double r, double c, double p, double* dims3)
{
	return p + c*dims3[2] + r*(dims3[1]*dims3[2]);
}

// Ancillary Function for sorting value-based indexes
struct str
{
    double value;
    double index;
};

int cmpDouble(const void *a, const void *b)
{
    struct str *a1 = (struct str*)a;
    struct str *a2 = (struct str*)b;
    if ((*a1).value > (*a2).value)
        return 1;
    else if ((*a1).value < (*a2).value)
        return -1;
    else
        return 0;
}