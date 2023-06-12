#ifndef ind2subs2D_h__
#define ind2subs2D_h__
/* Function Declarations */
void ind2subs2D(double idx, double* dims, double* r, double* c);
// NB: This is a C-like Ordering function!
#endif

#ifndef ind2subs3D_h__
#define ind2subs3D_h__
/* Function Declarations */
void ind2subs3D(double idx, double* dims, double* r, double* c, double* p);
// NB: This is a C-like Ordering function!
#endif

#ifndef subs2ind2D_h__
#define subs2ind2D_h__
/* Function Declarations */
double subs2ind2D(double r, double c, double* dims2);
// NB: This is a C-like Ordering function!
#endif

#ifndef subs2ind3D_h__
#define subs2ind3D_h__
/* Function Declarations */
double subs2ind3D(double r, double c, double p, double* dims3);
// NB: This is a C-like Ordering function!
#endif

#ifndef cmpDouble_h__
#define cmpDouble_h__
/* Function Declarations */
int cmpDouble(const void *a, const void *b);
#endif