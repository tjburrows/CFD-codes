Description: main.c solves the lid-driven cavity flow problem using an explicit, central difference projection method.  Multiple parameters can be modified at the beginning of the file to alter the problem solved.  SAVETXT controls whether an output file of u and v is created in the current directory.

Reference: Ferziger Computational Methods for Fluid Dynamics, section 7.3.2

Compiling: Dependencies are stdio.h, stdlib.h, math.h, and omp.h.  Make sure to use a C99-enabled compiler. If using gcc, use the -lm flag for the math library and -fopenmp for the openmp library.

Example: gcc main.c -std=c99 -lm -fopenmp

Usage: ./a.exe
