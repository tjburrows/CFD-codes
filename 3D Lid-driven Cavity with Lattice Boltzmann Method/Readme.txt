Description: main.c solves the 3D lid-driven cavity flow problem using Lattice Boltzmann method with single relaxation time (SRT / BGK) and two relaxation time (TRT).  This code can solve the flow for a cavity driven in a parallel for diagonal (corner to corner) direction.  Multiple parameters can be modified at the beginning of the file to alter the problem solved.  Size of grid, Reynolds number, and convergence threshold. can all be changed.  SAVETXT controls whether an output file of u and v is created in the current directory.

Reference: Kruger The Lattice Boltzmann Method

Compiling: Dependencies are stdio.h, stdlib.h, math.h, and omp.h.  Make sure to use a C99-enabled compiler. If using gcc, use the -lm flag for the math library and -fopenmp for the openmp library.

Example: gcc main.c -std=c99 -lm -fopenmp

Usage: ./a.exe
