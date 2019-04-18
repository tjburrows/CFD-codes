Description: main.cpp solves the lid-driven cavity flow problem using Lattice Boltzmann method with single relaxation time (SRT / BGK), two relaxatino time (TRT), and multiple relaxation time (MRT).  Boundary condition methods include half-way bounce back, non-equilibrium bounce-back, and non-equilibrium extrapolation.  Multiple parameters can be modified at the beginning of the file to alter the problem solved.  The header file LBMClass.h contains all functions required for this program.

This code indicates that it can solve Couette or Poiseuille flow, but I think that functionality is broken.

Reference: Kruger The Lattice Boltzmann Method

Compiling: Dependencies are iostream, vector, math.h, and omp.h.  Make sure to use a C++11-enabled compiler. If using g++, use the -lm flag for the math library and -fopenmp for the openmp library.

Example: g++ main.cpp -std=c++11 -lm -fopenmp -O3

Usage: ./a.exe
