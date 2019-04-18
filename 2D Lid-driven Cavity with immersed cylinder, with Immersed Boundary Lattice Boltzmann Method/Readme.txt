Description: main.cpp solves the lid-driven cavity flow problem using Lattice Boltzmann method with single relaxation time (SRT / BGK), two relaxatino time (TRT), and multiple relaxation time (MRT).  A 2D cylinder is embedded, and is solved with the implicit immersed boundary method.  To compute matrix inverses, the Eigen library is used (http://eigen.tuxfamily.org).  Boundary condition methods include half-way bounce back, non-equilibrium bounce-back, and non-equilibrium extrapolation.

Reference: Kruger The Lattice Boltzmann Method, section 11.4.4.2

Compiling: Dependencies are iostream, vector, math.h, omp.h, and Eigen/Dense.  Make sure to use a C++11-enabled compiler. If using g++, use the -lm flag for the math library and -fopenmp for the openmp library.

Example: g++ main.cpp -std=c++11 -lm -fopenmp -O3 -I/path/to/eigen

Usage: ./a.exe
