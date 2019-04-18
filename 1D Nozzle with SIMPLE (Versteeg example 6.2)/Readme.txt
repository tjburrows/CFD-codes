Description: main.cpp solves the steady centerline flow through a two-dimensional nozzle using SIMPLE and SIMPLER, with upwind differencing, a 1D staggered grid, and the use of the AMGCL library (https://github.com/ddemidov/amgcl) for advanced multigrid-enabled linear solvers.  I recommend trying this out, as it provides a major speedup of the solve!  An in-depth description of this problem and the assumptions made is in the reference below.  It is planned to include these details in this readme in the future.

Reference: Versteeg, An Introduction to Computational Fluid Dynamics, Example 6.2

Compiling without AMGCL: Dependencies are iostream, math.h, string.h, std::vector, std::array, and omp.h.  Make sure to use a C++11-enabled compiler. If using g++, use the -lm flag for the math library and -fopenmp for the openmp library.

Compiling with AMGCL: This requires the AMGCL header files, as well as the Boost library, which is a dependency of AMGCL.  Include / link both of these to your compiler.
