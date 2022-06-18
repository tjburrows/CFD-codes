## 2D Lid-driven Cavity, Lattice Boltzmann Method, MPI

This program solves the 2D lid driven cavity problem with Lattice Boltzmann method, TRT collision, and uses MPI parallelization.  Parameters are set at the top of the file which control Reynolds Number and size of the domain. A binary file is saved, which can be opened and plotted in python with the file `plot.py`.

### Compilation
`mpicxx main.cpp`

### Execution
`mpirun a.out`
