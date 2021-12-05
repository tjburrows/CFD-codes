// Lid-driven Cavity with explicit, central difference Artificial
// Compressibility method. See Ferziger Computational Methods for Fluid
// Dynamics, section 7.4.3. Travis Burrows.
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Simulation Parameters
#define N 50          // Number of points in X and Y (including BCs)
#define Re 100.0      // Reynolds number
#define Beta 0.5      // compressibility constant
#define SAVETXT 1     // Controls whether to save a text file output
#define dt 5E-3       // dt
#define THRESH 1E-10  // Defines convergence of dp/dt
#define THREADS 4     // parallel threads
#define DEBUG 0       // Prints extra information

// Global Constants
#define H 1.0        // length of side of square domain
#define MAXITER 1E7  // Maximum iterations

// Macros
#define LU(i, j, Ni) (((Ni) * (j)) + (i))  // Look-up function, 2d, Ni x Nj
#define P2(x) ((x) * (x))                  // x^2

// Function Prototypes
double *malloc_vectord(int n1);
void free_vectord(double *a);
void zeros(double *array, int n);
void linspace(double *array, double start, double stop, int num);
void copy(double *source, double *destination, int n);
void datwrite(char filename[], char name1[], double *x, char name2[], double *y,
              char name3[], double *value3, char name4[], double *value4,
              char name5[], double *value5);
void printVector(char name[], double *vector, int n);
void printMatrix(char name[], double *vector, int n1, int n2);
void enforceBCs(double *u, double *v, double *p);
double diff(double *array1, double *array2, int n);
double mean(double *array, int n);

int main(void) {
    // Allocate Memory
    double *U = malloc_vectord(N * N);
    double *V = malloc_vectord(N * N);
    double *P = malloc_vectord(N * N);
    double *Up = malloc_vectord(N * N);
    double *Vp = malloc_vectord(N * N);
    double *Pp = malloc_vectord(N * N);
    double *x = malloc_vectord(N * N);
    double *y = malloc_vectord(N * N);
    double dp, pmean, start, stop, du, Pe;

    // Initialize variables
    double dx = H / (N - 2);
    linspace(x, -dx / 2.0, H + dx / 2.0, N);
    copy(x, y, N);
    char filename[80];
    snprintf(filename, sizeof(filename), "Solution_n=%dRe=%.0f.txt", N, Re);

    if (DEBUG == 1) {
        printf("dx = %.3f\n", dx);
        printVector("x", x, N);
    }

    // Initial Guess
    zeros(U, N * N);
    zeros(V, N * N);
    zeros(P, N * N);
    enforceBCs(U, V, P);

    // Set number of parallel threads
    omp_set_num_threads(THREADS);

    // Iteration
    start = omp_get_wtime();
    for (int k = 0; k < MAXITER; k++) {
        if (DEBUG == 1) {
            printMatrix("U", U, N, N);
            printMatrix("V", V, N, N);
            printMatrix("P", P, N, N);
        }

        // Store previous values
        copy(U, Up, N * N);
        copy(V, Vp, N * N);
        copy(P, Pp, N * N);

        // Start parallel block
#pragma omp parallel
        {
            // Get thread ID
            int tid = omp_get_thread_num();

            // Solve for U, V, P with round-robin parallel scheme over i
            for (int i = 1 + tid; i < N - 1; i += THREADS) {
                for (int j = 1; j < N - 1; j++) {
                    U[LU(i, j, N)] =
                        Up[LU(i, j, N)] +
                        (dt / (Re * P2(dx))) *
                            (Up[LU(i + 1, j, N)] + Up[LU(i - 1, j, N)] +
                             Up[LU(i, j + 1, N)] + Up[LU(i, j - 1, N)] -
                             4.0 * Up[LU(i, j, N)]) -
                        (dt / (2.0 * dx)) *
                            (Pp[LU(i + 1, j, N)] - Pp[LU(i - 1, j, N)]) -
                        (dt * Up[LU(i, j, N)] / dx) *
                            (Up[LU(i + 1, j, N)] - Up[LU(i - 1, j, N)]) -
                        (dt * Vp[LU(i, j, N)] / (2.0 * dx)) *
                            (Up[LU(i, j + 1, N)] - Up[LU(i, j - 1, N)]) -
                        (dt * Up[LU(i, j, N)] / (2.0 * dx)) *
                            (Vp[LU(i, j + 1, N)] - Vp[LU(i, j - 1, N)]);

                    V[LU(i, j, N)] =
                        Vp[LU(i, j, N)] +
                        (dt / (Re * P2(dx))) *
                            (Vp[LU(i + 1, j, N)] + Vp[LU(i - 1, j, N)] +
                             Vp[LU(i, j + 1, N)] + Vp[LU(i, j - 1, N)] -
                             4.0 * Vp[LU(i, j, N)]) -
                        (dt / (2.0 * dx)) *
                            (Pp[LU(i, j + 1, N)] - Pp[LU(i, j - 1, N)]) -
                        (dt * Vp[LU(i, j, N)] / dx) *
                            (Vp[LU(i, j + 1, N)] - Vp[LU(i, j - 1, N)]) -
                        (dt * Up[LU(i, j, N)] / (2.0 * dx)) *
                            (Vp[LU(i + 1, j, N)] - Vp[LU(i - 1, j, N)]) -
                        (dt * Vp[LU(i, j, N)] / (2.0 * dx)) *
                            (Up[LU(i + 1, j, N)] - Up[LU(i - 1, j, N)]);
                    P[LU(i, j, N)] =
                        Pp[LU(i, j, N)] -
                        (dt / (2.0 * dx * Beta)) *
                            (Up[LU(i + 1, j, N)] - Up[LU(i - 1, j, N)] +
                             Vp[LU(i, j + 1, N)] - Vp[LU(i, j - 1, N)]);
                }
            }
        }

        // Ensure boundary conditions are not changed
        enforceBCs(U, V, P);

        // Calculate change in pressure and u
        dp = diff(P, Pp, N * N);
        du = diff(U, Up, N * N);
        pmean = mean(P, N * N);

        // Print iteration info every 500 iterations
        if (k % 500 == 0) {
            printf("\nIteration %d:\n", k);
            printf("dp:\t%.3e\n", dp);
            printf("du:\t%.3e\n", du);
            printf("pav:\t%.3e\n", pmean);
        }

        // Stop if dp/dt is below specified threshold
        if (dp<THRESH & k> 50) break;
    }

    // Print execution time
    stop = omp_get_wtime();
    printf("\nExecution Time:\t%.3e s\n", stop - start);

    // Save a tecplot-formatted file
    if (SAVETXT == 1) {
        datwrite(filename, "x", x, "y", y, "u", U, "v", V, "p", P);
    }

    // If a grid point falls on x = 0.5, print profile
    if (N % 2 == 1) {
        printf("U Values along x=0.5:\n");
        printf("y\tU\n");
        int xint = (N - 1) / 2;
        for (int i = 0; i < N; i++) {
            printf("%.4f\t%.5f\n", y[i], U[LU(xint, i, N)]);
        }
    }

    // Free Memory
    free_vectord(U);
    free_vectord(V);
    free_vectord(P);
    free_vectord(Up);
    free_vectord(Vp);
    free_vectord(Pp);
    free_vectord(x);
    free_vectord(y);

    return 0;
}

// Copies a vector
void copy(double *source, double *destination, int n) {
    for (int i = 0; i < n; i++) {
        destination[i] = source[i];
    }
}

// Returns evenly spaced numbers over a specified interval
void linspace(double *array, double start, double stop, int num) {
    for (int i = 0; i < num; i++) {
        array[i] = start + ((double)i) * (stop - start) / (double)(num - 1);
    }
}

// Allocates memory for 1D array of doubles
double *malloc_vectord(int n1) {
    if (n1 <= 0) {  // Checks for invalid inputs
        printf("Invalid input into malloc_vectord\n");
    } else {
        double *mat = malloc(n1 * sizeof(double));
        if (mat == NULL) printf("Error allocating memory!");
        return mat;
    }
}

// Frees memory for 1D double array
void free_vectord(double *a) {
    if (a == NULL) printf("Error: Null input in free_vectord");
    free((void *)a);
}

// Assigns zeros to a vector
void zeros(double *array, int n) {
    for (int i = 0; i < n; i++) {
        array[i] = 0.0;
    }
}

// Writes Tecplot file
void datwrite(char filename[], char name1[], double *x, char name2[], double *y,
              char name3[], double *value3, char name4[], double *value4,
              char name5[], double *value5) {
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    fprintf(f,
            "TITLE=\"%s\" VARIABLES=\"%s\", \"%s\", \"%s\", \"%s\", \"%s\" "
            "ZONE T=\"%s\" I=%d J=%d F=POINT\n",
            filename, name1, name2, name3, name4, name5, filename, N - 2,
            N - 2);
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            fprintf(f, "%.10e, %.10e, %.10e, %.10e, %.10e\n", x[i], y[j],
                    value3[LU(i, j, N)], value4[LU(i, j, N)],
                    value5[LU(i, j, N)]);
        }
    }
    fclose(f);
}

// Prints a vector
void printVector(char name[], double *vector, int n) {
    printf("%s:\t", name);
    for (int i = 0; i < n; i++) {
        printf("%.3f ", vector[i]);
    }
    printf("\n");
}

// Prints a matrix
void printMatrix(char name[], double *vector, int n1, int n2) {
    printf("%s:\n", name);
    for (int j = 0; j < n2; j++) {
        for (int i = 0; i < n1; i++) {
            printf("%.4f\t", vector[LU(i, j, n1)]);
        }
        printf("\n");
    }
    printf("\n");
}

// Enforces Lid driven cavity flow boundary conditions
void enforceBCs(double *u, double *v, double *p) {
    for (int i = 0; i < N; i++) {
        // Bottom Surface
        u[LU(i, 0, N)] = -u[LU(i, 1, N)];
        v[LU(i, 0, N)] = -v[LU(i, 1, N)];
        p[LU(i, 0, N)] = p[LU(i, 1, N)];

        // Top Surface
        u[LU(i, N - 1, N)] = 2.0 - u[LU(i, N - 2, N)];
        v[LU(i, N - 1, N)] = -v[LU(i, N - 2, N)];
        p[LU(i, N - 1, N)] = p[LU(i, N - 2, N)];

        // Left Surface
        u[LU(0, i, N)] = -u[LU(1, i, N)];
        v[LU(0, i, N)] = -v[LU(1, i, N)];
        p[LU(0, i, N)] = p[LU(1, i, N)];

        // Right Surface
        u[LU(N - 1, i, N)] = -u[LU(N - 2, i, N)];
        v[LU(N - 1, i, N)] = -v[LU(N - 2, i, N)];
        p[LU(N - 1, i, N)] = p[LU(N - 2, i, N)];
    }
}

// Returns L2 norm of difference of two arrays
double diff(double *array1, double *array2, int n) {
    double difference = 0;
    for (int i = 0; i < n; i++) {
        difference += P2(array1[i] - array2[i]);
    }
    return sqrt(difference);
}

// Returns mean of an array
double mean(double *array, int n) {
    double average = 0;
    for (int i = 0; i < n; i++) {
        average += array[i] / n;
    }
    return average;
}
