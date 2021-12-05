// 2D lid-driven Cavity flow with explicit, central-difference projection
// method. See Ferziger Computational Methods for Fluid Dynamics, section 7.3.2
// Travis Burrows

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Simulation Parameters
#define N 25         // Number of points in X and Y (including BCs)
#define Re 100.0     // Reynolds number
#define SAVETXT 1    // Controls whether to save a text file output
#define dt 0.04      // dt
#define THRESH 1E-8  // Defines convergence of dp/dt
#define DEBUG 1      // Prints extra information
#define OMEGA 1.95   // SOR Relaxation factor
#define PARALLEL 1   // toggles parallel spatial loops

// Global Constants
#define H 1.0        // length of side of square domain
#define MAXITER 1E6  // Maximum iterations

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
void datwrite1(char filename[], char name1[], double *x, char name2[],
               double *y, char name3[], double *value3, int nx, int ny);

int main(void) {
    /* U has gridpoints on side boundaries, and sandwiching top / bottom
      boundaries (N-1 x N) V has gridpoints on top / bottom boundaries, and
      sandwiching side boundaries (N x N-1). P has gridpoints sandwiching all
      boundaries (N x N). All grids have same dx, and are offset from each
      other by dx/2. U grid is vertical midpoints of P, V grid is horizontal
      midpoints of P. */

    // Allocate Memory
    double *U = malloc_vectord(N * (N - 1));
    double *V = malloc_vectord(N * (N - 1));
    double *P = malloc_vectord(N * N);
    double *Up = malloc_vectord(N * (N - 1));
    double *Vp = malloc_vectord(N * (N - 1));
    double *Pp = malloc_vectord(N * N);
    double *Pp2 = malloc_vectord(N * N);
    double *xP = malloc_vectord(N * N);
    double *yP = malloc_vectord(N * N);
    double *Hx = malloc_vectord(N * (N - 1));
    double *Hy = malloc_vectord(N * (N - 1));
    double Ue, Uw, Un, Us, Vn, Vs, UP, UE, UW, US, UN, Vne, Vnw, Vse, Vsw, VP,
        VE, VW, VN, VS, Une, Unw, Use, Usw, Ve, Vw;
    double du, dv, dp, start, stop;

    // Initialize variables
    double dx = H / (N - 2);
    double dy = dx;
    linspace(xP, -dx / 2.0, H + dx / 2.0, N);
    copy(xP, yP, N);
    char filename[80];
    snprintf(filename, sizeof(filename), "Solution_n=%dRe=%.0f.txt", N, Re);

    if (DEBUG == 1) {
        printf("dx = %.3f\n", dx);
        printVector("x_P", xP, N);
    }

    // Initial Guess
    zeros(U, N * (N - 1));
    zeros(V, N * (N - 1));
    zeros(Hx, N * (N - 1));
    zeros(Hy, N * (N - 1));
    zeros(P, N * N);
    enforceBCs(U, V, P);

    // Set number of parallel threads
    int maxthreads = 1;
    if (PARALLEL == 1) maxthreads = omp_get_max_threads();
    printf("Threads used:\t%d\n", maxthreads);
    omp_set_num_threads(maxthreads);

    // Main iteration
    for (int k = 0; k < MAXITER; k++) {
        start = omp_get_wtime();
        copy(U, Up, N * (N - 1));
        copy(V, Vp, N * (N - 1));
        copy(P, Pp, N * N);

        // Find Hx (on u grid)
#pragma omp parallel for private(Ue, Uw, Un, Us, Vn, Vs, UP, UE, UW, US, UN, \
                                 Vne, Vnw, Vse, Vsw) collapse(2)
        for (int i = 1; i < N - 2; i++) {
            for (int j = 1; j < N - 1; j++) {
                UP = Up[LU(i, j, N - 1)];
                UN = Up[LU(i, j + 1, N - 1)];
                US = Up[LU(i, j - 1, N - 1)];
                UE = Up[LU(i + 1, j, N - 1)];
                UW = Up[LU(i - 1, j, N - 1)];
                Ue = 0.5 * (UE + UP);
                Uw = 0.5 * (UW + UP);
                Un = 0.5 * (UN + UP);
                Us = 0.5 * (US + UP);
                Vne = Vp[LU(i + 1, j, N)];
                Vnw = Vp[LU(i, j, N)];
                Vse = Vp[LU(i + 1, j - 1, N)];
                Vsw = Vp[LU(i, j - 1, N)];
                Vn = 0.5 * (Vne + Vnw);
                Vs = 0.5 * (Vse + Vsw);
                Hx[LU(i, j, N - 1)] =
                    (1.0 / Re) * ((UE + UW - 2 * UP) / P2(dx) +
                                  (UN + US - 2 * UP) / P2(dy)) -
                    ((P2(Ue) - P2(Uw)) / dx + (Un * Vn - Us * Vs) / dy);
            }
        }

        // Find Hy (on v grid)
#pragma omp parallel for private(VP, VE, VW, VN, VS, Une, Unw, Use, Usw, Ve, \
                                 Vw, Vn, Vs, Ue, Uw) collapse(2)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 2; j++) {
                VP = Vp[LU(i, j, N)];
                VN = Vp[LU(i, j + 1, N)];
                VS = Vp[LU(i, j - 1, N)];
                VE = Vp[LU(i + 1, j, N)];
                VW = Vp[LU(i - 1, j, N)];
                Ve = 0.5 * (VE + VP);
                Vw = 0.5 * (VW + VP);
                Vn = 0.5 * (VN + VP);
                Vs = 0.5 * (VS + VP);
                Une = Up[LU(i, j + 1, N - 1)];
                Unw = Up[LU(i - 1, j + 1, N - 1)];
                Use = Up[LU(i, j, N - 1)];
                Usw = Up[LU(i - 1, j, N - 1)];
                Ue = 0.5 * (Une + Use);
                Uw = 0.5 * (Unw + Usw);
                Hy[LU(i, j, N)] =
                    (1.0 / Re) * ((VE + VW - 2 * VP) / P2(dx) +
                                  (VN + VS - 2 * VP) / P2(dy)) -
                    ((P2(Vn) - P2(Vs)) / dy + (Ue * Ve - Uw * Vw) / dx);
            }
        }

        // Solve for Pressure
        double PE, PW, PN, PS, Hxe, Hxw, Hyn, Hys;
        for (int kk = 0; kk < MAXITER; kk++) {
            copy(P, Pp2, N * N);
#pragma omp parallel for private(PE, PW, PN, PS, Hxe, Hxw, Hyn, Hys) collapse(2)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    PE = P[LU(i + 1, j, N)];
                    PW = P[LU(i - 1, j, N)];
                    PN = P[LU(i, j + 1, N)];
                    PS = P[LU(i, j - 1, N)];
                    Hxe = Hx[LU(i, j, N - 1)];
                    Hxw = Hx[LU(i - 1, j, N - 1)];
                    Hyn = Hy[LU(i, j, N)];
                    Hys = Hy[LU(i, j - 1, N)];
                    P[LU(i, j, N)] =
                        (1.0 - OMEGA) * Pp2[LU(i, j, N)] +
                        OMEGA * (-1.0 / (2.0 * (P2(dx) + P2(dy)))) *
                            (dx * P2(dy) * (Hxe - Hxw) +
                             dy * P2(dx) * (Hyn - Hys) - P2(dy) * (PE + PW) -
                             P2(dx) * (PN + PS));
                }
            }
            enforceBCs(U, V, P);
            dp = diff(P, Pp2, N * N);
            if (dp < 1E-10) break;
        }

        // Rezero Pressure mean
        double pmean = mean(P, N * N);
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                P[LU(i, j, N)] -= pmean;
            }
        }

        // Calculate u
        double Pe, Pw;
        for (int i = 1; i < N - 2; i++) {
            for (int j = 1; j < N - 1; j++) {
                Pe = P[LU(i + 1, j, N)];
                Pw = P[LU(i, j, N)];
                U[LU(i, j, N - 1)] =
                    Up[LU(i, j, N - 1)] +
                    dt * (Hx[LU(i, j, N - 1)] - (Pe - Pw) / dx);
            }
        }

        // Calculate v
        double Pn, Ps;
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 2; j++) {
                Pn = P[LU(i, j + 1, N)];
                Ps = P[LU(i, j, N)];
                V[LU(i, j, N)] =
                    Vp[LU(i, j, N)] + dt * (Hy[LU(i, j, N)] - (Pn - Ps) / dy);
            }
        }

        // Enforce BCs
        enforceBCs(U, V, P);

        // Determine convergence
        du = diff(U, Up, N * (N - 1)) / dt;
        dv = diff(V, Vp, N * (N - 1)) / dt;
        dp = diff(P, Pp, N * N) / dt;

        int printint = 50;
        if (DEBUG) printint = 1;
        if (k % printint == 0) {
            printf("\nIteration %d:\n", k);
            printf("du:\t%.3e\n", du);
            printf("dv:\t%.3e\n", dv);
            printf("dp:\t%.3e\n", dp);
        }
        stop = omp_get_wtime();
        printf("Time:\t%.3e s\n", stop - start);
        // Test if converged
        if (dp < THRESH) {
            break;
        }
    }

    // Export non-interpolated grids
    double *xU = malloc_vectord(N - 1);
    for (int i = 0; i < N - 1; i++) {
        xU[i] = 0.5 * (xP[i] + xP[i + 1]);
    }

    // Export tecplot files
    if (SAVETXT == 1) {
        snprintf(filename, sizeof(filename), "ugrid_n=%dRe=%.0f.dat", N, Re);
        datwrite1(filename, "x", xU, "y", yP, "u", U, N - 1, N);
        snprintf(filename, sizeof(filename), "vgrid_n=%dRe=%.0f.dat", N, Re);
        datwrite1(filename, "x", xP, "y", xU, "v", V, N, N - 1);
    }

    // Free memory
    free_vectord(U);
    free_vectord(V);
    free_vectord(P);
    free_vectord(Up);
    free_vectord(Vp);
    free_vectord(Pp);
    free_vectord(Pp2);
    free_vectord(xP);
    free_vectord(yP);
    free_vectord(Hx);
    free_vectord(Hy);
    free_vectord(xU);

    // Return
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

// Writes Tecplot file
void datwrite1(char filename[], char name1[], double *x, char name2[],
               double *y, char name3[], double *value3, int nx, int ny) {
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    fprintf(f,
            "TITLE=\"%s\" VARIABLES=\"%s\", \"%s\", \"%s\" ZONE T=\"%s\" I=%d "
            "J=%d F=POINT\n",
            filename, name1, name2, name3, filename, nx - 2, ny - 2);
    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            fprintf(f, "%.10e, %.10e, %.10e,\n", x[i], y[j],
                    value3[LU(i, j, nx)]);
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
    /*  U has gridpoints on side boundaries, and sandwiching top / bottom
      boundaries (N-1 x N). V has gridpoints on top / bottom boundaries, and
      sandwiching side boundaries (N x N-1). P has gridpoints sandwiching all
      boundaries. */

    for (int i = 0; i < N; i++) {
        // Bottom Surface
        v[LU(i, 0, N)] = 0.0;
        p[LU(i, 0, N)] = p[LU(i, 1, N)];
        if (i < N - 1) u[LU(i, 0, N - 1)] = -u[LU(i, 1, N - 1)];

        // Top Surface
        v[LU(i, N - 2, N)] = 0.0;
        p[LU(i, N - 1, N)] = p[LU(i, N - 2, N)];
        if (i < N - 1) u[LU(i, N - 1, N - 1)] = 2.0 - u[LU(i, N - 2, N - 1)];

        // Left Surface
        u[LU(0, i, N - 1)] = 0.0;
        p[LU(0, i, N)] = p[LU(1, i, N)];
        if (i < N - 1) v[LU(0, i, N)] = -v[LU(1, i, N)];

        // Right Surface
        u[LU(N - 2, i, N - 1)] = 0.0;
        p[LU(N - 1, i, N)] = p[LU(N - 2, i, N)];
        if (i < N - 1) v[LU(N - 1, i, N)] = -v[LU(N - 2, i, N)];
    }
}

// Returns L2 norm of difference of two arrays
double diff(double *array1, double *array2, int n) {
    double difference = 0;
    for (int i = 0; i < n; i++) {
        difference += P2(array1[i] - array2[i]);
    }
    return sqrt(difference / (double)n);
}

// Returns mean of an array
double mean(double *array, int n) {
    double average = 0;
    for (int i = 0; i < n; i++) {
        average += array[i] / n;
    }
    return average;
}
