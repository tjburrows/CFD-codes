#include <math.h>
#include <mpi.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

//  Simulation Parameters
const double Re = 100;       // Reynolds number
const int N = 100;           // Width
const int M = 300;           // Height
const double Ulid = 0.1;     // Velocity of lid
const bool SAVE = true;      // Whether to save data
const double THRESH = 2E-9;  // Stopping threshold

//  Global Constants
const int Q = 9;
const double MAXITER = 1E6;
const double NU = Ulid * N / Re;
const double TAU = NU * 3.0 + 0.5;
const double OMEGA = 1.0 / TAU;
const double MAGIC = 0.25;
const double OMEGAm = 1.0 / (0.5 + (MAGIC / (TAU - 0.5)));
const double w0 = 4.0 / 9.0;
const double wS = 1.0 / 9.0;
const double wd = 1.0 / 36.0;

const int prntInt = 1e3;  // Iteration interval to print status
const int nReq = 6;       // Number of MPI requests

//  Global Variables
int BOTTOM{}, TOP{}, INTERNAL{}, jStartWoGhosts{}, jEndWoGhosts{},
    jEndWGhosts{};
double df0{}, start{}, stop{};
MPI_Request requestsUp[nReq]{};
MPI_Request requestsDown[nReq]{};
MPI_Status requestStatsUp[nReq]{};
MPI_Status requestStatsDown[nReq]{};

//  Function Prototypes
typedef vector<double> vecType;
int LU2(int i, int j);
int LU3(int i, int j, int k);
void macroVars(const vecType f, vecType &u, vecType &v, vecType &rho,
               const int ystart, const int yend, double &diff);
bool convergence(double &diff, const int rank, const int t);
void exchangeDataUpward(vecType &f, const int rank, const int above,
                        const int below, const int nProcs);
void exchangeDataDownward(vecType &f, const int rank, const int above,
                          const int below, const int nProcs);
void outputScalar(const string scalarName, const vecType scalar,
                  const int rank);
void streamCollide(vecType &dist, const int yStart, const int yEnd);

int main(int argc, char *argv[]) {
    //  Initialize MPI
    int rank{}, nProcs{};
    MPI_Init(&argc, &argv);

    //  Save Rank of this process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //  Save number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    if (nProcs < 3 && rank == 0) {
        cout << "Error.  Must have 3 or more computing processors." << endl;
        exit(1);
    }

    if (rank == 0) {
        cout << "MPI LBM Simulation of Lid Driven Cavity" << endl;
        cout << "Domain Size: " << N << " x " << M << endl;
        cout << "Re\t= " << Re << endl;
        cout << "nu\t= " << NU << endl;
        cout << "tau\t= " << TAU << endl;
        cout << "Umax\t= " << Ulid << endl;
    }

    //  Define region
    //  Bottom region
    string region{};
    if (rank == 0) {
        BOTTOM = 1;
        region = "BOTTOM";
    } else if (rank == nProcs - 1) {
        TOP = 1;
        region = "TOP";
    } else {
        INTERNAL = 1;
        region = "INTERNAL";
    }

    //  Compute splitting of domain
    int Mj{}, rankjStartWoGhosts{};
    if (rank < M % nProcs) {
        Mj = M / nProcs + 1;
        rankjStartWoGhosts = rank * Mj;
    } else {
        Mj = M / nProcs;
        rankjStartWoGhosts = M - (nProcs - rank) * Mj;
    }

    //  Compute array sizes, and j loop limits
    const int sizeS = N * Mj;
    int sizeF{};
    if (INTERNAL) {
        sizeF = Q * N * (Mj + 2);
        jStartWoGhosts = 1;
        jEndWoGhosts = Mj + 1;
        jEndWGhosts = Mj + 2;
    } else if (TOP) {
        sizeF = Q * N * (Mj + 1);
        jStartWoGhosts = 1;
        jEndWoGhosts = Mj + 1;
        jEndWGhosts = Mj + 1;
    } else if (BOTTOM) {
        sizeF = Q * N * (Mj + 1);
        jStartWoGhosts = 0;
        jEndWoGhosts = Mj;
        jEndWGhosts = Mj + 1;
    }
    for (int r = 0; r < nProcs; r++) {
        if (r == rank)
            printf("Rank %d: %d nodes from y = %3d to y = %3d (%s)\n", rank, Mj,
                   rankjStartWoGhosts, rankjStartWoGhosts + Mj - 1,
                   region.c_str());
        MPI_Barrier(MPI_COMM_WORLD);
    }

    const int below = rank - 1;
    const int above = rank + 1;

    //  Allocate memory
    vecType dist(sizeF, 0.0);
    vecType u(sizeS, 0.0);
    vecType v(sizeS, 0.0);
    vecType x(sizeS, 0.0);
    vecType y(sizeS, 0.0);
    vecType rho(sizeS, 1.0);

    //  Initialize variables
    for (int i = 0; i < N; i++) {
        for (int j = jStartWoGhosts; j < jEndWoGhosts; j++) {
            x[LU2(i, j - 1 + BOTTOM)] = 0.5 + i;
            y[LU2(i, j - 1 + BOTTOM)] =
                0.5 + (rankjStartWoGhosts + j - !BOTTOM);
        }
    }

    double diff{};

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    //  Main loop
    for (int t = 0; t < MAXITER; t++) {
        //  Data Transfers (non-blocking)
        exchangeDataDownward(dist, rank, above, below, nProcs);
        exchangeDataUpward(dist, rank, above, below, nProcs);

        //  Stream and macro vars of nodes not touching boundary nodes
        streamCollide(dist, jStartWoGhosts + 1, jEndWoGhosts - 1);

        //  Wait for downward data transfers to finish
        MPI_Waitall(nReq, requestsDown, requestStatsDown);

        //  Stream nodes that touch ghost nodes on top
        streamCollide(dist, jEndWoGhosts - 1, jEndWoGhosts);

        //  Wait for upward data transfers to finish
        MPI_Waitall(nReq, requestsUp, requestStatsUp);

        //  Stream nodes that touch ghost nodes on bottom
        streamCollide(dist, jStartWoGhosts, jStartWoGhosts + 1);

        if (t % prntInt == 0) {
            diff = 0;
            macroVars(dist, u, v, rho, jStartWoGhosts, jEndWoGhosts, diff);

            //  Test for convergence
            if (convergence(diff, rank, t)) break;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (SAVE) {
        // Loop over each rank and append data to files
        for (int r = 0; r < nProcs; r++) {
            if (r == rank) {
                outputScalar("x", x, r);
                outputScalar("y", y, r);
                outputScalar("u", u, r);
                outputScalar("v", v, r);
                outputScalar("rho", rho, r);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;
}

// Send data from lower Y layers to upper Y layers
void exchangeDataUpward(vecType &f, const int rank, const int above,
                        const int below, const int nProcs) {
    //  Send top node row upward (all but top row) (only k = 2, 5, 6)
    if (!TOP) {
        MPI_Isend(&f[LU3(0, jEndWoGhosts - 1, 2)], N, MPI_DOUBLE, above, 1,
                  MPI_COMM_WORLD, &requestsUp[0]);
        MPI_Isend(&f[LU3(0, jEndWoGhosts - 1, 5)], N, MPI_DOUBLE, above, 1,
                  MPI_COMM_WORLD, &requestsUp[1]);
        MPI_Isend(&f[LU3(0, jEndWoGhosts - 1, 6)], N, MPI_DOUBLE, above, 1,
                  MPI_COMM_WORLD, &requestsUp[2]);
    } else {
        requestsUp[0] = MPI_REQUEST_NULL;
        requestsUp[1] = MPI_REQUEST_NULL;
        requestsUp[2] = MPI_REQUEST_NULL;
    }

    //  Receive bottom ghost node row from below (all but bottom row) (only 2,
    //  5, 6)
    if (!BOTTOM) {
        MPI_Irecv(&f[LU3(0, jStartWoGhosts - 1, 2)], N, MPI_DOUBLE, below, 1,
                  MPI_COMM_WORLD, &requestsUp[3]);
        MPI_Irecv(&f[LU3(0, jStartWoGhosts - 1, 5)], N, MPI_DOUBLE, below, 1,
                  MPI_COMM_WORLD, &requestsUp[4]);
        MPI_Irecv(&f[LU3(0, jStartWoGhosts - 1, 6)], N, MPI_DOUBLE, below, 1,
                  MPI_COMM_WORLD, &requestsUp[5]);
    } else {
        requestsUp[3] = MPI_REQUEST_NULL;
        requestsUp[4] = MPI_REQUEST_NULL;
        requestsUp[5] = MPI_REQUEST_NULL;
    }
}

// Send data from upper Y layers to lower Y layers
void exchangeDataDownward(vecType &f, const int rank, const int above,
                          const int below, const int nProcs) {
    //  Send bottom node row downward (all but bottom row) (only 4, 7, 8)
    if (!BOTTOM) {
        MPI_Isend(&f[LU3(0, jStartWoGhosts, 4)], N, MPI_DOUBLE, below, 2,
                  MPI_COMM_WORLD, &requestsDown[0]);
        MPI_Isend(&f[LU3(0, jStartWoGhosts, 7)], N, MPI_DOUBLE, below, 2,
                  MPI_COMM_WORLD, &requestsDown[1]);
        MPI_Isend(&f[LU3(0, jStartWoGhosts, 8)], N, MPI_DOUBLE, below, 2,
                  MPI_COMM_WORLD, &requestsDown[2]);
    } else {
        requestsDown[0] = MPI_REQUEST_NULL;
        requestsDown[1] = MPI_REQUEST_NULL;
        requestsDown[2] = MPI_REQUEST_NULL;
    }

    //  Receive top ghost node row from above, (all but top row) (only 4, 7, 8)
    if (!TOP) {
        MPI_Irecv(&f[LU3(0, jEndWoGhosts, 4)], N, MPI_DOUBLE, above, 2,
                  MPI_COMM_WORLD, &requestsDown[3]);
        MPI_Irecv(&f[LU3(0, jEndWoGhosts, 7)], N, MPI_DOUBLE, above, 2,
                  MPI_COMM_WORLD, &requestsDown[4]);
        MPI_Irecv(&f[LU3(0, jEndWoGhosts, 8)], N, MPI_DOUBLE, above, 2,
                  MPI_COMM_WORLD, &requestsDown[5]);
    } else {
        requestsDown[3] = MPI_REQUEST_NULL;
        requestsDown[4] = MPI_REQUEST_NULL;
        requestsDown[5] = MPI_REQUEST_NULL;
    }
}

// 2D index look-up
inline int LU2(const int i, const int j) { return N * j + i; }

// 3D index look-up
inline int LU3(const int i, const int j, const int k) {
    return Q * N * j + N * k + i;
}

// Stream and TRT collide in one step
void streamCollide(vecType &dist, const int yStart, const int yEnd) {
    double ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8;
    int im1, ip1, jm1, jp1;
    for (int i = 0; i < N; i++) {
        im1 = i - 1;
        ip1 = i + 1;
        for (int j = yStart; j < yEnd; j++) {
            jp1 = j + 1;
            jm1 = j - 1;

            //  k = 0
            ft0 = dist[LU3(i, j, 0)];

            // Interior
            if (i > 0) {
                ft1 = dist[LU3(im1, j, 1)];

                if (j > 0) {
                    ft5 = dist[LU3(im1, jm1, 5)];
                }

                if (jp1 < jEndWGhosts) {
                    ft8 = dist[LU3(im1, jp1, 8)];
                }
            }

            // Left Wall
            else {
                ft1 = dist[LU3(i, j, 3)];
                ft5 = dist[LU3(i, j, 7)];
                ft8 = dist[LU3(i, j, 6)];
            }

            // Interior
            if (ip1 < N) {
                ft3 = dist[LU3(ip1, j, 3)];

                if (j > 0) {
                    ft6 = dist[LU3(ip1, jm1, 6)];
                }

                if (jp1 < jEndWGhosts) {
                    ft7 = dist[LU3(ip1, jp1, 7)];
                }
            }

            // Right wall
            else {
                ft3 = dist[LU3(i, j, 1)];
                ft6 = dist[LU3(i, j, 8)];
                ft7 = dist[LU3(i, j, 5)];
            }

            // Interior
            if (j > 0) {
                ft2 = dist[LU3(i, jm1, 2)];
            }

            // Bottom wall
            else if (BOTTOM) {
                ft2 = dist[LU3(i, j, 4)];
                ft5 = dist[LU3(i, j, 7)];
                ft6 = dist[LU3(i, j, 8)];
            }

            // Interior
            if (jp1 < jEndWGhosts) {
                ft4 = dist[LU3(i, jp1, 4)];
            }

            // Top Wall
            else if (TOP) {
                ft4 = dist[LU3(i, j, 2)];
                ft7 = dist[LU3(i, j, 5)] - 6.0 * wd * Ulid;
                ft8 = dist[LU3(i, j, 6)] + 6.0 * wd * Ulid;
            }

            // Macroscopic Variables
            // compute moments
            const double drho =
                ft0 + ft1 + ft2 + ft3 + ft4 + ft5 + ft6 + ft7 + ft8;
            const double ux = ft1 + ft5 + ft8 - (ft3 + ft6 + ft7);
            const double uy = ft2 + ft5 + ft6 - (ft4 + ft7 + ft8);

            // Collision
            // Two relaxation time (TRT)
            double fplus, fminus, feqTerm1, feqTerm2, cdotu;
            const double halfOmega = 0.5 * OMEGA;
            const double halfOmegaMinus = 0.5 * OMEGAm;
            const double u2x15 = 1.5 * ((ux * ux) + (uy * uy));
            const double wsu2x15 = wS * u2x15;
            const double wdu2x15 = wd * u2x15;
            const double wsdrho = wS * drho;
            const double wddrho = wd * drho;

            // k = 0
            dist[LU3(i, j, 0)] = ft0 - OMEGA * (ft0 - w0 * (drho - u2x15));

            // k = 1,3
            cdotu = ux;
            feqTerm1 = wsdrho + wS * (4.5 * (cdotu * cdotu)) - wsu2x15;
            feqTerm2 = 3.0 * wS * cdotu;
            fplus = halfOmega * ((ft1 + ft3) - 2 * feqTerm1);
            fminus = halfOmegaMinus * ((ft1 - ft3) - 2 * feqTerm2);
            dist[LU3(i, j, 1)] = ft1 - fplus - fminus;
            dist[LU3(i, j, 3)] = ft3 - fplus + fminus;

            // k = 2,4
            cdotu = uy;
            feqTerm1 = wsdrho + wS * (4.5 * (cdotu * cdotu)) - wsu2x15;
            feqTerm2 = 3.0 * wS * cdotu;
            fplus = halfOmega * ((ft2 + ft4) - 2 * feqTerm1);
            fminus = halfOmegaMinus * ((ft2 - ft4) - 2 * feqTerm2);
            dist[LU3(i, j, 2)] = ft2 - fplus - fminus;
            dist[LU3(i, j, 4)] = ft4 - fplus + fminus;

            // k = 5,7
            cdotu = ux + uy;
            feqTerm1 = wddrho + wd * (4.5 * (cdotu * cdotu)) - wdu2x15;
            feqTerm2 = 3.0 * wd * cdotu;
            fplus = halfOmega * ((ft5 + ft7) - 2 * feqTerm1);
            fminus = halfOmegaMinus * ((ft5 - ft7) - 2 * feqTerm2);
            dist[LU3(i, j, 5)] = ft5 - fplus - fminus;
            dist[LU3(i, j, 7)] = ft7 - fplus + fminus;

            // k = 6,8
            cdotu = -ux + uy;
            feqTerm1 = wddrho + wd * (4.5 * (cdotu * cdotu)) - wdu2x15;
            feqTerm2 = 3.0 * wd * cdotu;
            fplus = halfOmega * ((ft6 + ft8) - 2 * feqTerm1);
            fminus = halfOmegaMinus * ((ft6 - ft8) - 2 * feqTerm2);
            dist[LU3(i, j, 6)] = ft6 - fplus - fminus;
            dist[LU3(i, j, 8)] = ft8 - fplus + fminus;
        }
    }
}

// Compute macro variables (velocity, density) from distribution
void macroVars(const vecType f, vecType &u, vecType &v, vecType &rho,
               const int yStart, const int yEnd, double &diff) {
    double temp, new_r, new_u, new_v;
    double f0, f1, f2, f3, f4, f5, f6, f7, f8;
    int ind1{};
    for (int i = 0; i < N; i++) {
        for (int j = yStart; j < yEnd; j++) {
            ind1 = LU2(i, j - 1 + BOTTOM);
            f0 = f[LU3(i, j, 0)];
            f1 = f[LU3(i, j, 1)];
            f2 = f[LU3(i, j, 2)];
            f3 = f[LU3(i, j, 3)];
            f4 = f[LU3(i, j, 4)];
            f5 = f[LU3(i, j, 5)];
            f6 = f[LU3(i, j, 6)];
            f7 = f[LU3(i, j, 7)];
            f8 = f[LU3(i, j, 8)];
            new_r = 1.0 + f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
            new_u = f1 - f3 + f5 - f6 - f7 + f8;
            new_v = f2 - f4 + f5 + f6 - f7 - f8;

            // Compute diff
            diff += pow(u[ind1] - new_u, 2) + pow(v[ind1] - new_v, 2) +
                    pow(rho[ind1] - new_r, 2);

            u[ind1] = new_u;
            rho[ind1] = new_r;
            v[ind1] = new_v;
        }
    }
}

// Test if solution is converged
bool convergence(double &diff, const int rank, const int t) {
    const double localProps[1] = {diff};
    double globalProps[1]{}, df{1.0};

    //  Compute global sums
    MPI_Allreduce(&localProps[0], &globalProps[0], 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    diff = sqrt(globalProps[0]);

    if (t == 0) {
        df0 = 1.0 / diff;
        if (rank == 0) printf("Iteration\tdf/df0\t\tMLUPS\n");
    }

    df = diff * df0;
    if (t % prntInt == 0) {
        stop = MPI_Wtime();
        if (rank == 0) {
            printf("%.3e\t%.3e\t%.1f\n", (double)t, df,
                   prntInt * N * M / (1E6 * (stop - start)));
            if (isinf(df)) exit(1);
        }
        start = stop;
    }
    return (df < THRESH);
}

// Write to disk portion of solution
// Rank 0 starts file, others append
void outputScalar(const string scalarName, const vecType scalar,
                  const int rank) {
    char fileNameChar[100]{};
    snprintf(fileNameChar, sizeof(fileNameChar), "Re=%.0f_N=%d_M=%d_%s.bin", Re,
             N, M, scalarName.c_str());
    string fileName = fileNameChar;
    ofstream myfile{};
    if (rank == 0)
        myfile.open(fileName, ios::out | ios::binary);
    else
        myfile.open(fileName, ios::out | ios::binary | ios::app);
    myfile.write((char *)&scalar[0], scalar.size() * sizeof(double));
    myfile.close();
}
