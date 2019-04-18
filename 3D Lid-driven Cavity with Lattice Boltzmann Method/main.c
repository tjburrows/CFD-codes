#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

//  Simulation Parameters
#define Re          100     //  Reynolds number
#define N           50      //   N
#define UMAX        0.1     //  lid velocity
#define SAVETXT     1       //  Controls whether to save a text file output
#define THRESH      1E-12   //  Defines convergence of dp/dt
#define METHOD      1       //  0 = BGK, 1 = TRT
#define GAMMA       1.0     //  Preconditioning Coefficent.
#define DIAG        0       //  Simulate diagonally driven lid
#define THREADS     16       //  Number of threads to use

//  Global Constants
#define MAXITER     1E8
#define NU          ((UMAX) * (double) (N+1.0) / (double) (Re))       //  Kinematic viscosity
#define TAU         ((NU) * (3.0 / (double) GAMMA) + 0.5)          //  Tau
#define OMEGA       (1.0 / (double) (TAU) )          //  Relaxation constant
#define MAGIC       (0.25)      //  TRT magic parameter
#define OMEGAm      (1.0 / (0.5 + (MAGIC / ((1.0 / OMEGA) - 0.5))))     //  TRT free parameter omega-
#define RHO0        1.0                //  Initial density
#define N2          ((N) * (N))
#define N3          ((N2) * (N))
#define Q           19
#define NQ          ((N3) * (Q))
#define Nm1         ((N) - 1)
#define OmOMEGA     (1.0 - (OMEGA))
#define w0          (1.0/3.0)
#define w1          (1.0/18.0)
#define w2          (1.0/36.0)

const double  w[Q]      = {w0,w1,w1,w1,w1,w1,w1,w2,w2,w2,w2,w2,w2,w2,w2,w2,w2,w2,w2};
const int     c[Q][3]   = {{0,0,0},
                           {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
                           {1,1,0},{-1,-1,0}, {1,0,1},{-1,0,-1},{0,1,1},{0,-1,-1},{1,-1,0},{-1,1,0},{1,0,-1},{-1,0,1},{0,1,-1},{0,-1,1}};

const double gammainv = 1.0 / GAMMA;
const double sixth = 1.0 / 6.0;
const double third = 1.0 / 3.0;
const int skip = 500;

//  Macros
#define P2(x)                 ((x) * (x))              //  x^2
#define LU3(i,j,k)            (((k) * (N2)) + ((j) * (N)) + (i))     //  Look-up function, 3d, N x N x anything
#define LU4(i,j,k,q)          (((q) * (N3)) + ((k) * (N2)) + ((j) * (N)) + (i))     //  Look-up function, 4d, N x N x N x anything

//  Function Prototypes
double* malloc_vectord          (int n1);
void    free_vectord            (double *a);
void    zeros                   (double *array, int n);
void    linspace                (double *array, double start, double stop, int num);
void    datwrite                (double *x, double *y, double *z, char name3[], double *value3, char name4[], double *value4,char name5[], double *value5);
double  diff                    (double *array1, double *array2, int n);
void    checkfeq                (double feq);
void    macrovars               (double* F);
double  calcfeq                 (int c1, int c2, int c3, double w, double u1, double u2, double u3, double rho, double U2);
void    allocate                (void);
void    initialize              (void);
void    initOMP                 (void);
void    freemem                 (void);
void    swap                    (double **a, double **b);
void    calcvmag                (void);
void    macrocollideandstream   (void);
void    HechtBC                 (double *fstar);

//  Global Variables
double  *fstar,*fp,*u1,*u2,*u3,*rho,*x,*f1,*f2,*vmag;
double  df,df0,ulid,vlid,wlid;
char    filename[80];

int main(void) {

    allocate();     //  Allocate Memory

    initialize();   //  Initialize variables

    printf("Re =\t%d\n", Re);
    printf("Re_g =\t%.3e\n", 3.0*UMAX / (TAU - 0.5));
    printf("U =\t%.3e\n", UMAX);
    printf("N =\t%d\n", N);
    printf("tau =\t%.3e\n", TAU);
    printf("nu =\t%.3e\n", NU);
    printf("M =\t%.3e\n", UMAX * sqrt(3));
    if (GAMMA != 1.0)
        printf("M* =\t%.3e\n", UMAX * sqrt(3)/sqrt(GAMMA));

    //  Main Loop
    double start,stop;

    //  First iteration
    macrocollideandstream();
    HechtBC(f2);
    df0 = diff(f2, f1, NQ);
    printf("\nIteration 0:\n");
    printf("df:\t%.3e\n",df0);
    df0 = 1.0 / df0;
    swap(&f1,&f2);

    //  Rest of iterations
    start = omp_get_wtime();
    for (int t = 1; t < MAXITER; t++) {

        macrocollideandstream();    //  Calculate macro variables, collide, stream

        HechtBC(f2);                //  Apply Hecht NEBB boundary condition

        //  Convergence
        if (t % skip == 0) {
            df = diff(f2, f1, NQ) * df0;
            printf("\nIteration %d:\n", t);
            printf("df/df0:\t%.3e\n", df);
            stop = omp_get_wtime() - start;
            printf("Time:\t%.3e s\n", stop);
            printf("LUPS:\t%.3e s\n", N3 * skip / stop);
            start = omp_get_wtime();
            if (df < THRESH) {
                break;
            }
        }
        swap(&f1, &f2);
    }

    //  Output to text file
    datwrite(x,x,x,"u",u1,"v",u2,"w",u3);

    //  Free Memory
    freemem();
    return 0;
}

//  Allocate Memory
void allocate(void){
    initOMP();
    fstar   = malloc_vectord(NQ);
    fp      = malloc_vectord(NQ);
    f1      = malloc_vectord(NQ);
    f2      = malloc_vectord(NQ);
    u1      = malloc_vectord(N3);
    u2      = malloc_vectord(N3);
    u3      = malloc_vectord(N3);
    rho     = malloc_vectord(N3);
    x       = malloc_vectord(N);
    vmag    = malloc_vectord(N3);
}

//  Free Memory
void freemem(void){
    free_vectord(fstar);
    free_vectord(fp);
    free_vectord(f1);
    free_vectord(f2);
    free_vectord(u1);
    free_vectord(u2);
    free_vectord(u3);
    free_vectord(rho);
    free_vectord(x);
}

//  Calculate Equilibrium Distribution
double calcfeq(int c1, int c2, int c3, double w, double u1, double u2, double u3, double rho, double U2){
    double cdotu,feq;
    cdotu = c1 * u1 + c2 * u2 + c3 * u3;
    feq = w * rho * (1.0 + 3.0 * cdotu + (4.5 * P2(cdotu) - 1.5 * U2) * gammainv);
    checkfeq(feq);
    return feq;
}

//  Initialize Variables
void initialize(void){
    double feq,U2;
    sprintf(filename,"Solution_n=%dRe=%d_DIAG=%d_%dRT_M=%.3f.dat",N,Re,DIAG,METHOD+1,MAGIC);
    linspace(x,0,Nm1,N);
    zeros(u1,N3);
    zeros(u2,N3);
    zeros(u3,N3);
    for (int i = 0; i < N3; i++){
        rho[i] = RHO0;
    }

    if (DIAG == 1) {
        ulid = (double) UMAX / sqrt(2);
        wlid = ulid;
    }
    else {
        ulid = UMAX;
        wlid = 0.0;
    }
    vlid = 0.0;
#pragma omp parallel for private(feq,U2) collapse(3)
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            for (int k = 0; k < N; k++) {
                U2 = P2(u1[LU3(i, j, k)])+P2(u2[LU3(i, j, k)])+P2(u3[LU3(i, j, k)]);
                for (int q = 0; q < Q; q++) {
                    feq = calcfeq(c[q][0], c[q][1], c[q][2], w[q], u1[LU3(i, j, k)], u2[LU3(i, j, k)], u3[LU3(i, j, k)], rho[LU3(i, j, k)],U2);
                    fp[LU4(i, j, k, q)] = feq;
                    f1[LU4(i, j, k, q)] = feq;
                    f2[LU4(i, j, k, q)] = feq;
                }
            }
        }
    }
}

//  Initialize OpenMP
void initOMP(void){
    printf("Threads used:\t%d\n", THREADS);
    omp_set_num_threads(THREADS);
}

//  Calculate macro variables, collide, and stream
void macrocollideandstream(void){
    if (METHOD == 0){
        double feq,rhotemp,utemp,vtemp,wtemp,ftemp,U2;
        int inew,jnew,knew;
#pragma omp parallel for private(feq,inew,jnew,knew,rhotemp,utemp,vtemp,wtemp,ftemp,U2) collapse(3)
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                for (int k = 0; k < N; k++){

                    //  Macro variables
                    rhotemp = 0.0;
                    utemp = 0.0;
                    vtemp = 0.0;
                    wtemp = 0.0;
                    for (int q = 0; q < Q; q++) {
                        ftemp = f1[LU4(i, j, k, q)];
                        rhotemp += ftemp;
                        utemp += c[q][0] * ftemp;
                        vtemp += c[q][1] * ftemp;
                        wtemp += c[q][2] * ftemp;
                    }
                    rho[LU3(i, j, k)] = rhotemp;
                    utemp /= rhotemp;
                    vtemp /= rhotemp;
                    wtemp /= rhotemp;
                    U2 = P2(utemp)+P2(vtemp)+P2(wtemp);
                    for (int q = 0; q < Q; q++) {
                        //  Calculate feq
                        feq = calcfeq(c[q][0], c[q][1], c[q][2], w[q], utemp, vtemp, wtemp, rhotemp,U2);

                        //  Collision
                        fstar[LU4(i, j, k, q)] = OmOMEGA * f1[LU4(i, j, k, q)] + OMEGA * feq;

                        //  Stream
                        inew = i + c[q][0];
                        jnew = j + c[q][1];
                        knew = k + c[q][2];
                        if (inew < N && jnew < N && knew < N && inew >= 0 && jnew >= 0 && knew >= 0)
                            f2[LU4(inew, jnew, knew, q)] = fstar[LU4(i, j, k, q)];
                    }
                }
            }
        }
    }

    if (METHOD == 1){
        double fplus,fminus,feqplus,feqminus,rhotemp,rhotempinv,utemp,vtemp,wtemp,U2,fstartemp1,fstartemp2,feq1,feq2;
        int inew,jnew,knew,oppq;
#pragma omp parallel for private(fplus,fminus,feqplus,feqminus,inew,jnew,knew,utemp,vtemp,wtemp,rhotemp,rhotempinv,oppq,U2,fstartemp1,fstartemp2,feq1,feq2) collapse(3)
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                for (int k = 0; k < N; k++){

                    rhotemp = f1[LU4(i, j, k, 0)] + f1[LU4(i, j, k, 1)] + f1[LU4(i, j, k, 2)] + f1[LU4(i, j, k, 3)] + f1[LU4(i, j, k, 4)] + f1[LU4(i, j, k, 5)] + f1[LU4(i, j, k, 6)]
                              + f1[LU4(i, j, k, 7)] + f1[LU4(i, j, k, 8)] + f1[LU4(i, j, k, 9)] + f1[LU4(i, j, k, 10)] + f1[LU4(i, j, k, 11)] + f1[LU4(i, j, k, 12)] + f1[LU4(i, j, k,13)]
                              + f1[LU4(i, j, k, 14)] + f1[LU4(i, j, k, 15)] + f1[LU4(i, j, k, 16)] + f1[LU4(i, j, k, 17)] + f1[LU4(i, j, k, 18)];
                    rhotempinv = 1.0 / rhotemp;
                    utemp = (c[1][0]*f1[LU4(i, j, k, 1)] + c[2][0]*f1[LU4(i, j, k, 2)] + c[7][0]*f1[LU4(i, j, k, 7)] + c[8][0]*f1[LU4(i, j, k, 8)] + c[9][0]*f1[LU4(i, j, k, 9)] + c[10][0]*f1[LU4(i, j, k, 10)] + c[13][0]*f1[LU4(i, j, k,13)]
                             + c[14][0]*f1[LU4(i, j, k, 14)] + c[15][0]*f1[LU4(i, j, k, 15)] + c[16][0]*f1[LU4(i, j, k, 16)]) * rhotempinv;

                    vtemp = (c[3][1]*f1[LU4(i, j, k, 3)] + c[4][1]*f1[LU4(i, j, k, 4)]
                             + c[7][1]*f1[LU4(i, j, k, 7)] + c[8][1]*f1[LU4(i, j, k, 8)] + c[11][1]*f1[LU4(i, j, k, 11)] + c[12][1]*f1[LU4(i, j, k, 12)] + c[13][1]*f1[LU4(i, j, k,13)]
                             + c[14][1]*f1[LU4(i, j, k, 14)] + c[17][1]*f1[LU4(i, j, k, 17)] + c[18][1]*f1[LU4(i, j, k, 18)])* rhotempinv;

                    wtemp = (c[5][2]*f1[LU4(i, j, k, 5)] + c[6][2]*f1[LU4(i, j, k, 6)]
                             + c[9][2]*f1[LU4(i, j, k, 9)] + c[10][2]*f1[LU4(i, j, k, 10)] + c[11][2]*f1[LU4(i, j, k, 11)] + c[12][2]*f1[LU4(i, j, k, 12)]
                             + c[15][2]*f1[LU4(i, j, k, 15)] + c[16][2]*f1[LU4(i, j, k, 16)] + c[17][2]*f1[LU4(i, j, k, 17)] + c[18][2]*f1[LU4(i, j, k, 18)])* rhotempinv;

                    //  Calculate feq
                    U2 = P2(utemp)+P2(vtemp)+P2(wtemp);
                    //  TRT Collide
                    f2[LU4(i, j, k, 0)] = f1[LU4(i, j, k, 0)] - OMEGA * (f1[LU4(i, j, k, 0)] - calcfeq(c[0][0], c[0][1], c[0][2], w[0], utemp, vtemp, wtemp, rhotemp, U2));

                    for (int q = 1; q < Q; q+=2) {
                        oppq = q+1;
                        feq1 = calcfeq(c[q][0], c[q][1], c[q][2], w[q], utemp, vtemp, wtemp, rhotemp, U2);
                        feq2 = calcfeq(c[oppq][0], c[oppq][1], c[oppq][2], w[oppq], utemp, vtemp, wtemp, rhotemp, U2);

                        fplus = 0.5 * (f1[LU4(i,j,k,q)] + f1[LU4(i,j,k,oppq)]);
                        fminus = 0.5 * (f1[LU4(i,j,k,q)] - f1[LU4(i,j,k,oppq)]);
                        feqplus = 0.5 * (feq1 + feq2);
                        feqminus = 0.5 * (feq1 - feq2);

                        fplus = OMEGA * (fplus - feqplus);
                        fminus = OMEGAm * (fminus - feqminus);

                        fstartemp1 = f1[LU4(i, j, k, q)] - fplus - fminus;
                        fstartemp2 = f1[LU4(i, j, k, oppq)] - fplus + fminus;

                        //  Stream
                        inew = i + c[q][0];
                        jnew = j + c[q][1];
                        knew = k + c[q][2];
                        if (inew < N && jnew < N && knew < N && inew >= 0 && jnew >= 0 && knew >= 0)
                            f2[LU4(inew, jnew, knew, q)] = fstartemp1;

                        inew = i + c[oppq][0];
                        jnew = j + c[oppq][1];
                        knew = k + c[oppq][2];
                        if (inew < N && jnew < N && knew < N && inew >= 0 && jnew >= 0 && knew >= 0)
                            f2[LU4(inew, jnew, knew, oppq)] = fstartemp2;
                    }
                }
            }
        }
    }
}


//  Returns evenly spaced numbers over a specified interval
void linspace(double *array, double start, double stop, int num){
#pragma omp parallel for
    for (int i = 0; i < num; i++){
        array[i] = start + ((double) i) * (stop - start) / (double) (num-1);
    }
}

//  Allocates memory for 1D array of doubles
double *malloc_vectord(int n1) {
    if (n1 <= 0)                                    //  Checks for invalid inputs
        printf("Invalid input into malloc_vectord\n");
    else {
        double *mat = malloc(n1 * sizeof(double));
        if (mat == NULL) {
            printf("Error allocating memory!\n");
            exit(1);
        }
        return mat;
    }
    exit(1);
}

//  Frees memory for 1D double array
void free_vectord(double *a) {
    if (a == NULL)
        printf("Error: Null input in free_vectord");
    free((void *)a);
}

//  Assigns zeros to a vector
void zeros(double *array, int n){
#pragma omp parallel for
    for (int i = 0; i < n; i++){
        array[i] = 0.0;
    }
}

//  Writes Tecplot file
void datwrite(double *x, double *y, double *z, char name3[], double *value3, char name4[], double *value4, char name5[], double *value5){
    if (SAVETXT==1){
        macrovars(f2);
        calcvmag();
        FILE *fstar = fopen(filename,"w");
        if (fstar == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }
        fprintf(fstar, "TITLE=\"%s\" VARIABLES=\"x\", \"y\", \"z\", \"%s\", \"%s\", \"%s\", \"vmag\" ZONE T=\"%s\" I=%d J=%d K=%d F=POINT\n", filename, name3, name4, name5, filename,N,N,N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N ; j++) {
                for (int k = 0; k < N; k++) {
                    fprintf(fstar, "%.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f\n", x[i], y[j], z[k], value3[LU3(i, j, k)],value4[LU3(i, j, k)],value5[LU3(i, j, k)],vmag[LU3(i, j, k)]);
                }
            }
        }
        fclose(fstar);
    }
}

//  Non-equilibrium bounce back, Hecht, Harting 2010
void HechtBC(double *fstar){
    double rhotemp,Nzx,Nzy,Nxy,Nxz,Nyx,Nyz,temp, vfrac = 1.0 / (vlid + 1.0);
#pragma omp parallel for collapse(2) private(rhotemp,Nzx,Nzy,Nxy,Nxz,Nyx,Nyz)
    for (int i = 1; i < Nm1; i++) {
        for (int j = 1; j < Nm1; j++) {

            //  z=0
            f2[LU4(i,j,0,5)] = f2[LU4(i,j,0,6)];
            Nzx = 0.5*((f2[LU4(i,j,0,1)]+f2[LU4(i,j,0,7)]+f2[LU4(i,j,0,13)]) - (f2[LU4(i,j,0,2)]+f2[LU4(i,j,0,14)]+f2[LU4(i,j,0,8)]));
            Nzy = 0.5*((f2[LU4(i,j,0,3)]+f2[LU4(i,j,0,7)]+f2[LU4(i,j,0,14)]) - (f2[LU4(i,j,0,4)]+f2[LU4(i,j,0,13)]+f2[LU4(i,j,0,8)]));
            f2[LU4(i,j,0,9)] = f2[LU4(i,j,0,10)] - Nzx;
            f2[LU4(i,j,0,16)] = f2[LU4(i,j,0,15)] + Nzx;
            f2[LU4(i,j,0,11)] = f2[LU4(i,j,0,12)] - Nzy;
            f2[LU4(i,j,0,18)] = f2[LU4(i,j,0,17)] + Nzy;

            //  z = Nm1
            f2[LU4(i,j,Nm1,6)] = f2[LU4(i,j,Nm1,5)];
            Nzx = 0.5*((f2[LU4(i,j,Nm1,1)]+f2[LU4(i,j,Nm1,7)]+f2[LU4(i,j,Nm1,13)]) - (f2[LU4(i,j,Nm1,2)]+f2[LU4(i,j,Nm1,14)]+f2[LU4(i,j,Nm1,8)]));
            Nzy = 0.5*((f2[LU4(i,j,Nm1,3)]+f2[LU4(i,j,Nm1,7)]+f2[LU4(i,j,Nm1,14)]) - (f2[LU4(i,j,Nm1,4)]+f2[LU4(i,j,Nm1,13)]+f2[LU4(i,j,Nm1,8)]));
            f2[LU4(i,j,Nm1,10)] = f2[LU4(i,j,Nm1,9)] + Nzx;
            f2[LU4(i,j,Nm1,15)] = f2[LU4(i,j,Nm1,16)] - Nzx;
            f2[LU4(i,j,Nm1,12)] = f2[LU4(i,j,Nm1,11)] + Nzy;
            f2[LU4(i,j,Nm1,17)] = f2[LU4(i,j,Nm1,18)] - Nzy;

            //  x = 0
            f2[LU4(0,i,j,1)] = f2[LU4(0,i,j,2)];
            Nxy = 0.5*((f2[LU4(0,i,j,3)]+f2[LU4(0,i,j,11)]+f2[LU4(0,i,j,17)]) - (f2[LU4(0,i,j,4)]+f2[LU4(0,i,j,18)]+f2[LU4(0,i,j,12)]));
            Nxz = 0.5*((f2[LU4(0,i,j,5)]+f2[LU4(0,i,j,14)]+f2[LU4(0,i,j,11)]) - (f2[LU4(0,i,j,6)]+f2[LU4(0,i,j,17)]+f2[LU4(0,i,j,12)]));
            f2[LU4(0,i,j,13)] = f2[LU4(0,i,j,14)] + Nxy;
            f2[LU4(0,i,j,7)] = f2[LU4(0,i,j,8)] - Nxy;
            f2[LU4(0,i,j,9)] = f2[LU4(0,i,j,10)] - Nxz;
            f2[LU4(0,i,j,15)] = f2[LU4(0,i,j,16)] + Nxz;

            //  x = Nm1
            f2[LU4(Nm1,i,j,2)] = f2[LU4(Nm1,i,j,1)];
            Nxy = 0.5*((f2[LU4(Nm1,i,j,3)]+f2[LU4(Nm1,i,j,11)]+f2[LU4(Nm1,i,j,17)]) - (f2[LU4(Nm1,i,j,4)]+f2[LU4(Nm1,i,j,18)]+f2[LU4(Nm1,i,j,12)]));
            Nxz = 0.5*((f2[LU4(Nm1,i,j,5)]+f2[LU4(Nm1,i,j,14)]+f2[LU4(Nm1,i,j,11)]) - (f2[LU4(Nm1,i,j,6)]+f2[LU4(Nm1,i,j,17)]+f2[LU4(Nm1,i,j,12)]));
            f2[LU4(Nm1,i,j,14)] = f2[LU4(Nm1,i,j,13)] - Nxy;
            f2[LU4(Nm1,i,j,8)] = f2[LU4(Nm1,i,j,7)] + Nxy;
            f2[LU4(Nm1,i,j,10)] = f2[LU4(Nm1,i,j,9)] + Nxz;
            f2[LU4(Nm1,i,j,16)] = f2[LU4(Nm1,i,j,15)] - Nxz;

            //  y = 0
            f2[LU4(i,0,j,3)] = f2[LU4(i,0,j,4)];
            Nyx = 0.5*((f2[LU4(i,0,j,1)]+f2[LU4(i,0,j,9)]+f2[LU4(i,0,j,15)]) - (f2[LU4(i,0,j,2)]+f2[LU4(i,0,j,16)]+f2[LU4(i,0,j,10)]));
            Nyz = 0.5*((f2[LU4(i,0,j,5)]+f2[LU4(i,0,j,9)]+f2[LU4(i,0,j,16)]) - (f2[LU4(i,0,j,6)]+f2[LU4(i,0,j,15)]+f2[LU4(i,0,j,10)]));
            f2[LU4(i,0,j,7)] = f2[LU4(i,0,j,8)] - Nyx;
            f2[LU4(i,0,j,14)] = f2[LU4(i,0,j,13)] + Nyx;
            f2[LU4(i,0,j,11)] = f2[LU4(i,0,j,12)] - Nyz;
            f2[LU4(i,0,j,17)] = f2[LU4(i,0,j,18)] + Nyz;

            //  y = Nm1 (Lid)
            rhotemp = vfrac * (f2[LU4(i,Nm1,j,1)] + f2[LU4(i,Nm1,j,2)] + f2[LU4(i,Nm1,j,5)] + f2[LU4(i,Nm1,j,6)] + f2[LU4(i,Nm1,j,9)] + f2[LU4(i,Nm1,j,15)] + f2[LU4(i,Nm1,j,16)]
                               + f2[LU4(i,Nm1,j,10)] + f2[LU4(i,Nm1,j,0)] + 2.0*(f2[LU4(i,Nm1,j,3)] + f2[LU4(i,Nm1,j,7)] + f2[LU4(i,Nm1,j,14)] + f2[LU4(i,Nm1,j,11)] + f2[LU4(i,Nm1,j,17)]));
            f2[LU4(i,Nm1,j,4)] = f2[LU4(i,Nm1,j,3)] - rhotemp * vlid * third;
            Nyx = 0.5*((f2[LU4(i,Nm1,j,1)]+f2[LU4(i,Nm1,j,9)]+f2[LU4(i,Nm1,j,15)]) - (f2[LU4(i,Nm1,j,2)]+f2[LU4(i,Nm1,j,16)]+f2[LU4(i,Nm1,j,10)])) - rhotemp * ulid * third;
            Nyz = 0.5*((f2[LU4(i,Nm1,j,5)]+f2[LU4(i,Nm1,j,9)]+f2[LU4(i,Nm1,j,16)]) - (f2[LU4(i,Nm1,j,6)]+f2[LU4(i,Nm1,j,15)]+f2[LU4(i,Nm1,j,10)])) - rhotemp * wlid * third;
            f2[LU4(i,Nm1,j,8)] = f2[LU4(i,Nm1,j,7)] - rhotemp * (vlid + ulid) * sixth + Nyx;
            f2[LU4(i,Nm1,j,13)] = f2[LU4(i,Nm1,j,14)] + rhotemp * (-vlid + ulid) * sixth - Nyx;
            f2[LU4(i,Nm1,j,12)] = f2[LU4(i,Nm1,j,11)] - rhotemp * (vlid + wlid) * sixth + Nyz;
            f2[LU4(i,Nm1,j,18)] = f2[LU4(i,Nm1,j,17)] + rhotemp * (-vlid + wlid) * sixth - Nyz;
        }
    }

    //  Edges
#pragma omp parallel for private(temp)
    for (int i = 1; i < Nm1; i++){
        temp = 0.25 * (f2[LU4(i, 0, 0, 1)] - f2[LU4(i, 0, 0, 2)]);
        //  Bottom Back
        f2[LU4(i, 0, 0, 3)]  = fstar[LU4(i, 0, 0, 4)];
        f2[LU4(i, 0, 0, 7)]  = fstar[LU4(i, 0, 0, 8)] - temp;
        f2[LU4(i, 0, 0, 11)] = fstar[LU4(i, 0, 0, 12)];
        f2[LU4(i, 0, 0, 14)] = fstar[LU4(i, 0, 0, 13)] + temp;
        f2[LU4(i, 0, 0, 17)] = fstar[LU4(i, 0, 0, 18)];

        f2[LU4(i, 0, 0, 5)]  = fstar[LU4(i, 0, 0, 6)];
        f2[LU4(i, 0, 0, 9)]  = fstar[LU4(i, 0, 0, 10)] - temp;
        f2[LU4(i, 0, 0, 16)] = fstar[LU4(i, 0, 0, 15)] + temp;
        f2[LU4(i, 0, 0, 18)] = fstar[LU4(i, 0, 0, 17)];

        //  Bottom Front
        temp = 0.25 * (f2[LU4(i, 0, Nm1, 1)] - f2[LU4(i, 0, Nm1, 2)]);
        f2[LU4(i, 0, Nm1, 3)]  = fstar[LU4(i, 0, Nm1, 4)];
        f2[LU4(i, 0, Nm1, 7)]  = fstar[LU4(i, 0, Nm1, 8)] - temp;
        f2[LU4(i, 0, Nm1, 11)] = fstar[LU4(i, 0, Nm1, 12)];
        f2[LU4(i, 0, Nm1, 14)] = fstar[LU4(i, 0, Nm1, 13)] + temp;
        f2[LU4(i, 0, Nm1, 17)] = fstar[LU4(i, 0, Nm1, 18)];

        f2[LU4(i, 0, Nm1, 6)]  = fstar[LU4(i, 0, Nm1, 5)];
        f2[LU4(i, 0, Nm1, 10)] = fstar[LU4(i, 0, Nm1, 9)] + temp;
        f2[LU4(i, 0, Nm1, 12)] = fstar[LU4(i, 0, Nm1, 11)];
        f2[LU4(i, 0, Nm1, 15)] = fstar[LU4(i, 0, Nm1, 16)]- temp;

        //  Top Back
        temp = 0.25 * (f2[LU4(i, Nm1, 0, 1)] - f2[LU4(i, Nm1, 0, 2)]);
        f2[LU4(i, Nm1, 0, 4)]  = fstar[LU4(i, Nm1, 0, 3)];
        f2[LU4(i, Nm1, 0, 8)]  = fstar[LU4(i, Nm1, 0, 7)] + temp;
        f2[LU4(i, Nm1, 0, 12)] = fstar[LU4(i, Nm1,0, 11)];
        f2[LU4(i, Nm1, 0, 13)] = fstar[LU4(i, Nm1, 0, 14)] - temp;
        f2[LU4(i, Nm1, 0, 18)] = fstar[LU4(i, Nm1, 0, 17)];

        f2[LU4(i, Nm1,0, 5)]  = fstar[LU4(i, Nm1,0, 6)];
        f2[LU4(i, Nm1,0, 9)]  = fstar[LU4(i, Nm1,0, 10)] - temp;
        f2[LU4(i, Nm1,0, 11)] = fstar[LU4(i, Nm1, 0, 12)];
        f2[LU4(i, Nm1,0, 16)] = fstar[LU4(i, Nm1,0, 15)] + temp;

        //  Top Front
        temp = 0.25 * (f2[LU4(i, Nm1, Nm1, 1)] - f2[LU4(i, Nm1, Nm1, 2)]);
        f2[LU4(i, Nm1, Nm1, 4)]  = fstar[LU4(i, Nm1, Nm1, 3)];
        f2[LU4(i, Nm1, Nm1, 8)]  = fstar[LU4(i, Nm1, Nm1, 7)]   + temp;
        f2[LU4(i, Nm1, Nm1, 12)] = fstar[LU4(i, Nm1, Nm1, 11)];
        f2[LU4(i, Nm1, Nm1, 13)] = fstar[LU4(i, Nm1, Nm1, 14)]- temp;
        f2[LU4(i, Nm1, Nm1, 18)] = fstar[LU4(i, Nm1,Nm1, 17)];

        f2[LU4(i, Nm1,Nm1, 6)]  = fstar[LU4(i, Nm1,Nm1, 5)];
        f2[LU4(i, Nm1,Nm1, 10)] = fstar[LU4(i, Nm1,Nm1, 9)] + temp;
        f2[LU4(i, Nm1,Nm1, 15)] = fstar[LU4(i, Nm1,Nm1, 16)] - temp;
        f2[LU4(i, Nm1,Nm1, 17)] = fstar[LU4(i, Nm1, Nm1, 18)] ;

        //  Left Back
        temp = 0.25 * (f2[LU4(0, i, 0, 3)] - f2[LU4(0, i, 0, 4)]);
        f2[LU4(0, i, 0, 1)]  = fstar[LU4(0, i, 0, 2)];
        f2[LU4(0, i, 0, 7)]  = fstar[LU4(0, i, 0, 8)] - temp;
        f2[LU4(0, i, 0, 9)]  = fstar[LU4(0, i, 0, 10)];
        f2[LU4(0, i, 0, 13)] = fstar[LU4(0, i, 0, 14)] + temp;
        f2[LU4(0, i, 0, 15)] = fstar[LU4(0, i, 0, 16)];

        f2[LU4(0, i, 0, 5)]  = fstar[LU4(0, i, 0, 6)];
        f2[LU4(0, i, 0, 11)] = fstar[LU4(0, i, 0, 12)] - temp;
        f2[LU4(0, i, 0, 16)] = fstar[LU4(0, i, 0, 15)];
        f2[LU4(0, i, 0, 18)] = fstar[LU4(0, i, 0, 17)] + temp;

        //  Left Front
        temp = 0.25 * (f2[LU4(0, i, Nm1, 3)] - f2[LU4(0, i, Nm1, 4)]);
        f2[LU4(0, i, Nm1, 1)]  = fstar[LU4(0, i, Nm1, 2)];
        f2[LU4(0, i, Nm1, 7)]  = fstar[LU4(0, i, Nm1, 8)] - temp;
        f2[LU4(0, i, Nm1, 9)]  = fstar[LU4(0, i, Nm1, 10)];
        f2[LU4(0, i, Nm1, 13)] = fstar[LU4(0, i, Nm1, 14)] + temp;
        f2[LU4(0, i, Nm1, 15)] = fstar[LU4(0, i, Nm1, 16)];

        f2[LU4(0, i, Nm1, 6)]  = fstar[LU4(0, i, Nm1, 5)];
        f2[LU4(0, i, Nm1, 10)] = fstar[LU4(0, i, Nm1, 9)];
        f2[LU4(0, i, Nm1, 12)] = fstar[LU4(0, i, Nm1, 11)] + temp;
        f2[LU4(0, i, Nm1, 17)] = fstar[LU4(0, i, Nm1, 18)] - temp;

        //  Right Back
        temp = 0.25 * (f2[LU4(Nm1, i, 0, 3)] - f2[LU4(Nm1, i, 0, 4)]);
        f2[LU4(Nm1, i, 0, 2)]  = fstar[LU4(Nm1, i, 0, 1)];
        f2[LU4(Nm1, i, 0, 8)]  = fstar[LU4(Nm1, i, 0, 7)] - temp;
        f2[LU4(Nm1, i, 0, 10)] = fstar[LU4(Nm1, i,0, 9)];
        f2[LU4(Nm1, i, 0, 14)] = fstar[LU4(Nm1, i, 0, 13)] + temp;
        f2[LU4(Nm1, i, 0, 16)] = fstar[LU4(Nm1, i, 0, 15)];

        f2[LU4(Nm1, i,0, 5)]  = fstar[LU4(Nm1, i,0, 6)];
        f2[LU4(Nm1, i,0, 9)]  = fstar[LU4(Nm1, i, 0, 10)] ;
        f2[LU4(Nm1, i,0, 11)] = fstar[LU4(Nm1, i,0, 12)] - temp;
        f2[LU4(Nm1, i,0, 18)] = fstar[LU4(Nm1, i,0, 17)] + temp;

        //  Right Front
        temp = -0.25 * (f2[LU4(Nm1, i, Nm1, 3)] - f2[LU4(Nm1, i, Nm1, 4)]);
        f2[LU4(Nm1, i, Nm1, 2)]  = fstar[LU4(Nm1, i, Nm1, 1)];
        f2[LU4(Nm1, i, Nm1, 8)]  = fstar[LU4(Nm1, i, Nm1, 7)] - temp;
        f2[LU4(Nm1, i, Nm1, 10)] = fstar[LU4(Nm1, i, Nm1, 9)];
        f2[LU4(Nm1, i, Nm1, 14)] = fstar[LU4(Nm1, i, Nm1, 13)] + temp;
        f2[LU4(Nm1, i, Nm1, 16)] = fstar[LU4(Nm1, i,Nm1, 15)];

        f2[LU4(Nm1, i,Nm1, 6)]  = fstar[LU4(Nm1, i,Nm1, 5)];
        f2[LU4(Nm1, i,Nm1, 12)] = fstar[LU4(Nm1, i,Nm1, 11)] - temp;
        f2[LU4(Nm1, i,Nm1, 15)] = fstar[LU4(Nm1, i, Nm1, 16)];
        f2[LU4(Nm1, i,Nm1, 17)] = fstar[LU4(Nm1, i,Nm1, 18)] + temp;

        //  Bottom Left
        temp = 0.25 * (f2[LU4(0, 0, i, 5)] - f2[LU4(0, 0, i, 6)]);
        f2[LU4(0, 0, i, 3)]  = fstar[LU4(0, 0, i, 4)];
        f2[LU4(0, 0, i, 7)]  = fstar[LU4(0, 0, i, 8)];
        f2[LU4(0, 0, i, 11)] = fstar[LU4(0, 0, i, 12)] - temp;
        f2[LU4(0, 0, i, 14)] = fstar[LU4(0, 0, i, 13)];
        f2[LU4(0, 0, i, 17)] = fstar[LU4(0, 0, i, 18)] + temp;

        f2[LU4(0, 0, i, 1)]  = fstar[LU4(0, 0, i, 2)];
        f2[LU4(0, 0, i, 9)]  = fstar[LU4(0, 0, i, 10)] - temp;
        f2[LU4(0, 0, i, 13)] = fstar[LU4(0, 0, i, 14)];
        f2[LU4(0, 0, i, 15)] = fstar[LU4(0, 0, i, 16)] + temp;

        //  Top Left
        temp = 0.25 * (f2[LU4(0, Nm1, i, 5)] - f2[LU4(0, Nm1, i, 6)]);
        f2[LU4(0, Nm1, i, 4)]  = fstar[LU4(0, Nm1, i, 3)];
        f2[LU4(0, Nm1, i, 8)]  = fstar[LU4(0, Nm1, i, 7)];
        f2[LU4(0, Nm1, i, 12)] = fstar[LU4(0, Nm1, i, 11)]+ temp;
        f2[LU4(0, Nm1, i, 13)] = fstar[LU4(0, Nm1, i, 14)];
        f2[LU4(0, Nm1, i, 18)] = fstar[LU4(0, Nm1, i, 17)]- temp;

        f2[LU4(0, Nm1, i, 1)]  = fstar[LU4(0, Nm1, i, 2)];
        f2[LU4(0, Nm1, i, 7)]  = fstar[LU4(0, Nm1, i, 8)];
        f2[LU4(0, Nm1, i, 9)]  = fstar[LU4(0, Nm1, i, 10)] - temp;
        f2[LU4(0, Nm1, i, 15)] = fstar[LU4(0, Nm1, i, 16)] + temp;


        //  Bottom Right
        temp = 0.25 * (f2[LU4(Nm1, 0, i, 5)] - f2[LU4(Nm1, 0, i, 6)]);
        f2[LU4(Nm1, 0, i, 3)]  = fstar[LU4(Nm1, 0, i, 4)];
        f2[LU4(Nm1, 0, i, 7)]  = fstar[LU4(Nm1, 0, i, 8)];
        f2[LU4(Nm1, 0, i, 11)] = fstar[LU4(Nm1, 0, i, 12)] - temp;
        f2[LU4(Nm1, 0, i, 14)] = fstar[LU4(Nm1, 0, i, 13)];
        f2[LU4(Nm1, 0, i, 17)] = fstar[LU4(Nm1, 0, i, 18)] + temp;

        f2[LU4(Nm1, 0, i, 2)]  = fstar[LU4(Nm1, 0, i, 1)];
        f2[LU4(Nm1, 0, i, 8)]  = fstar[LU4(Nm1, 0, i, 7)];
        f2[LU4(Nm1, 0, i, 10)] = fstar[LU4(Nm1, 0, i, 9)] + temp;
        f2[LU4(Nm1, 0, i, 16)] = fstar[LU4(Nm1, 0, i, 15)] - temp;


        //  Top Right
        temp = 0.25 * (f2[LU4(Nm1, Nm1, i, 5)] - f2[LU4(Nm1, Nm1, i, 6)]);
        f2[LU4(Nm1, Nm1, i, 4)]  = fstar[LU4(Nm1, Nm1, i, 3)];
        f2[LU4(Nm1, Nm1, i, 8)]  = fstar[LU4(Nm1, Nm1, i, 7)];
        f2[LU4(Nm1, Nm1, i, 12)] = fstar[LU4(Nm1, Nm1, i, 11)] + temp;
        f2[LU4(Nm1, Nm1, i, 13)] = fstar[LU4(Nm1, Nm1, i, 14)];
        f2[LU4(Nm1, Nm1, i, 18)] = fstar[LU4(Nm1, Nm1, i, 17)] - temp;

        f2[LU4(Nm1, Nm1, i, 2)]  = fstar[LU4(Nm1, Nm1, i, 1)];
        f2[LU4(Nm1, Nm1, i, 10)] = fstar[LU4(Nm1, Nm1, i, 9)] + temp;
        f2[LU4(Nm1, Nm1, i, 14)] = fstar[LU4(Nm1, Nm1, i, 13)];
        f2[LU4(Nm1, Nm1, i, 16)] = fstar[LU4(Nm1, Nm1, i, 15)] - temp;
    }

    //  Corners
    //  Back Top left
    f2[LU4(0, Nm1, 0, 1)] = fstar[LU4(0, Nm1, 0, 2)];
    f2[LU4(0, Nm1, 0, 4)] = fstar[LU4(0, Nm1, 0, 3)];
    f2[LU4(0, Nm1, 0, 5)] = fstar[LU4(0, Nm1, 0, 6)];
    f2[LU4(0, Nm1, 0, 9)] = fstar[LU4(0, Nm1, 0, 10)];
    f2[LU4(0, Nm1, 0, 13)] = fstar[LU4(0, Nm1, 0, 14)];
    f2[LU4(0, Nm1, 0, 18)] = fstar[LU4(0, Nm1, 0, 17)];

    //  Back Bottom Left
    f2[LU4(0, 0, 0, 1)] = fstar[LU4(0, 0, 0, 2)];
    f2[LU4(0, 0, 0, 3)] = fstar[LU4(0, 0, 0, 4)];
    f2[LU4(0, 0, 0, 5)] = fstar[LU4(0, 0, 0, 6)];
    f2[LU4(0, 0, 0, 9)] = fstar[LU4(0, 0, 0, 10)];
    f2[LU4(0, 0, 0, 7)] = fstar[LU4(0, 0, 0, 8)];
    f2[LU4(0, 0, 0, 11)] = fstar[LU4(0, 0, 0, 12)];

    //  Back Top Right
    f2[LU4(Nm1, Nm1, 0, 2)] = fstar[LU4(Nm1, Nm1, 0, 1)];
    f2[LU4(Nm1, Nm1, 0, 4)] = fstar[LU4(Nm1, Nm1, 0, 3)];
    f2[LU4(Nm1, Nm1, 0, 5)] = fstar[LU4(Nm1, Nm1, 0, 6)];
    f2[LU4(Nm1, Nm1, 0, 16)] = fstar[LU4(Nm1, Nm1, 0, 15)];
    f2[LU4(Nm1, Nm1, 0, 8)] = fstar[LU4(Nm1, Nm1, 0, 7)];
    f2[LU4(Nm1, Nm1, 0, 18)] = fstar[LU4(Nm1, Nm1, 0, 17)];

    //  Back Bottom Right
    f2[LU4(Nm1, 0, 0, 2)] = fstar[LU4(Nm1, 0, 0, 1)];
    f2[LU4(Nm1, 0, 0, 3)] = fstar[LU4(Nm1, 0, 0, 4)];
    f2[LU4(Nm1, 0, 0, 5)] = fstar[LU4(Nm1, 0, 0, 6)];
    f2[LU4(Nm1, 0, 0, 16)] = fstar[LU4(Nm1, 0, 0, 15)];
    f2[LU4(Nm1, 0, 0, 14)] = fstar[LU4(Nm1, 0, 0, 13)];
    f2[LU4(Nm1, 0, 0, 11)] = fstar[LU4(Nm1, 0, 0, 12)];

    //  Front Top left
    f2[LU4(0, Nm1, Nm1, 1)] = fstar[LU4(0, Nm1, Nm1, 2)];
    f2[LU4(0, Nm1, Nm1, 4)] = fstar[LU4(0, Nm1, Nm1, 3)];
    f2[LU4(0, Nm1, Nm1, 6)] = fstar[LU4(0, Nm1, Nm1, 5)];
    f2[LU4(0, Nm1, Nm1, 15)] = fstar[LU4(0, Nm1, Nm1, 16)];
    f2[LU4(0, Nm1, Nm1, 13)] = fstar[LU4(0, Nm1, Nm1, 14)];
    f2[LU4(0, Nm1, Nm1, 12)] = fstar[LU4(0, Nm1, Nm1, 11)];

    //  Front Bottom Left
    f2[LU4(0, 0, Nm1, 1)] = fstar[LU4(0, 0, Nm1, 2)];
    f2[LU4(0, 0, Nm1, 3)] = fstar[LU4(0, 0, Nm1, 4)];
    f2[LU4(0, 0, Nm1, 6)] = fstar[LU4(0, 0, Nm1, 5)];
    f2[LU4(0, 0, Nm1, 15)] = fstar[LU4(0, 0, Nm1, 16)];
    f2[LU4(0, 0, Nm1, 7)] = fstar[LU4(0, 0, Nm1, 8)];
    f2[LU4(0, 0, Nm1, 17)] = fstar[LU4(0, 0, Nm1, 18)];

    //  Front Top Right
    f2[LU4(Nm1, Nm1, Nm1, 2)] = fstar[LU4(Nm1, Nm1, Nm1, 1)];
    f2[LU4(Nm1, Nm1, Nm1, 4)] = fstar[LU4(Nm1, Nm1, Nm1, 3)];
    f2[LU4(Nm1, Nm1, Nm1, 6)] = fstar[LU4(Nm1, Nm1, Nm1, 5)];
    f2[LU4(Nm1, Nm1, Nm1, 10)] = fstar[LU4(Nm1, Nm1, Nm1, 9)];
    f2[LU4(Nm1, Nm1, Nm1, 8)] = fstar[LU4(Nm1, Nm1, Nm1, 7)];
    f2[LU4(Nm1, Nm1, Nm1, 12)] = fstar[LU4(Nm1, Nm1, Nm1, 11)];

    //  Front Bottom Right
    f2[LU4(Nm1, 0, Nm1, 2)] = fstar[LU4(Nm1, 0, Nm1, 1)];
    f2[LU4(Nm1, 0, Nm1, 3)] = fstar[LU4(Nm1, 0, Nm1, 4)];
    f2[LU4(Nm1, 0, Nm1, 6)] = fstar[LU4(Nm1, 0, Nm1, 5)];
    f2[LU4(Nm1, 0, Nm1, 10)] = fstar[LU4(Nm1, 0, Nm1, 9)];
    f2[LU4(Nm1, 0, Nm1, 14)] = fstar[LU4(Nm1, 0, Nm1, 13)];
    f2[LU4(Nm1, 0, Nm1, 17)] = fstar[LU4(Nm1, 0, Nm1, 18)];
}

//  Returns L2 norm of difference of two arrays
double diff(double *array1, double *array2, int n){
    double difference = 0;
#pragma omp parallel for default(none) shared(array1,array2,n) reduction(+:difference)
    for (int i = 0; i < n; i++){
        difference += P2(array1[i] - array2[i]);
    }
    return sqrt(difference);
}

void checkfeq(double feq){
    if (feq < 0.0) {
        printf("Error: negative feq.  Therefore, unstable.\n");
        exit(1);
    }
}

//  Calculate macro variables
void macrovars(double* F){
    double rhotemp,utemp,vtemp,wtemp,ftemp;
#pragma omp parallel for private(rhotemp, utemp, vtemp, wtemp, ftemp) collapse(3)
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            for (int k = 0; k < N; k++) {
                rhotemp = 0.0;
                utemp = 0.0;
                vtemp = 0.0;
                wtemp = 0.0;
                for (int q = 0; q < Q; q++) {
                    ftemp = F[LU4(i, j, k, q)];
                    rhotemp += ftemp;
                    utemp += c[q][0] * ftemp;
                    vtemp += c[q][1] * ftemp;
                    wtemp += c[q][2] * ftemp;
                }
                rho[LU3(i, j, k)] = rhotemp;
                u1[LU3(i, j, k)] = utemp / rhotemp;
                u2[LU3(i, j, k)] = vtemp / rhotemp;
                u3[LU3(i, j, k)] = wtemp / rhotemp;
            }
        }
    }
}

//  Swap pointers
void swap(double **a, double **b){
    double *temp;
    temp=*a;
    *a = *b;
    *b = temp;
}

//  Calculate velocity magnitude
void calcvmag(void){
#pragma omp parallel for default(none) shared(vmag,u1,u2,u3) collapse(3)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                vmag[LU3(i,j,k)] = sqrt(P2(u1[LU3(i,j,k)]) + P2(u2[LU3(i,j,k)]) + P2(u3[LU3(i,j,k)]));
            }
        }
    }
}