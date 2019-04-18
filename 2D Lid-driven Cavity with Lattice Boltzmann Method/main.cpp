#include <iostream>
#include <vector>
#include <math.h>
#include <omp.h>
#include "LBMClass.h"

using namespace std;

//  Simulation Parameters
const double        Re      = 100;
const unsigned int  N       = 51;
const double        Utop    = 0.1;
const double        Ubot    = 0.0;
const double        Vlef    = 0.0;
const double        Vrig    = 0.0;
const int           SAVETXT = 1;
const double        THRESH  = 1E-10;
const int           THREADS = 4;

//  Methods
const int       MBounds     = 0;    //  0 = half-way bounce-back, 1 = Non-equilibrium bounce-back, 2 = Non-equilbrium extrapolation
const int       MCollide    = 0;    //  0 = SRT, 1 = TRT, 2 = MRT

//  Problem
const int       BC          = 2;    //  0 = Couette, 1 = Poiseuille, 2 = Lid driven cavity

//  Global Constants
const unsigned  int     Q       = 9;
const double            MAXITER = 1E10;
const double            NU      = Utop * N / Re;
const double            TAU     = NU * 3.0 + 0.5;;
const double            MAGIC   = 0.25;
const double            RHO0    = 1.0;
const unsigned int      N2      = N*N;
const unsigned int      NQ      = N2*Q;
const double            w[9]    = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
const int               c[9][2] = {{0,0},{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1}};
const int               opp[9]  = {0,3,4,1,2,7,8,5,6};
const int               half[4] = {1,2,5,6};
const int               prntInt = 5000;
const unsigned int      Nm1     = N - 1;
const unsigned int      Nm2     = N - 2;

//  MRT, Hermite Polynomials
const int GM[9][9] = {
        {1, 1, 1, 1, 1, 1, 1 , 1, 1},
        {-4, -1, -1, -1, -1, 2, 2, 2, 2},
        {4, -2, -2, -2, -2, 1, 1, 1, 1},
        {0, 1, 0, -1, 0, 1, -1, -1, 1},
        {0, -2, 0, 2, 0, 1, -1, -1, 1},
        {0, 0, 1, 0, -1, 1, 1, -1, -1},
        {0, 0, -2, 0 ,2, 1, 1, -1, -1},
        {0, 1, -1, 1, -1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, -1, 1, -1}
};

const double GMinv[9][9] = {
        {1.0/9.0, -1.0/9.0, 1.0/9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0/9.0, -1.0/36.0, -1.0/18.0, 1.0/6.0, -1.0/6.0, 0.0, 0.0, 0.25, 0.0},
        {1.0/9.0, -1.0/36.0, -1.0/18.0, 0.0, 0.0, 1.0/6.0, -1.0/6.0, -0.25, 0.0},
        {1.0/9.0, -1.0/36.0, -1.0/18.0, -1.0/6.0, 1.0/6.0, 0.0, 0.0, 0.25, 0.0},
        {1.0/9.0, -1.0/36.0, -1.0/18.0, 0.0, 0.0, -1.0/6.0, 1.0/6.0, -0.25, 0.0},
        {1.0/9.0, 1.0/18.0, 1.0/36.0, 1.0/6.0, 1.0/12.0, 1.0/6.0, 1.0/12.0, 0.0, 0.25},
        {1.0/9.0, 1.0/18.0, 1.0/36.0, -1.0/6.0, -1.0/12.0, 1.0/6.0, 1.0/12.0, 0.0, -0.25},
        {1.0/9.0, 1.0/18.0, 1.0/36.0, -1.0/6.0, -1.0/12.0, -1.0/6.0, -1.0/12.0, 0.0, 0.25},
        {1.0/9.0, 1.0/18.0, 1.0/36.0, 1.0/6.0, 1.0/12.0, -1.0/6.0, -1.0/12.0, 0.0, -0.25}
};

double start,stop;

int main() {

    //  Initialize OMP
    omp_set_num_threads(THREADS);
    cout << "Threads used:\t" << THREADS << endl;

    //  Initialize Class
    LBMClass LBM;

    //  Main Loop
    start = omp_get_wtime();
    for (unsigned int t = 0; t < MAXITER; t++) {

        //  Collide
        switch(MCollide){
            case 1:
                LBM.collideTRT();
                break;
            case 2:
                LBM.collideMRT();
                break;
            default:
                LBM.collideSRT();
        }

        //  Stream
        LBM.streamPull();

        //  Boundary conditions
        switch (MBounds){
            case 1:
                LBM.NEBB();
                break;
            case 2:
                LBM.NEE();
                break;
            default:
                LBM.HWBB();
        }

        //  Macroscopic Variables
        LBM.macroVars();

        //  Convergence
        if (LBM.convergence(t)){
            if (SAVETXT==1)
                LBM.output();
            break;
        }

        LBM.swap();
    }

    return 0;
}

