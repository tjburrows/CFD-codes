//	2D Lid driven cavity with immersed cylinder using immersed boundary lattice boltzmann method.
//	See Kruger The Lattice Boltzmann Method, section 11.4.4.2
//	Travis Burrows

#include <iostream>
#include <vector>
#include <math.h>
#include <omp.h>
#include "LBMClass.h"

using namespace std;

//  Simulation Parameters
const double        Re      = 400;
const  int 			N       = 29;
const int  			M       = 29;
const double        Utop    = 0.1;
const double        Vtop    = 0.0;
const double        Ubot    = 0.0;
const double        Vbot    = 0.0;
const double        Ulef    = 0.0;
const double        Vlef    = 0.0;
const double        Urig    = 0.0;
const double        Vrig    = 0.0;
const int           SAVETXT = 1;
const double        THRESH  = 1E-12;
const int           THREADS = 8;

//  Methods
const int       MBounds     = 0;    //  0 = half-way bounce-back, 1 = Non-equilibrium bounce-back, 2 = Non-equilbrium extrapolation, 3 == wind tunnel
const int       MCollide    = 2;    //  0 = SRT, 1 = TRT, 2 = MRT
const int       PRECOND     = 0;
const double    GAMMA       = 1.0;
const int 		INCOMP		= 0;

//	Immersed Boundary
const int		IB			= 1;
const int		IBN			= 10;
const double	IBcenter[2]	= {10,15};
const double	IBradius	= 3.0;

//  Problem
const int       BC          = 2;    //  0 = Couette, 1 = Poiseuille, 2 = Lid driven cavity

//  Global Constants
const int     			Q       = 9;
const double            MAXITER = 1E10;
const double            NU      = Utop * N / Re;
const double            TAU     = NU * 3.0 + 0.5;;
const double            MAGIC   = 0.25;
const double            RHO0    = 1.0;
const double            U0      = 0.0;
const double            V0      = 0.0;
const int      			N2      = N * M;
const int      			NQ      = N2 * Q;
const double            w[9]    = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
const int               c[9][2] = {{0,0},{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1}};
const int               opp[9]  = {0,3,4,1,2,7,8,5,6};
const int               half[4] = {1,2,5,6};
const int               prntInt = 5000;
const int               Nm1     = N - 1;
const int               Nm2     = N - 2;
const int               Mm1     = M - 1;
const int               Mm2     = M - 2;

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
			case 0:
				LBM.collideSRT();
				break;
            case 1:
                LBM.collideTRT();
                break;
            case 2:
                LBM.collideMRT();
                break;
            default:
				printf("Error: Invalid collision method\n");
				exit(1);
        }

        //  Stream
        LBM.streamPull();

        //  Boundary conditions
        switch (MBounds){
			case 0:
				LBM.HWBB();
				break;
            case 1:
                LBM.NEBB();
                break;
            case 2:
                LBM.NEE();
                break;
			case 3:
				LBM.uniformFlow();
				break;
            default:
				printf("Error: Invalid Boundary method\n");
				exit(1);
        }

        //  Macroscopic Variables
        LBM.macroVars();
		
		//	Immersed Boundary
		if (IB == 1)
			LBM.immersedBoundary();

        //  Convergence
        if (LBM.convergence(t)){
            if (SAVETXT==1)
                LBM.output();
            break;
        }
        else
            LBM.swap();
    }
    return 0;
}
