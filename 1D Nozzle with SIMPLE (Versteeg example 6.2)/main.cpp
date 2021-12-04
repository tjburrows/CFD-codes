//	Steady 1D Nozzle using SIMPLE and SIMPLER, upwind differencing
//	See Versteeg, An Introduction to Computational Fluid Dynamics, Example 6.2
//  Travis Burrows

#include <iostream>
#include <array>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <vector>

//  Uncomment or comment this to enable or disable the AMGCL library
#define USE_AMGCL

#if defined(USE_AMGCL)
#include <amgcl/amg.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

//  Solvers
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/solver/lgmres.hpp>
#include <amgcl/solver/fgmres.hpp>

//  Relxation methods
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/chebyshev.hpp>


//  Coursening methods
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#endif

using namespace std;

//  Parameters
const int       N       = 10;           //  Number of points
const double    THRESH  = 1E-7;        //  Outer Iteration L2 norm threshold
const double    OMEGAu  = 0.02;          //  Momentum relaxation coefficient
const double    OMEGAp  = 0.02;          //  Pressure relaxation coefficient
const double    MAXITER = 1E7;          //  Maximum iterations
const int       DEBUG = 0;              //  Print extra information
const int       printint    = 10;       //  Interval to print convergence information
const int       METHOD  = 1;             //  0 = SIMPLE, 1 = SIMPLER

//  Global Constants
const double    L       = 2.0;                  //  Length of domain (1D)
const double    Ai      = 0.5;                  //  Initial area in meters squared
const double    Af      = 0.1;                  //  Final area in meters squared
const double    rho     = 1.0;                  //  Fluid density
const double    Pi      = 10;                   //  Inlet pressure
const double    Po      = 0;                    //  Outlet pressure
const double    Mi      = 1.0;                  //  initial mass flow rate
const double    OmOMEGAu  = 1.0 - OMEGAu;       //  One minus momentum relaxation coefficient
const double    OmOMEGAp  = 1.0 - OMEGAp;       //  One minus perssure relaxation coefficient
const int       Nm1 = N - 1;
const int       Nm2 = N - 2;
const double    dx = L / Nm1;                   //  delta x
const double    THRESHinner = THRESH / 150.0;   //  Inner iteration residual threshold
const double    Mexact = sqrt(0.2);             //  Exact solution mass flow rate
const double De = 0.0, Dw = 0.0;                //  Diffusion coefficients

//  Data types
typedef array<double, N>    Pvec;     //  1D vector of doubles for pressure
typedef array<double, Nm1>  Uvec;     //  1D vector of doubles for velocity

//  Function Declarations
template<std::size_t SIZE>
inline void         printMatrix         (const string &name, const array<double,SIZE> &vec);
inline double       P2                  (const double &value);
inline void         calcuHat            (const int &i,Uvec &uHat,const Pvec &p,const Uvec &areaU, const Pvec &areaP,double &UE, double &UW, double &UP, double &d);
inline double       Pexact              (const double &area);
inline double       Uexact              (const double &area);
inline void         poisson             (const int &n, vector<int> &ptr, vector<int> &col, vector<double> &val, vector<double> &rhs, const Pvec &aWp,const Pvec &aEp,const Pvec &aPp,const Pvec &bPrime,const int &mode = 0, const double &P0 = 0);
inline void         momentum            (const int &n, vector<int> &ptr, vector<int> &col, vector<double> &val, vector<double> &rhs, const Uvec &aWu,const Uvec &aEu,const Uvec &aPu,const Uvec &Su);
inline void         buildMomentumCoeffs (const Pvec &p, const Pvec &areaP, const Uvec &uPrev, const Uvec &areaU, Uvec &Su, Uvec &aPu, Uvec &aWu, Uvec &aEu, Uvec &aPuinv, Uvec &d);
inline void         buildPressureCoeffs (Pvec &aWp, Pvec &aEp, Pvec &aPp, Pvec &aPpinv, Pvec &bPrimep, const Uvec &d, const Uvec &usource, const Uvec &areaU);
inline void         solnError           (double &perror, double &uerror,const Pvec &p, const Pvec &pexact,const Uvec &u, const Uvec &uexact);
inline void         throwError          (const string &message);
inline void         checkparams         ();

int main() {

    //  Check for valid parameters
    checkparams();

    //  Declare variables
    Pvec xP{},areaP{},p{},pPrime{},aWp{},aEp{},aPp{},aPpinv{},bPrimep{},pPrev{},Pexsoln{};
    Uvec xU{},areaU{},u{},uStar{},d{},mfr{},Su{},uPrev{},Uexsoln{},uHat{},aPu{},aEu{},aWu{},aPuinv{};
    double dif{},UP{},UE{},UW{},temp{},PP{},totaldif{},totaldif0{},start{},stop{},uError{},pError{};
    int Mcount{},Pcount{},Pprimecount{};

    //  Declare AMGCL-specific variables
#if defined(USE_AMGCL)
    double amgerror{};
    vector<int> ptr{},col{};
    vector <double> val{},rhs2{};
    vector <double> x(N, 0.0);

    //  Define backend
    typedef amgcl::backend::builtin<double> Backend;

    //  Define AMGCL solver parameters
    typedef amgcl::make_solver<
            amgcl::amg<
                    Backend,
                    amgcl::coarsening::smoothed_aggregation,    //  Coursening method
                    amgcl::relaxation::spai0                    //  Relaxation method
            >,
            amgcl::solver::bicgstabl<Backend>                   //  linear solver method
    > Solver;
#else
    double rhs{};
#endif

    //  Initialize OpenMP with maximum threads
    const int maxthreads = omp_get_max_threads();
    omp_set_num_threads(maxthreads);

    //  Print some parameters
    printf("Threads: %d\n",maxthreads);
    printf("dx: %.2e\n",dx);
    printf("N: %d\n",N);
    cout << endl;

    //  Initialize Variables
    const double Aslope = (Af - Ai) / L;
    const double Pslope = (Po - Pi) / L;
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        xP[i] = i * dx;
        areaP[i] = Ai + Aslope * i * dx;        //  area at P nodes
        Pexsoln[i] = Pexact(areaP[i]);          //  exact pressure solution
        p[i] = Pi + Pslope * i * dx;            //  initial guess of pressure
        if (i < Nm1) {
            xU[i] = 0.5 * dx + i * dx;
            areaU[i] = Ai + Aslope * 0.5 * dx + Aslope * i * dx;    //  area at U nodes
            Uexsoln[i] = Uexact(areaU[i]);                          //  exact u solution
            u[i] = Mi / (rho * areaU[i]);                           //  initial guess of velocity
        }
    }

    if (DEBUG) {
        printMatrix("xP",xP);
        printMatrix("areaP",areaP);
        printMatrix("areaU",areaU);
        printMatrix("u",u);
        printMatrix("p",p);
    }

    if (METHOD == 0)
        printf("Iterations\tU* iters\tP' iters\tL2 norm\t\tP error\t\tU error\n");
    else if (METHOD == 1)
        printf("Iterations\tP iters\t\tU* iters\tP' iters\tL2 norm\t\tP error\t\tU error\n");

    //  Outer iterations
    start = omp_get_wtime();
    for (int outer = 0; outer < MAXITER; outer++) {

        //  Save previous iteration values
        pPrev = p;
        uPrev = u;

        //  SIMPLER-specific steps
        if (METHOD == 1) {

            //  Calculate u-hat
#pragma omp parallel for private(UP, UE, UW)
            for (int i = 0; i < Nm1; i++) {
                UP = uPrev[i];
                UW = 0.0; UE = 0.0;
                if (i > 0)
                    UW = uPrev[i - 1];
                if (i < Nm2)
                    UE = uPrev[i + 1];
                calcuHat(i, uHat, p, areaU, areaP, UE, UW, UP, d[i]);
            }

            //  build vector of coefficients for pressure equation
            buildPressureCoeffs(aWp,aEp,aPp,aPpinv,bPrimep,d,uHat,areaU);

            //  solve pressure equation
#if defined(USE_AMGCL)
            PP = p[0];      //  Keep first pressure value from previous iteration (boundary condition)
            poisson(N,ptr,col,val,rhs2,aWp,aEp,aPp,bPrimep,1,PP);   //  build AMGCL matrices
            Solver solveP( std::tie(N, ptr, col, val));             //  define solver
            x.clear();x.reserve(N);
            std::tie(Pcount, amgerror) = solveP(rhs2, x);           //  solve
            copy_n(x.begin(),N,p.begin());                          //  copy solution to pressure array
#else
            dif = 1.0;
            Pcount = 0;
            p[0] = Pi - 0.5 * P2(uPrev[0] * areaU[0] / areaP[0]);   //  define first pressure value (probabaly redundant)
            p[Nm1] = 0.0;                                           //  define last pressure value

            //  Iterate until pressure residual converges
            while (dif > THRESH) {
                dif = 0.0;
#pragma omp parallel for private(temp, PP, rhs) reduction(+:dif)
                for (int i = 1; i < Nm1; i++) {
                    temp = p[i];
                    rhs = aWp[i] * p[i - 1] + aEp[i] * p[i + 1] + bPrimep[i];
                    dif += P2(rhs - temp * aPp[i]);
                    PP = rhs * aPpinv[i];
                    PP = OmOMEGAp * temp + OMEGAp * PP;         //  under-relaxation
                    p[i] = PP;
                }
                dif = sqrt(dif / Nm2);
                Pcount++;
                if (DEBUG)
                    cout << dif << endl;
            }
#endif
        }
        
        //  Solve momentum equation for u*
        Mcount = 0;
        uStar = u;
        buildMomentumCoeffs(p,areaP,uPrev,areaU,Su,aPu,aWu,aEu,aPuinv,d);    //  build momentum coefficients

#if defined(USE_AMGCL)
        momentum(Nm1,ptr,col,val,rhs2,aWu,aEu,aPu,Su);   //  build AMGCL matrices
        Solver solveM( std::tie(Nm1, ptr, col, val));       //  define momentum solver
        x.clear();x.reserve(Nm1);
        std::tie(Mcount, amgerror) = solveM(rhs2, x);       //  solve
        copy_n(x.begin(),Nm1,uStar.begin());                //  copy solution to uStar
#else
        //  Iterate until uStar converges
        dif = 1.0;
        while (dif > THRESHinner) {
            dif = 0.0;
#pragma omp parallel for private(UP,UE,UW,temp,rhs) reduction(+:dif)
            for (int i = 0; i < Nm1; i++) {
                UP = uStar[i];
                switch (i) {

                    //  Edge cases
                    case 0 : {
                        rhs = Su[i];
                    } break;
                    case Nm2 : {
                        UW = uStar[i - 1];
                        rhs = aWu[i] * UW + Su[i];
                    } break;

                    //  Interior
                    default : {
                        UE = uStar[i + 1];
                        UW = uStar[i - 1];
                        rhs = aWu[i] * UW + aEu[i] * UE + Su[i];
                    }
                }
                dif += P2(rhs - aPu[i] * UP);       //  calculate residual
                temp = UP;
                UP = rhs * aPuinv[i];
                UP = OmOMEGAu * temp + OMEGAu * UP; //  Under-relaxation
                uStar[i] = UP;
            }
            dif = sqrt(dif / Nm1);
            Mcount++;

            if (DEBUG)
                cout << dif << endl;
        }
#endif

        if (DEBUG)
            printMatrix("uStar", uStar);

        //  Iteratively solve for pressure correction
        pPrime[0] = 0.0;
        pPrime[Nm1] = 0.0;
        dif = 1.0;
        Pprimecount = 0;
        
        //  build vector of coefficients
        buildPressureCoeffs(aWp,aEp,aPp,aPpinv,bPrimep,d,uStar,areaU);

#if defined(USE_AMGCL)
        poisson(N,ptr,col,val,rhs2,aWp,aEp,aPp,bPrimep,0);          //  build AMGCL matrices
        Solver solvePprime( std::tie(N, ptr, col, val));            //  define solver
        x.clear();x.reserve(N);
        std::tie(Pprimecount, amgerror) = solvePprime(rhs2, x);     //  solve
        copy_n(x.begin(),N,pPrime.begin());                         //  copy solution to pprime
#else
        //  Iterate until P' converges
        while (dif > THRESHinner) {
            dif = 0.0;
#pragma omp parallel for private(temp,PP,rhs) reduction(+:dif)
            for (int i = 1; i < Nm1; i++) {
                temp = pPrime[i];
                rhs = aWp[i] * pPrime[i - 1] + aEp[i] * pPrime[i + 1] + bPrimep[i];
                dif += P2(rhs - temp * aPp[i]);         //  calculate residual
                PP = rhs * aPpinv[i];
                PP = OmOMEGAp * temp + OMEGAp * PP;     //  under-relaxation
                pPrime[i] = PP;
            }
            dif = sqrt(dif / Nm2);
            Pprimecount++;
            if (DEBUG)
                cout << dif << endl;
        }
#endif

        if (DEBUG)
            printMatrix("pPrime", pPrime);

        //  Correct pressure and velocity
        totaldif = 0.0;
#pragma omp parallel for private(PP,UP,temp) reduction(+:totaldif)
        for (int i = 0; i < Nm1; i++) {

            //  Correct all velocities
            UP = u[i];
            temp = UP;
            UP = uStar[i] + d[i] * (pPrime[i] - pPrime[i + 1]);
            UP = OMEGAu * UP + OmOMEGAu * temp;
            u[i] = UP;
            totaldif += P2(uPrev[i] - UP);

            //  Correct pressure, except on edges
            PP = p[i];
            if (i == 0)
                PP = Pi - 0.5 * P2(u[0]) * P2(areaU[0] / areaP[0]);
            else {
                if (METHOD==0)
                    PP += OMEGAp * pPrime[i];       //  under-relaxation
                else
                    PP += pPrime[i];
            }
            if ( i == 0 || METHOD != 1)
                p[i] = PP;
            totaldif += P2(pPrev[i] - p[i]);
        }

        //  Calculate L2 norm of pressure and velocity difference
        if (outer == 0)
            totaldif0 = 1.0 / sqrt(totaldif / (2.0 * Nm1));

        totaldif = sqrt(totaldif / ((2.0 * Nm1))) * totaldif0;

        //  Print information
        if (outer % printint == 0) {

            //  Calculate error from exact solution
            solnError(pError,uError,p,Pexsoln, u,Uexsoln);
            if (METHOD == 0)
                printf("%.2e\t%.2e\t%.2e\t%.2e\t%.3e\t%.3e\t\n",(double) outer,(double) Mcount,(double) Pprimecount,totaldif,pError,uError);
            else if (METHOD == 1)
                printf("%.2e\t%.2e\t%.2e\t%.2e\t%.3e\t%.3e\t%.3e\n",(double) outer,(double) Pcount ,(double) Mcount,(double) Pprimecount,totaldif,pError,uError);

            //  If converged, print final information
            if (totaldif < THRESH) {

                //  Calculate mass flow rate
#pragma omp parallel for
                for (int i = 0; i < N; i++)
                    mfr[i] = rho * areaU[i] * u[i];

                stop = omp_get_wtime();
                printMatrix("Mass Flow Rate", mfr);
                printMatrix("u", u);
                if (METHOD == 1)
                    printMatrix("uHat", uHat);
                printMatrix("p", p);
                printf("\nTime: %.3e s",stop - start);
                break;
            }
        }
        if (isnan(totaldif) || isinf(totaldif) || !isnormal(dif))
            throwError("dif error\n");
    }
    return 0;
}

//  power of 2
inline double P2(const double &value) {
    return value * value;
}

//  Prints a matrix
template<std::size_t SIZE>
inline void printMatrix(const string &name, const array<double, SIZE> &vec){
    printf("\n%s:\n", name.c_str());
    for (size_t i = 0; i < SIZE; i++)
        printf("%.4f\t", vec[i]);
    printf("\n");
}

//  Calculates u-hat for SIMPLER
inline void calcuHat(const int &i,Uvec &uHat,const Pvec &p,const Uvec &areaU, const Pvec &areaP,double &UE, double &UW, double &UP, double &d){
    double Fw{},Fe{},Aw{},Ae{},aW{},aE{},aP{},Su{},b{};
    switch (i) {

        //  Left boundary
        case 0 : {
            aW = 0.0;
            aE = 0.0;
            Fw = rho * UP * areaU[0];
            Fe = 0.5 * rho * areaP[1] * (UP + UE);
            aP = Fe + Fw * 0.5 * pow(areaU[0] / areaP[0], 2.0);
            Su = (Pi - p[1]) * areaU[0] + UP * Fw * (areaU[0] / areaP[0]);
            b = Su - (p[0] - p[1]) * areaU[0] ;
        } break;

        //  Right boundary
        case Nm2 : {
            Fw = 0.5 * rho * areaP[Nm2] * (UP + UW);
            Fe = rho * UP * areaU[Nm2];
            aW = Fw;
            aE = 0.0;
            aP = aW + aE + (Fe - Fw);
            b = 0;
        } break;

        //  Interior
        default : {

            //  Cell-face areas
            Aw = areaP[i];
            Ae = areaP[i + 1];

            //  Flux terms (rho == 1)
            Fw = 0.5 * Aw * (UP + UW);
            Fe = 0.5 * Ae * (UP + UE);

            //  Upwind difference coefficients
            aW = max(Fw, 0.0);
            aE = max(0.0, -Fe);
            aP = aW + aE + (Fe - Fw);

            b = 0;
        }
    }
    d = areaU[i] / aP;
    uHat[i] = (aW * UW + aE * UE + b) / aP;
}

//  Calculates solution error
inline void solnError(double &perror, double &uerror,const Pvec &p, const Pvec &pexact,const Uvec &u, const Uvec &uexact) {
    perror = 0.0; uerror = 0.0;
#pragma omp parallel for reduction(+:perror,uerror)
    for (int i = 0; i < N; i++) {
        perror += P2(p[i] - pexact[i]);
        if (i < Nm1)
            uerror += P2(u[i] - uexact[i]);
    }
    uerror = sqrt(uerror / Nm1);
    perror = sqrt(perror / N);
}

//  Exact solution for pressure
inline double Pexact(const double &area){
    return 10.0 - 0.5 * P2(Mexact) / P2(area);
}

//  Exact solution for velocity
inline double Uexact(const double &area){
    return Mexact / area;
}

// Assembles matrix for pressure equation
inline void poisson(const int &n, vector<int> &ptr, vector<int> &col, vector<double> &val, vector<double> &rhs, const Pvec &aWp,const Pvec &aEp,const Pvec &aPp,const Pvec &bPrime,const int &mode, const double &P0) {
    ptr.clear(); ptr.reserve(n + 1); ptr.push_back(0);
    col.clear(); col.reserve(n * 3);
    val.clear(); val.reserve(n * 3);
    rhs.resize(n);

    for(int i = 0; i < n; i++) {
        if (mode == 0 && (i == 0 || i == n - 1)) {
            col.push_back(i);
            val.push_back(1.0);
            rhs[i] = 0.0;
        }
        else if (mode == 1 && i == n-1){
            col.push_back(i);
            val.push_back(1.0);
            rhs[i] = 0.0;
        }
        else if (mode == 1 && i == 0) {
            col.push_back(i);
            val.push_back(1.0);
            rhs[i] = P0;
        }
        else {
            col.push_back(i - 1);
            val.push_back(-aWp[i]);

            col.push_back(i);
            val.push_back(aPp[i]);

            col.push_back(i + 1);
            val.push_back(-aEp[i]);

            rhs[i] = bPrime[i];
        }
        ptr.push_back(col.size());
    }
}

inline void momentum(const int &n, vector<int> &ptr, vector<int> &col, vector<double> &val, vector<double> &rhs, const Uvec &aWu,const Uvec &aEu,const Uvec &aPu,const Uvec &Su) {
    ptr.clear(); ptr.reserve(n + 1); ptr.push_back(0);
    col.clear(); col.reserve(n * 3);
    val.clear(); val.reserve(n * 3);
    rhs.resize(n);

    for(int i = 0; i < n; i++) {
        if (i == 0) {
            col.push_back(i);
            val.push_back(aPu[i]);
        }
        else if (i == n-1) {
            col.push_back(i - 1);
            val.push_back(-aWu[i]);
            col.push_back(i);
            val.push_back(aPu[i]);
        }
        else {
            col.push_back(i - 1);
            val.push_back(-aWu[i]);

            col.push_back(i);
            val.push_back(aPu[i]);

            col.push_back(i + 1);
            val.push_back(-aEu[i]);
        }
        rhs[i] = Su[i];
        ptr.push_back(col.size());
    }
}


//  Build Momentum coefficient vectors
inline void buildMomentumCoeffs(const Pvec &p, const Pvec &areaP, const Uvec &uPrev, const Uvec &areaU, Uvec &Su, Uvec &aPu, Uvec &aWu, Uvec &aEu, Uvec &aPuinv, Uvec &d){
    double UP{},UE{},UW{},Aw{},Ae{},Fw{},Fe{},Pe{},Pw{},Vol{},dPdx{};
#pragma omp parallel for private(UP,UE,UW,Aw,Ae,Fw,Fe,Pe,Pw,Vol,dPdx)
    for (int i = 0; i < Nm1; i++) {
        //  Cell-face pressures
        if (i == 0)
            Pw = Pi;
        else
            Pw = p[i];
        Pe = p[i + 1];

        //  Cell Volume
        Vol = areaU[i] * dx;

        //  dp/dx
        dPdx = (Pw - Pe) / dx;

        //  Source term
        UP = uPrev[i];
        Su[i] = dPdx * Vol;
        if  (i == 0)
            Su[i] += P2(UP * areaU[i]) / areaP[i];

        //  Neighbor velocities
        if (i != Nm2)
            UE = uPrev[i + 1];
        if (i != 0)
            UW = uPrev[i - 1];

        //  Cell-face areas
        Aw = areaP[i];
        Ae = areaP[i + 1];

        //  Flux terms (rho == 1)
        if (i == 0)
            Fw = areaU[i] * UP;
        else
            Fw = 0.5 * Aw * (UP + UW);

        if (i == Nm2)
            Fe = areaU[i] * UP;
        else
            Fe = 0.5 * Ae * (UP + UE);

        //  Upwind difference coefficients
        if (i == 0)
            aWu[i] = 0.0;
        else
            aWu[i] = Dw + max(Fw, 0.0);

        if (i == Nm2)
            aEu[i] = 0.0;
        else
            aEu[i] = De + max(0.0, -Fe);

        if (i == 0)
            aPu[i] = Fe + Fw * 0.5 * P2(areaU[i] / areaP[i]);
        else
            aPu[i] = aWu[i] + aEu[i] + (Fe - Fw);

        aPuinv[i] = 1.0 / aPu[i];

        d[i] = areaU[i] * aPuinv[i];
    }
}

//  build vector of pressure coefficients
inline void buildPressureCoeffs(Pvec &aWp, Pvec &aEp, Pvec &aPp, Pvec &aPpinv, Pvec &bPrimep, const Uvec &d, const Uvec &usource, const Uvec &areaU){
    double Fw{},Fe{};
#pragma omp parallel for private(Fw,Fe)
    for (int i = 1; i < Nm1; i++) {
        aWp[i] = d[i - 1] * areaU[i - 1];
        aEp[i] = d[i] * areaU[i];
        Fw = usource[i - 1] * areaU[i - 1];
        Fe = usource[i] * areaU[i];

        aPp[i] = aWp[i] + aEp[i];
        aPpinv[i] = 1.0 / aPp[i];
        bPrimep[i] = Fw - Fe;
    }
}

//  Check for valid parameter values
inline void checkparams(){
    if (N < 3)
        throwError("Error: invalid grid size.  Pick N > 2\n");

    if (THRESH  < 0 || THRESH > 1)
        throwError("Error: invalid convergence threshold\n");

    if (THRESH > 1E-5)
        printf("Warning: threshold is recommended to be at or below 1E-5\n");

    if (OMEGAu > 1 || OMEGAu < 0 || OMEGAp > 1 || OMEGAp < 0)
        throwError("Error: relaxation should be between 0 and 1\n");

    if (DEBUG  != 0 && DEBUG != 1)
        throwError("Error: invalid debug value.  Pick 0 or 1\n");

    if (printint < 1)
        throwError("Invalid print interval.  Must be greater than 0.");

    if (METHOD != 0 && METHOD != 1)
        throwError("Error: invalid method.  Choose 0 or 1.\n");
}

//  Throw error message and exit
inline void throwError(const string &message) {
    cout << message << endl;
    exit(1);
}

