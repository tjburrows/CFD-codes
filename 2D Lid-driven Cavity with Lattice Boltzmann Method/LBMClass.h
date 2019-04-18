#ifndef LIDDRIVENCAVITYLBM_LBMCLASS_H
#define LIDDRIVENCAVITYLBM_LBMCLASS_H
#include <iostream>
#include <vector>
#include <math.h>
#include <omp.h>

using namespace std;
extern const double             Re;
extern const unsigned int       N;
extern const double             Utop;
extern const double             Ubot;
extern const double             Vlef;
extern const double             Vrig;
extern const int                SAVETXT;
extern const double             THRESH;
extern const int                THREADS;
extern const int                MBounds;
extern const int                MCollide;
extern const int                BC;
extern const unsigned int       Q;
extern const double             MAXITER;
extern const double             NU;
extern const double             TAU;
extern const double             MAGIC;
extern const double             RHO0;
extern const unsigned int       N2;
extern const unsigned int       NQ;
extern const double             w[9];
extern const int                c[9][2];
extern const int                opp[9];
extern const int                half[4];
extern const int                prntInt;
extern const unsigned int       Nm1;
extern const unsigned int       Nm2;
extern const int                GM[9][9];
extern const double             GMinv[9][9];
extern double                   start,stop;

class LBMClass{
public:
    LBMClass(): _f1(NQ,0.0), _f2(NQ,0.0), _fstar(NQ, 0.0),
            _u1(N2, 0.0) ,_u2(N2, 0.0), _rho(N2, RHO0), _x(N, 0.0), _error(100, 0.0),_vmag(N2,0.0),_stress(N2, 0.0),  _vort(N2, 0.0),_df(),_df0(), _filename1(),_filename2(),
            _OMEGA(0.0), _OMEGAm(0.0), _CS(0.0), _MACH(0.0), _rhobar(1.0),_omega_e(),_omega_eps(), _omega_q(),_omega_nu(),  _GS{}
        {

        //  Initialize x
        if (MBounds == 0)
            linspace(_x,(0.5 / N), (N - 0.5) / N,N);
        else
            linspace(_x, 0.0,1.0 ,N);
        _CS = 1.0 / sqrt(3.0);
        _MACH = Utop / _CS;

        _OMEGA   = 1.0 / TAU;
        _OMEGAm  = 1.0 / (0.5 + ((MAGIC) / ((1.0 / (_OMEGA)) - 0.5)));

        _omega_e = 1.1;       //  Bulk viscosity
        _omega_eps = 1.1;     //  free parameter
        _omega_q = 1.1;       //  free parameter
        _omega_nu = _OMEGA;    //  Shear viscosity


        _GS[0] = 0.0;
        _GS[1] = _omega_e;
        _GS[2] = _omega_eps;
        _GS[3] = 0.0;
        _GS[4] = _omega_q;
        _GS[5] = 0.0;
        _GS[6] = _omega_q;
        _GS[7] = _omega_nu;
        _GS[8] = _omega_nu;

        sprintf(_filename1,"Solution_n=%d_Re=%.0f_BCM=%d_CM=%d_U=%0.2f.dat",N,Re,MBounds,MCollide, Utop);
        sprintf(_filename2,"Error_n=%d_Re=%.0f_BCM=%d_CM=%d_U=%0.2f.txt",N,Re,MBounds,MCollide,Utop);

        //  Initialize f
        int ind1,ind2;
#pragma omp parallel for private(ind1,ind2) collapse(2)
        for (unsigned int i = 0; i < N; i++) {
            for (unsigned int j = 0; j < N; j++) {
                ind1 = LU2(i, j);
                for (unsigned int k = 0; k < Q; k++) {
                    ind2 = LU3(i, j, k);
                    _f1[ind2] = calcfeq(k,ind1,1);
                    _f2[ind2] = _f1[ind2];
                }
            }
        }

        //  Print parameters
        printf("Re =\t%.0f\n", Re);
        printf("U =\t%.3e\n", Utop);
        printf("M =\t%.3e\n", Utop * sqrt(3));
        printf("N =\t%d\n", N);
        printf("tau =\t%.3e\n", TAU);
        printf("nu =\t%.3e\n", NU);
    }

    //  Collide Methods
    inline void collideSRT() {
        int ind1{}, ind2{};
        double Fsource{};
#pragma omp parallel for private(ind1, ind2,Fsource) collapse(2)
        for (unsigned int i = 0; i < N; i++) {
            for (unsigned int j = 0; j < N; j++) {
                ind1 = LU2(i, j);
                for (unsigned int k = 0; k < Q; k++) {
                    ind2 = LU3(i, j, k);
                    _fstar[ind2] = (1.0 - _OMEGA) * _f1[ind2] + _OMEGA * calcfeq(k, ind1,1);
                    if (BC == 1){
                        Fsource = (1.0 - 0.5 * _OMEGA) * w[k] *
                                (3.0 * (c[k][0] - _u1[ind1]) + 9.0 * (c[k][0] * _u1[ind1] + c[k][1] * _u2[ind1]) * c[k][0]) * _forceX +
                                (3.0 * (c[k][1] - _u2[ind1]) + 9.0 * (c[k][0] * _u1[ind1] + c[k][1] * _u2[ind1]) * c[k][1]) * _forceY;
                        _fstar[ind2] += Fsource;
                    }
                }
            }
        }

        if (BC == 0 || BC == 1)
            virtualnode();
    }

    inline void collideTRT(){
        double fplus{},fminus{},feqplus{},feqminus{},feq[9]{};
        int ind1{},l{},notl{};
#pragma omp parallel for private(fplus,fminus,feqplus,feqminus,feq,l,notl,ind1) collapse(2)
        for (unsigned int i = 0; i < N; i++){
            for (unsigned int j = 0; j < N; j++){
                ind1 = LU2(i, j);
                for (unsigned int k = 0; k < Q; k++)
                    feq[k] = calcfeq(k,ind1,1);

                //  Rest population
                fplus = _f1[LU3(i,j,0)];
                feqplus = feq[0];
                _fstar[LU3(i, j, 0)] = _f1[LU3(i, j, 0)] - _OMEGA * (fplus - feqplus);

                for (unsigned int k = 0; k < 4; k++) {
                    l = half[k];
                    notl = opp[l];
                    fplus = 0.5 * (_f1[LU3(i,j,l)] + _f1[LU3(i,j,notl)]);
                    fminus = 0.5 * (_f1[LU3(i,j,l)] - _f1[LU3(i,j,notl)]);
                    feqplus = 0.5 * (feq[l] + feq[notl]);
                    feqminus = 0.5 * (feq[l] - feq[notl]);

                    fplus = _OMEGA * (fplus - feqplus);
                    fminus = _OMEGAm * (fminus - feqminus);
                    _fstar[LU3(i, j, l)] = _f1[LU3(i, j, l)] - fplus - fminus;
                    _fstar[LU3(i, j, notl)] = _f1[LU3(i, j, notl)] - fplus + fminus;
                }
            }
        }
        if (BC == 0 || BC == 1)
            virtualnode();
    }

    inline void collideMRT(){
#pragma omp parallel
        {
            int ind1;
            vector<double> _meq(Q,0.0),_mstar(Q,0.0);  //  moments
            double _m{}; // moments
#pragma omp for collapse(2)
            for (unsigned int i = 0; i < N; i++) {
                for (unsigned int j = 0; j < N; j++) {
                    ind1 = LU2(i,j);
                    calcmeq(_meq, _u1[ind1], _u2[ind1], _rho[ind1]);
                    for (unsigned int k = 0; k < Q; k++){
                        _m = GM[k][0] * _f1[LU3(i,j,0)] + GM[k][1] * _f1[LU3(i,j,1)] + GM[k][2] * _f1[LU3(i,j,2)] +GM[k][3] * _f1[LU3(i,j,3)] + GM[k][4] * _f1[LU3(i,j,4)]
                             + GM[k][5] * _f1[LU3(i,j,5)] + GM[k][6] * _f1[LU3(i,j,6)] + GM[k][7] * _f1[LU3(i,j,7)] + GM[k][8] * _f1[LU3(i,j,8)];
                        _mstar[k] = _m - _GS[k] * (_m - _meq[k]);
                    }
                    if (BC == 1) {
                        _mstar[1] += (1.0 - 0.5 * _GS[1]) * (6.0 * (_forceX * _u1[ind1] + _forceY * _u2[ind1]));
                        _mstar[2] += (1.0 - 0.5 * _GS[2]) * (-6.0 * (_forceX * _u1[ind1] + _forceY * _u2[ind1]));
                        _mstar[3] += (1.0 - 0.5 * _GS[3]) * (_forceX);
                        _mstar[4] += (1.0 - 0.5 * _GS[4]) * (-_forceX);
                        _mstar[5] += (1.0 - 0.5 * _GS[5]) * (_forceY);
                        _mstar[6] += (1.0 - 0.5 * _GS[6]) * (-_forceY);
                        _mstar[7] += (1.0 - 0.5 * _GS[7]) * (2.0 * (_forceX * _u1[ind1] - _forceY * _u2[ind1]));
                        _mstar[8] += (1.0 - 0.5 * _GS[8]) * (_forceY * _u1[ind1] + _forceX * _u2[ind1]);
                    }
                    for (unsigned int k = 0; k < Q; k++){
                        _fstar[LU3(i, j, k)] = GMinv[k][0] * _mstar[0] + GMinv[k][1] * _mstar[1] + GMinv[k][2] * _mstar[2] + GMinv[k][3] * _mstar[3]
                                               + GMinv[k][4] * _mstar[4] + GMinv[k][5] * _mstar[5] + GMinv[k][6] * _mstar[6] + GMinv[k][7] * _mstar[7] + GMinv[k][8] * _mstar[8];
                    }
                }
            }
        }
        if (BC == 0 || BC == 1)
            virtualnode();
    }

    //  Stream Methods
    inline void streamPush() {
#pragma omp parallel
        {
            int inew, jnew;
#pragma omp for collapse(2)
            for (unsigned int i = 0; i < N; i++) {
                for (unsigned int j = 0; j < N; j++) {
                    for (unsigned int k = 0; k < Q; k++) {
                        inew = i + c[k][0];
                        jnew = j + c[k][1];
                        if (inew < N && inew >= 0 && jnew < N && jnew >= 0)
                            _f2[LU3(inew, jnew, k)] = _fstar[LU3(i, j, k)];
                    }
                }
            }
        }
    }

    inline void streamPull() {
#pragma omp parallel
        {
            int iold, jold;
#pragma omp for collapse(2)
            for (unsigned int i = 0; i < N; i++) {
                for (unsigned int j = 0; j < N; j++) {
                    for (unsigned int k = 0; k < Q; k++) {
                        iold = i - c[k][0];
                        jold = j - c[k][1];
                        if (iold < N && iold >= 0 && jold < N && jold >= 0)
                            _f2[LU3(i, j, k)] = _fstar[LU3(iold, jold, k)];
                    }
                }
            }
        }
    }

    //  Bounday Condition Methods
    inline void NEBB(){
        switch (BC){

            case 2 : {  //  Lid-driven cavity flow
                const double sixth = 1.0 / 6.0, twothirds = 2.0 / 3.0, twelfth = 1.0 / 12.0;
                double rho{_rhobar};
                for (unsigned int i = 1; i < N - 1; i++) {
                    //  Left wall, general case
                    rho = _f2[LU3(0, i, 0)] + _f2[LU3(0, i, 2)] + _f2[LU3(0, i, 4)] + 2.0 * (_f2[LU3(0, i, 3)] + _f2[LU3(0, i, 6)] + _f2[LU3(0, i, 7)]);
                    _f2[LU3(0, i, 1)] = _f2[LU3(0, i, 3)];
                    _f2[LU3(0, i, 5)] = _f2[LU3(0, i, 7)] - 0.5 * (_f2[LU3(0, i, 2)] - _f2[LU3(0, i, 4)]) + 0.5 * rho * Vlef;
                    _f2[LU3(0, i, 8)] = _f2[LU3(0, i, 6)] + 0.5 * (_f2[LU3(0, i, 2)] - _f2[LU3(0, i, 4)]) - 0.5 * rho * Vlef;

                    //  Right wall, general case
                    rho = _f2[LU3(Nm1, i, 0)] + _f2[LU3(Nm1, i, 2)] + _f2[LU3(Nm1, i, 4)] + 2.0 * (_f2[LU3(Nm1, i, 1)] + _f2[LU3(Nm1, i, 5)] + _f2[LU3(Nm1, i, 8)]);
                    _f2[LU3(Nm1, i, 3)] = _f2[LU3(Nm1, i, 1)];
                    _f2[LU3(Nm1, i, 7)] = _f2[LU3(Nm1, i, 5)] + 0.5 * (_f2[LU3(Nm1, i, 2)] - _f2[LU3(Nm1, i, 4)]) - 0.5 * rho * Vrig;
                    _f2[LU3(Nm1, i, 6)] = _f2[LU3(Nm1, i, 8)] - 0.5 * (_f2[LU3(Nm1, i, 2)] - _f2[LU3(Nm1, i, 4)]) + 0.5 * rho * Vrig;

                    //  Top wall, general case
                    rho = _f2[LU3(i, Nm1, 0)] + _f2[LU3(i, Nm1, 1)] + _f2[LU3(i, Nm1, 3)] + 2.0 * (_f2[LU3(i, Nm1, 2)] + _f2[LU3(i, Nm1, 6)] + _f2[LU3(i, Nm1, 5)]);
                    _f2[LU3(i, Nm1, 4)] = _f2[LU3(i, Nm1, 2)];
                    _f2[LU3(i, Nm1, 7)] = _f2[LU3(i, Nm1, 5)] + 0.5 * (_f2[LU3(i, Nm1, 1)] - _f2[LU3(i, Nm1, 3)]) - 0.5 * rho * Utop;
                    _f2[LU3(i, Nm1, 8)] = _f2[LU3(i, Nm1, 6)] - 0.5 * (_f2[LU3(i, Nm1, 1)] - _f2[LU3(i, Nm1, 3)]) + 0.5 * rho * Utop;

                    //  Bottom wall, general case
                    rho = _f2[LU3(i, 0, 0)] + _f2[LU3(i, 0, 1)] + _f2[LU3(i, 0, 3)] + 2.0 * (_f2[LU3(i, 0, 4)] + _f2[LU3(i, 0, 7)] + _f2[LU3(i, 0, 8)]);
                    _f2[LU3(i, 0, 2)] = _f2[LU3(i, 0, 4)];
                    _f2[LU3(i, 0, 5)] = _f2[LU3(i, 0, 7)] - 0.5 * (_f2[LU3(i, 0, 1)] - _f2[LU3(i, 0, 3)]) + 0.5 * rho * Ubot;
                    _f2[LU3(i, 0, 6)] = _f2[LU3(i, 0, 8)] + 0.5 * (_f2[LU3(i, 0, 1)] - _f2[LU3(i, 0, 3)]) - 0.5 * rho * Ubot;
                }

                // Corners
                rho = 1.0;
                //  Bottom Left (0,0) knowns: 1,5,2, unknowns: 0,6,8
                _f2[LU3(0, 0, 1)] = _f2[LU3(0, 0, 3)] + twothirds * rho * Ubot;
                _f2[LU3(0, 0, 2)] = _f2[LU3(0, 0, 4)] + twothirds * rho * Vlef;
                _f2[LU3(0, 0, 5)] = _f2[LU3(0, 0, 7)] + sixth * rho * (Ubot + Vlef);
                _f2[LU3(0, 0, 6)] = twelfth * rho * (Vlef - Ubot);
                _f2[LU3(0, 0, 8)] = -_f2[LU3(0, 0, 6)];
                _f2[LU3(0, 0, 0)] = rho - (_f2[LU3(0, 0, 1)] + _f2[LU3(0, 0, 2)] + _f2[LU3(0, 0, 3)] + _f2[LU3(0, 0, 4)] + _f2[LU3(0, 0, 5)] + _f2[LU3(0, 0, 6)] + _f2[LU3(0, 0, 7)] + _f2[LU3(0, 0, 8)]);

                //  Bottom Right (Nm1,0) knowns: 2,3,6, unknowns: 0, 5, 7
//                rho = 0.5 * (_rho[LU2(Nm2,0)] + _rho[LU2(Nm1,1)]);
                _f2[LU3(Nm1, 0, 2)] = _f2[LU3(Nm1, 0, 4)] + twothirds * rho * Vrig;
                _f2[LU3(Nm1, 0, 3)] = _f2[LU3(Nm1, 0, 1)] - twothirds * rho * Ubot;
                _f2[LU3(Nm1, 0, 6)] = _f2[LU3(Nm1, 0, 8)] + sixth * rho * (-Ubot + Vrig);
                _f2[LU3(Nm1, 0, 5)] = twelfth * rho * (Vrig + Ubot);
                _f2[LU3(Nm1, 0, 7)] = -_f2[LU3(Nm1, 0, 5)];
                _f2[LU3(Nm1, 0, 0)] = rho - (_f2[LU3(Nm1, 0, 1)] + _f2[LU3(Nm1, 0, 2)] + _f2[LU3(Nm1, 0, 3)] + _f2[LU3(Nm1, 0, 4)] + _f2[LU3(Nm1, 0, 5)] + _f2[LU3(Nm1, 0, 6)] + _f2[LU3(Nm1, 0, 7)] + _f2[LU3(Nm1, 0, 8)]);

                //  Top Left (0,Nm1) knowns: 1,4,8, unknowns: 0, 5, 7
//                rho = 0.5 * (_rho[LU2(0, Nm2)] + _rho[LU2(1, Nm1)]);
                _f2[LU3(0, Nm1, 1)] = _f2[LU3(0, Nm1, 3)] + twothirds * rho * Utop;
                _f2[LU3(0, Nm1, 4)] = _f2[LU3(0, Nm1, 2)] - twothirds * rho * Vlef;
                _f2[LU3(0, Nm1, 8)] = _f2[LU3(0, Nm1, 6)] + sixth * rho * (Utop - Vlef);
                _f2[LU3(0, Nm1, 5)] = twelfth * rho * (Vlef + Utop);
                _f2[LU3(0, Nm1, 7)] = -_f2[LU3(0, Nm1, 5)];
                _f2[LU3(0, Nm1, 0)] = rho - (_f2[LU3(0, Nm1, 1)] + _f2[LU3(0, Nm1, 2)] + _f2[LU3(0, Nm1, 3)] + _f2[LU3(0, Nm1, 4)] + _f2[LU3(0, Nm1, 5)] + _f2[LU3(0, Nm1, 6)] + _f2[LU3(0, Nm1, 7)] + _f2[LU3(0, Nm1, 8)]);

                //  Top Right (Nm1,Nm1) knowns: 3,7,4, unknowns: 0, 6, 8
//                rho = 0.5 * (_rho[LU2(Nm2, Nm1)] + _rho[LU2(Nm1, Nm2)]);
                _f2[LU3(Nm1, Nm1, 4)] = _f2[LU3(Nm1, Nm1, 2)] - twothirds * rho * Vrig;
                _f2[LU3(Nm1, Nm1, 3)] = _f2[LU3(Nm1, Nm1, 1)] - twothirds * rho * Utop;
                _f2[LU3(Nm1, Nm1, 7)] = _f2[LU3(Nm1, Nm1, 5)] - sixth * rho * (Utop + Vrig);
                _f2[LU3(Nm1, Nm1, 6)] = twelfth * rho * (Vrig - Utop);
                _f2[LU3(Nm1, Nm1, 8)] = -_f2[LU3(Nm1, Nm1, 6)];
                _f2[LU3(Nm1, Nm1, 0)] = rho - (_f2[LU3(Nm1, Nm1, 1)] + _f2[LU3(Nm1, Nm1, 2)] + _f2[LU3(Nm1, Nm1, 3)] + _f2[LU3(Nm1, Nm1, 4)] + _f2[LU3(Nm1, Nm1, 5)] + _f2[LU3(Nm1, Nm1, 6)] + _f2[LU3(Nm1, Nm1, 7)] + _f2[LU3(Nm1, Nm1, 8)]);
                break;
            }

            case 1: {   //  Poiseuille Flow
                for (unsigned int i = 0; i < N; i++) {
                    //  Top wall, general case
                    _f2[LU3(i, Nm1, 4)] = _f2[LU3(i, Nm1, 2)];
                    _f2[LU3(i, Nm1, 7)] = _f2[LU3(i, Nm1, 5)] + 0.5 * (_f2[LU3(i, Nm1, 1)] - _f2[LU3(i, Nm1, 3)]);
                    _f2[LU3(i, Nm1, 8)] = _f2[LU3(i, Nm1, 6)] - 0.5 * (_f2[LU3(i, Nm1, 1)] - _f2[LU3(i, Nm1, 3)]);

                    //  Bottom wall, general case
                    _f2[LU3(i, 0, 2)] = _f2[LU3(i, 0, 4)];
                    _f2[LU3(i, 0, 5)] = _f2[LU3(i, 0, 7)] - 0.5 * (_f2[LU3(i, 0, 1)] - _f2[LU3(i, 0, 3)]);
                    _f2[LU3(i, 0, 6)] = _f2[LU3(i, 0, 8)] + 0.5 * (_f2[LU3(i, 0, 1)] - _f2[LU3(i, 0, 3)]);
                }
                break;
            }

            case 0: {   //  Couette Flow
                double rho{};
                for (unsigned int i = 0; i < N; i++) {
                    //  Top wall, general case
                    rho = _f2[LU3(i, Nm1, 0)] + _f2[LU3(i, Nm1, 1)] + _f2[LU3(i, Nm1, 3)] + 2.0 * (_f2[LU3(i, Nm1, 2)] + _f2[LU3(i, Nm1, 6)] + _f2[LU3(i, Nm1, 5)]);
                    _f2[LU3(i, Nm1, 4)] = _f2[LU3(i, Nm1, 2)];
                    _f2[LU3(i, Nm1, 7)] = _f2[LU3(i, Nm1, 5)] + 0.5 * (_f2[LU3(i, Nm1, 1)] - _f2[LU3(i, Nm1, 3)]) - 0.5 * rho * Utop;
                    _f2[LU3(i, Nm1, 8)] = _f2[LU3(i, Nm1, 6)] - 0.5 * (_f2[LU3(i, Nm1, 1)] - _f2[LU3(i, Nm1, 3)]) + 0.5 * rho * Utop;

                    //  Bottom wall, general case
                    rho = _f2[LU3(i, 0, 0)] + _f2[LU3(i, 0, 1)] + _f2[LU3(i, 0, 3)] + 2.0 * (_f2[LU3(i, 0, 4)] + _f2[LU3(i, 0, 7)] + _f2[LU3(i, 0, 8)]);
                    _f2[LU3(i, 0, 2)] = _f2[LU3(i, 0, 4)];
                    _f2[LU3(i, 0, 5)] = _f2[LU3(i, 0, 7)] - 0.5 * (_f2[LU3(i, 0, 1)] - _f2[LU3(i, 0, 3)]) + 0.5 * rho * Ubot;
                    _f2[LU3(i, 0, 6)] = _f2[LU3(i, 0, 8)] + 0.5 * (_f2[LU3(i, 0, 1)] - _f2[LU3(i, 0, 3)]) - 0.5 * rho * Ubot;
                }
                break;
            }

            default: {
                std::printf("Error: Invalid Boundary condition case number\n");
                exit(1);
            }
        }

    }


    //  Non-equilibrium extrapolation
    inline void NEE() {
        switch (BC){

            case 2: {   //  Lid-driven cavity flow
#pragma omp parallel
                {
                    double rhof{},u1f{},u2f{};
                    double rhowall{1.0};
#pragma omp for
                    for (unsigned int i = 1; i < N-1; i++){
                        //  Bottom wall, (i, 0)
                        rhowall = _f2[LU3(i,0,0)] +_f2[LU3(i,0,1)] +_f2[LU3(i,0,3)] + 2.0 * (_f2[LU3(i,0,4)] +_f2[LU3(i,0,7)] +_f2[LU3(i,0,8)]);
                        rhof = _f2[LU3(i,1,0)] + _f2[LU3(i,1,1)] + _f2[LU3(i,1,2)] + _f2[LU3(i,1,3)] + _f2[LU3(i,1,4)] + _f2[LU3(i,1,5)]+ _f2[LU3(i,1,6)]+ _f2[LU3(i,1,7)]+ _f2[LU3(i,1,8)];
                        u1f = ((_f2[LU3(i,1,1)] + _f2[LU3(i,1,5)] + _f2[LU3(i,1,8)]) - (_f2[LU3(i,1,3)] + _f2[LU3(i,1,6)] + _f2[LU3(i,1,7)])) / rhof;
                        u2f = ((_f2[LU3(i,1,2)] + _f2[LU3(i,1,5)] + _f2[LU3(i,1,6)]) - (_f2[LU3(i,1,4)] + _f2[LU3(i,1,7)] + _f2[LU3(i,1,8)])) / rhof;
                        for (unsigned int k = 0; k < Q; k++)
                            _f2[LU3(i,0,k)] = calcfeq(k,Ubot,0.0,rhowall,1) + (_f2[LU3(i,1,k)] - calcfeq(k, u1f, u2f, rhof,1));

                        //  Top Wall, (i,Nm1)
                        rhowall = _f2[LU3(i,Nm1,0)] +_f2[LU3(i,Nm1,1)] +_f2[LU3(i,Nm1,3)] + 2.0 * (_f2[LU3(i,Nm1,2)] +_f2[LU3(i,Nm1,6)] +_f2[LU3(i,Nm1,5)]);
                        rhof = _f2[LU3(i,Nm2,0)] + _f2[LU3(i,Nm2,1)] + _f2[LU3(i,Nm2,2)] + _f2[LU3(i,Nm2,3)] + _f2[LU3(i,Nm2,4)] + _f2[LU3(i,Nm2,5)]+ _f2[LU3(i,Nm2,6)]+ _f2[LU3(i,Nm2,7)]+ _f2[LU3(i,Nm2,8)];
                        u1f = ((_f2[LU3(i,Nm2,1)] + _f2[LU3(i,Nm2,5)] + _f2[LU3(i,Nm2,8)]) - (_f2[LU3(i,Nm2,3)] + _f2[LU3(i,Nm2,6)] + _f2[LU3(i,Nm2,7)])) / rhof;
                        u2f = ((_f2[LU3(i,Nm2,2)] + _f2[LU3(i,Nm2,5)] + _f2[LU3(i,Nm2,6)]) - (_f2[LU3(i,Nm2,4)] + _f2[LU3(i,Nm2,7)] + _f2[LU3(i,Nm2,8)])) / rhof;
                        for (unsigned int k = 0; k < Q; k++)
                            _f2[LU3(i,Nm1,k)] = calcfeq(k,Utop,0.0,rhowall,1) + (_f2[LU3(i,Nm2,k)] - calcfeq(k, u1f, u2f, rhof,1));

                        //  Left wall, (0,i)
                        rhowall = _f2[LU3(0,i,0)] +_f2[LU3(0,i,2)] +_f2[LU3(0,i,4)] + 2.0 * (_f2[LU3(0,i,3)] +_f2[LU3(0,i,6)] +_f2[LU3(0,i,7)]);
                        rhof = _f2[LU3(1,i,0)] + _f2[LU3(1,i,1)] + _f2[LU3(1,i,2)] + _f2[LU3(1,i,3)] + _f2[LU3(1,i,4)] + _f2[LU3(1,i,5)]+ _f2[LU3(1,i,6)]+ _f2[LU3(1,i,7)]+ _f2[LU3(1,i,8)];
                        u1f = ((_f2[LU3(1,i,1)] + _f2[LU3(1,i,5)] + _f2[LU3(1,i,8)]) - (_f2[LU3(1,i,3)] + _f2[LU3(1,i,6)] + _f2[LU3(1,i,7)])) / rhof;
                        u2f = ((_f2[LU3(1,i,2)] + _f2[LU3(1,i,5)] + _f2[LU3(1,i,6)]) - (_f2[LU3(1,i,4)] + _f2[LU3(1,i,7)] + _f2[LU3(1,i,8)])) / rhof;
                        for (unsigned int k = 0; k < Q; k++)
                            _f2[LU3(0,i,k)] = calcfeq(k,0.0,Vlef,rhowall,1) + (_f2[LU3(1,i,k)] - calcfeq(k, u1f, u2f, rhof,1));

                        //  Right Wall (Nm1, i)
                        rhowall = _f2[LU3(Nm1,i,0)] +_f2[LU3(Nm1,i,2)] +_f2[LU3(Nm1,i,4)] + 2.0 * (_f2[LU3(Nm1,i,1)] +_f2[LU3(Nm1,i,5)] +_f2[LU3(Nm1,i,8)]);
                        rhof = _f2[LU3(Nm2,i,0)] + _f2[LU3(Nm2,i,1)] + _f2[LU3(Nm2,i,2)] + _f2[LU3(Nm2,i,3)] + _f2[LU3(Nm2,i,4)] + _f2[LU3(Nm2,i,5)]+ _f2[LU3(Nm2,i,6)]+ _f2[LU3(Nm2,i,7)]+ _f2[LU3(Nm2,i,8)];
                        u1f = ((_f2[LU3(Nm2,i,1)] + _f2[LU3(Nm2,i,5)] + _f2[LU3(Nm2,i,8)]) - (_f2[LU3(Nm2,i,3)] + _f2[LU3(Nm2,i,6)] + _f2[LU3(Nm2,i,7)])) / rhof;
                        u2f = ((_f2[LU3(Nm2,i,2)] + _f2[LU3(Nm2,i,5)] + _f2[LU3(Nm2,i,6)]) - (_f2[LU3(Nm2,i,4)] + _f2[LU3(Nm2,i,7)] + _f2[LU3(Nm2,i,8)])) / rhof;
                        for (unsigned int k = 0; k < Q; k++)
                            _f2[LU3(Nm1,i,k)] = calcfeq(k,0.0,Vrig,rhowall,1) + (_f2[LU3(Nm2,i,k)] - calcfeq(k, u1f, u2f, rhof,1));
                    }

                    //  Corners
                    //  Bottom left (0,0)
                    rhof = _f2[LU3(1,1,0)] + _f2[LU3(1,1,1)] + _f2[LU3(1,1,2)] + _f2[LU3(1,1,3)] + _f2[LU3(1,1,4)] + _f2[LU3(1,1,5)]+ _f2[LU3(1,1,6)]+ _f2[LU3(1,1,7)]+ _f2[LU3(1,1,8)];
                    u1f = ((_f2[LU3(1,1,1)] + _f2[LU3(1,1,5)] + _f2[LU3(1,1,8)]) - (_f2[LU3(1,1,3)] + _f2[LU3(1,1,6)] + _f2[LU3(1,1,7)])) / rhof;
                    u2f = ((_f2[LU3(1,1,2)] + _f2[LU3(1,1,5)] + _f2[LU3(1,1,6)]) - (_f2[LU3(1,1,4)] + _f2[LU3(1,1,7)] + _f2[LU3(1,1,8)])) / rhof;
                    rhowall = 1.0;
#pragma omp for
                    for (unsigned int k = 0; k < Q; k++)
                        _f2[LU3(0,0,k)] = calcfeq(k,Ubot,Vlef,rhowall,1) + (_f2[LU3(1,1,k)] - calcfeq(k, u1f, u2f, rhof,1));

                    //  Bottom Right (Nm1,0)
                    rhof = _f2[LU3(Nm2,1,0)] + _f2[LU3(Nm2,1,1)] + _f2[LU3(Nm2,1,2)] + _f2[LU3(Nm2,1,3)] + _f2[LU3(Nm2,1,4)] + _f2[LU3(Nm2,1,5)]+ _f2[LU3(Nm2,1,6)]+ _f2[LU3(Nm2,1,7)]+ _f2[LU3(Nm2,1,8)];
                    u1f = ((_f2[LU3(Nm2,1,1)] + _f2[LU3(Nm2,1,5)] + _f2[LU3(Nm2,1,8)]) - (_f2[LU3(Nm2,1,3)] + _f2[LU3(Nm2,1,6)] + _f2[LU3(Nm2,1,7)])) / rhof;
                    u2f = ((_f2[LU3(Nm2,1,2)] + _f2[LU3(Nm2,1,5)] + _f2[LU3(Nm2,1,6)]) - (_f2[LU3(Nm2,1,4)] + _f2[LU3(Nm2,1,7)] + _f2[LU3(Nm2,1,8)])) / rhof;
#pragma omp for
                    for (unsigned int k = 0; k < Q; k++)
                        _f2[LU3(Nm1,0,k)] = calcfeq(k,Ubot,Vrig,rhowall,1) + (_f2[LU3(Nm2,1,k)] - calcfeq(k, u1f, u2f, rhof,1));

                    //  Top Right (Nm1,Nm1)
                    rhof = _f2[LU3(Nm2,Nm2,0)] + _f2[LU3(Nm2,Nm2,1)] + _f2[LU3(Nm2,Nm2,2)] + _f2[LU3(Nm2,Nm2,3)] + _f2[LU3(Nm2,Nm2,4)] + _f2[LU3(Nm2,Nm2,5)]+ _f2[LU3(Nm2,Nm2,6)]+ _f2[LU3(Nm2,Nm2,7)]+ _f2[LU3(Nm2,Nm2,8)];
                    u1f = ((_f2[LU3(Nm2,Nm2,1)] + _f2[LU3(Nm2,Nm2,5)] + _f2[LU3(Nm2,Nm2,8)]) - (_f2[LU3(Nm2,Nm2,3)] + _f2[LU3(Nm2,Nm2,6)] + _f2[LU3(Nm2,Nm2,7)])) / rhof;
                    u2f = ((_f2[LU3(Nm2,Nm2,2)] + _f2[LU3(Nm2,Nm2,5)] + _f2[LU3(Nm2,Nm2,6)]) - (_f2[LU3(Nm2,Nm2,4)] + _f2[LU3(Nm2,Nm2,7)] + _f2[LU3(Nm2,Nm2,8)])) / rhof;
#pragma omp for
                    for (unsigned int k = 0; k < Q; k++)
                        _f2[LU3(Nm1,Nm1,k)] = calcfeq(k,Utop,Vrig,rhowall,1) + (_f2[LU3(Nm2,Nm2,k)] - calcfeq(k, u1f, u2f, rhof,1));

                    //  Top Left (0,Nm1)
                    rhof = _f2[LU3(1,Nm2,0)] + _f2[LU3(1,Nm2,1)] + _f2[LU3(1,Nm2,2)] + _f2[LU3(1,Nm2,3)] + _f2[LU3(1,Nm2,4)] + _f2[LU3(1,Nm2,5)]+ _f2[LU3(1,Nm2,6)]+ _f2[LU3(1,Nm2,7)]+ _f2[LU3(1,Nm2,8)];
                    u1f = ((_f2[LU3(1,Nm2,1)] + _f2[LU3(1,Nm2,5)] + _f2[LU3(1,Nm2,8)]) - (_f2[LU3(1,Nm2,3)] + _f2[LU3(1,Nm2,6)] + _f2[LU3(1,Nm2,7)])) / rhof;
                    u2f = ((_f2[LU3(1,Nm2,2)] + _f2[LU3(1,Nm2,5)] + _f2[LU3(1,Nm2,6)]) - (_f2[LU3(1,Nm2,4)] + _f2[LU3(1,Nm2,7)] + _f2[LU3(1,Nm2,8)])) / rhof;
#pragma omp for
                    for (unsigned int k = 0; k < Q; k++)
                        _f2[LU3(0,Nm1,k)] = calcfeq(k,Utop,Vlef,rhowall,1) + (_f2[LU3(1,Nm2,k)] - calcfeq(k, u1f, u2f, rhof,1));
                };
                break;
                }

            case 1: {   //  Poiseuille Flow
#pragma omp parallel
                {
                    double rhof{},u1f{},u2f{};
                    double rhowall{1.0};
#pragma omp for
                    for (unsigned int i = 1; i < N-1; i++){
                        //  Bottom wall, (i, 0)
                        rhowall = _f2[LU3(i,0,0)] +_f2[LU3(i,0,1)] +_f2[LU3(i,0,3)] + 2.0 * (_f2[LU3(i,0,4)] +_f2[LU3(i,0,7)] +_f2[LU3(i,0,8)]);
                        rhof = _f2[LU3(i,1,0)] + _f2[LU3(i,1,1)] + _f2[LU3(i,1,2)] + _f2[LU3(i,1,3)] + _f2[LU3(i,1,4)] + _f2[LU3(i,1,5)]+ _f2[LU3(i,1,6)]+ _f2[LU3(i,1,7)]+ _f2[LU3(i,1,8)];
                        u1f = ((_f2[LU3(i,1,1)] + _f2[LU3(i,1,5)] + _f2[LU3(i,1,8)]) - (_f2[LU3(i,1,3)] + _f2[LU3(i,1,6)] + _f2[LU3(i,1,7)])) / rhof;
                        u2f = ((_f2[LU3(i,1,2)] + _f2[LU3(i,1,5)] + _f2[LU3(i,1,6)]) - (_f2[LU3(i,1,4)] + _f2[LU3(i,1,7)] + _f2[LU3(i,1,8)])) / rhof;
                        for (unsigned int k = 0; k < Q; k++)
                            _f2[LU3(i,0,k)] = calcfeq(k,0.0,0.0,rhowall,1) + (_f2[LU3(i,1,k)] - calcfeq(k, u1f, u2f, rhof,1));

                        //  Top Wall, (i,Nm1)
                        rhowall = _f2[LU3(i,Nm1,0)] +_f2[LU3(i,Nm1,1)] +_f2[LU3(i,Nm1,3)] + 2.0 * (_f2[LU3(i,Nm1,2)] +_f2[LU3(i,Nm1,6)] +_f2[LU3(i,Nm1,5)]);
                        rhof = _f2[LU3(i,Nm2,0)] + _f2[LU3(i,Nm2,1)] + _f2[LU3(i,Nm2,2)] + _f2[LU3(i,Nm2,3)] + _f2[LU3(i,Nm2,4)] + _f2[LU3(i,Nm2,5)]+ _f2[LU3(i,Nm2,6)]+ _f2[LU3(i,Nm2,7)]+ _f2[LU3(i,Nm2,8)];
                        u1f = ((_f2[LU3(i,Nm2,1)] + _f2[LU3(i,Nm2,5)] + _f2[LU3(i,Nm2,8)]) - (_f2[LU3(i,Nm2,3)] + _f2[LU3(i,Nm2,6)] + _f2[LU3(i,Nm2,7)])) / rhof;
                        u2f = ((_f2[LU3(i,Nm2,2)] + _f2[LU3(i,Nm2,5)] + _f2[LU3(i,Nm2,6)]) - (_f2[LU3(i,Nm2,4)] + _f2[LU3(i,Nm2,7)] + _f2[LU3(i,Nm2,8)])) / rhof;
                        for (unsigned int k = 0; k < Q; k++)
                            _f2[LU3(i,Nm1,k)] = calcfeq(k,0.0,0.0,rhowall,1) + (_f2[LU3(i,Nm2,k)] - calcfeq(k, u1f, u2f, rhof,1));
                    }
                }
                break;
            }

            case 0: {   //  Couette Flow
#pragma omp parallel
                {
                double rhof{},u1f{},u2f{};
                double rhowall{1.0};
#pragma omp for
                for (unsigned int i = 1; i < N-1; i++){
                    //  Bottom wall, (i, 0)
                    rhowall = _f2[LU3(i,0,0)] +_f2[LU3(i,0,1)] +_f2[LU3(i,0,3)] + 2.0 * (_f2[LU3(i,0,4)] +_f2[LU3(i,0,7)] +_f2[LU3(i,0,8)]);
                    rhof = _f2[LU3(i,1,0)] + _f2[LU3(i,1,1)] + _f2[LU3(i,1,2)] + _f2[LU3(i,1,3)] + _f2[LU3(i,1,4)] + _f2[LU3(i,1,5)]+ _f2[LU3(i,1,6)]+ _f2[LU3(i,1,7)]+ _f2[LU3(i,1,8)];
                    u1f = ((_f2[LU3(i,1,1)] + _f2[LU3(i,1,5)] + _f2[LU3(i,1,8)]) - (_f2[LU3(i,1,3)] + _f2[LU3(i,1,6)] + _f2[LU3(i,1,7)])) / rhof;
                    u2f = ((_f2[LU3(i,1,2)] + _f2[LU3(i,1,5)] + _f2[LU3(i,1,6)]) - (_f2[LU3(i,1,4)] + _f2[LU3(i,1,7)] + _f2[LU3(i,1,8)])) / rhof;
                    for (unsigned int k = 0; k < Q; k++)
                        _f2[LU3(i,0,k)] = calcfeq(k,Ubot,0.0,rhowall,1) + (_f2[LU3(i,1,k)] - calcfeq(k, u1f, u2f, rhof,1));

                    //  Top Wall, (i,Nm1)
                    rhowall = _f2[LU3(i,Nm1,0)] +_f2[LU3(i,Nm1,1)] +_f2[LU3(i,Nm1,3)] + 2.0 * (_f2[LU3(i,Nm1,2)] +_f2[LU3(i,Nm1,6)] +_f2[LU3(i,Nm1,5)]);
                    rhof = _f2[LU3(i,Nm2,0)] + _f2[LU3(i,Nm2,1)] + _f2[LU3(i,Nm2,2)] + _f2[LU3(i,Nm2,3)] + _f2[LU3(i,Nm2,4)] + _f2[LU3(i,Nm2,5)]+ _f2[LU3(i,Nm2,6)]+ _f2[LU3(i,Nm2,7)]+ _f2[LU3(i,Nm2,8)];
                    u1f = ((_f2[LU3(i,Nm2,1)] + _f2[LU3(i,Nm2,5)] + _f2[LU3(i,Nm2,8)]) - (_f2[LU3(i,Nm2,3)] + _f2[LU3(i,Nm2,6)] + _f2[LU3(i,Nm2,7)])) / rhof;
                    u2f = ((_f2[LU3(i,Nm2,2)] + _f2[LU3(i,Nm2,5)] + _f2[LU3(i,Nm2,6)]) - (_f2[LU3(i,Nm2,4)] + _f2[LU3(i,Nm2,7)] + _f2[LU3(i,Nm2,8)])) / rhof;
                    for (unsigned int k = 0; k < Q; k++)
                        _f2[LU3(i,Nm1,k)] = calcfeq(k,Utop,0.0,rhowall,1) + (_f2[LU3(i,Nm2,k)] - calcfeq(k, u1f, u2f, rhof,1));
                }
                }
                break;
            }

            default: {
                std::printf("Error: Invalid Boundary condition case number\n");
                exit(1);
            }
            }

    };
    inline void HWBB(){
        double rhowall = 1.0;
        switch (BC){
            case 2: {    //  Lid-driven cavity flow
#pragma omp parallel for
                for (unsigned int i = 1; i < N-1; i++){
                    //  Bottom wall
                    _f2[LU3(i, 0, 6)] = _fstar[LU3(i, 0, 8)] - 2.0 * w[8] * rhowall * c[8][0] * Ubot * 3.0;
                    _f2[LU3(i, 0, 2)] = _fstar[LU3(i, 0, 4)];
                    _f2[LU3(i, 0, 5)] = _fstar[LU3(i, 0, 7)] - 2.0 * w[7] * rhowall * c[7][0] * Ubot * 3.0;

                    //  Top wall
                    _f2[LU3(i, Nm1, 8)] = _fstar[LU3(i, Nm1, 6)] - 2.0 * w[6] * rhowall * c[6][0] * Utop * 3.0;
                    _f2[LU3(i, Nm1, 4)] = _fstar[LU3(i, Nm1, 2)];
                    _f2[LU3(i, Nm1, 7)] = _fstar[LU3(i, Nm1, 5)] - 2.0 * w[5] * rhowall * c[5][0] * Utop * 3.0;

                    //  Left wall
                    _f2[LU3(0, i, 5)] = _fstar[LU3(0, i, 7)] - 2.0 * w[7] * rhowall * c[7][1] * Vlef * 3.0;
                    _f2[LU3(0, i, 1)] = _fstar[LU3(0, i, 3)];
                    _f2[LU3(0, i, 8)] = _fstar[LU3(0, i, 6)] - 2.0 * w[6] * rhowall * c[6][1] * Vlef * 3.0;

                    //  Right wall
                    _f2[LU3(Nm1, i, 7)] = _fstar[LU3(Nm1, i, 5)] - 2.0 * w[5] * rhowall * c[5][1] * Vrig * 3.0;
                    _f2[LU3(Nm1, i, 3)] = _fstar[LU3(Nm1, i, 1)];
                    _f2[LU3(Nm1, i, 6)] = _fstar[LU3(Nm1, i, 8)] - 2.0 * w[8] * rhowall * c[8][1] * Vrig * 3.0;
                }

                //  Corners
                //  Top left
                _f2[LU3(0, Nm1, 1)] = _fstar[LU3(0, Nm1, 3)];
                _f2[LU3(0, Nm1, 8)] = _fstar[LU3(0, Nm1, 6)] - 2.0 * w[6] * rhowall * (c[6][0] * Utop + c[6][1] * Vlef) * 3.0;
                _f2[LU3(0, Nm1, 4)] = _fstar[LU3(0, Nm1, 2)];

                //  Bottom Left
                _f2[LU3(0, 0, 1)] = _fstar[LU3(0, 0, 3)];
                _f2[LU3(0, 0, 5)] = _fstar[LU3(0, 0, 7)] - 2.0 * w[7] * rhowall * (c[7][0] * Ubot + c[7][1] * Vlef) * 3.0;
                _f2[LU3(0, 0, 2)] = _fstar[LU3(0, 0, 4)];

                //  Top Right
                _f2[LU3(Nm1, Nm1, 3)] = _fstar[LU3(Nm1, Nm1, 1)];
                _f2[LU3(Nm1, Nm1, 7)] = _fstar[LU3(Nm1, Nm1, 5)] - 2.0 * w[5] * rhowall * (c[5][0] * Utop + c[5][1] * Vrig) * 3.0;
                _f2[LU3(Nm1, Nm1, 4)] = _fstar[LU3(Nm1, Nm1, 2)];

                //  Bottom Right
                _f2[LU3(Nm1, 0, 3)] = _fstar[LU3(Nm1, 0, 1)];
                _f2[LU3(Nm1, 0, 6)] = _fstar[LU3(Nm1, 0, 8)] - 2.0 * w[8] * rhowall * (c[8][0] * Ubot + c[8][1] * Vrig) * 3.0;
                _f2[LU3(Nm1, 0, 2)] = _fstar[LU3(Nm1, 0, 4)];
                break;
            }

            case 1: {   //  Poiseuille Flow
#pragma omp parallel for
                for (unsigned int i = 0; i < N; i++){
                    //  Bottom wall
                    _f2[LU3(i, 0, 6)] = _fstar[LU3(i, 0, 8)];
                    _f2[LU3(i, 0, 2)] = _fstar[LU3(i, 0, 4)];
                    _f2[LU3(i, 0, 5)] = _fstar[LU3(i, 0, 7)];

                    //  Top wall
                    _f2[LU3(i, Nm1, 8)] = _fstar[LU3(i, Nm1, 6)];
                    _f2[LU3(i, Nm1, 4)] = _fstar[LU3(i, Nm1, 2)];
                    _f2[LU3(i, Nm1, 7)] = _fstar[LU3(i, Nm1, 5)];
                }
                break;
            }

            case 0: {       //  Couette flow
#pragma omp parallel for
                for (unsigned int i = 0; i < N; i++){
                    //  Bottom wall
                    _f2[LU3(i, 0, 6)] = _fstar[LU3(i, 0, 8)] - 2.0 * w[8] * _rhobar * c[8][0] * Ubot * 3.0;
                    _f2[LU3(i, 0, 2)] = _fstar[LU3(i, 0, 4)] - 2.0 * w[4] * _rhobar * c[4][0] * Ubot * 3.0;
                    _f2[LU3(i, 0, 5)] = _fstar[LU3(i, 0, 7)] - 2.0 * w[7] * _rhobar * c[7][0] * Ubot * 3.0;

                    //  Top wall
                    _f2[LU3(i, Nm1, 8)] = _fstar[LU3(i, Nm1, 6)] - 2.0 * w[6] * _rhobar * c[6][0] * Utop * 3.0;
                    _f2[LU3(i, Nm1, 4)] = _fstar[LU3(i, Nm1, 2)] - 2.0 * w[2] * _rhobar * c[2][0] * Utop * 3.0;
                    _f2[LU3(i, Nm1, 7)] = _fstar[LU3(i, Nm1, 5)] - 2.0 * w[5] * _rhobar * c[5][0] * Utop * 3.0;
                }
                break;
            }

            default: {
                std::printf("Error: Invalid Boundary condition case number\n");
                exit(1);
            }

        }
    }

    inline void swap(){
        _f1.swap(_f2);
    }

    inline bool convergence(const unsigned int t){
        if (t == 0)
            _df0 = 1.0 / rmsError();
        if (t % 1000 == 0) {
            _df = rmsError() * _df0;
            if (t / 1000 == _error.size())
                _error.resize(2 * t / 1000);
            _error[t / 1000] = _df;
        }
        if (t % prntInt == 0) {
            cout << "\nIteration " << t << ":" << endl;
            printf("df/df0:\t%.3e\n", _df);
            stop = omp_get_wtime();
            printf("Time:\t%.3e s\n", stop-start);
            printf("rho:\t%.3e\n", _rhobar);
            start = omp_get_wtime();
        }
        return (_df < THRESH);
    }

    inline void macroVars(){
        double temp;
        int ind1;
        _rhobar = 0.0;
#pragma omp parallel for private(temp,ind1) collapse(2) reduction(+:_rhobar)
        for (unsigned int i = 0; i < N; i++){
            for (unsigned int j = 0; j < N; j++){
                ind1 = LU2(i,j);
                temp = _f2[LU3(i,j,0)] + _f2[LU3(i,j,1)] + _f2[LU3(i,j,2)] + _f2[LU3(i,j,3)] + _f2[LU3(i,j,4)] + _f2[LU3(i,j,5)] + _f2[LU3(i,j,6)] + _f2[LU3(i,j,7)] + _f2[LU3(i,j,8)];
                _rho[ind1] = temp;
                _rhobar += (temp / N2);
                _u1[ind1] = ((_f2[LU3(i,j,1)] + _f2[LU3(i,j,5)] + _f2[LU3(i,j,8)]) - (_f2[LU3(i,j,3)] + _f2[LU3(i,j,6)] + _f2[LU3(i,j,7)])) / temp;
                _u2[ind1] = ((_f2[LU3(i,j,2)] + _f2[LU3(i,j,5)] + _f2[LU3(i,j,6)]) - (_f2[LU3(i,j,4)] + _f2[LU3(i,j,7)] + _f2[LU3(i,j,8)])) / temp;
                if (BC == 1) {
                    _u1[ind1] += 0.5 * _forceX / temp;
                    _u2[ind1] += 0.5 * _forceY / temp;
                }
            }
        }
    }

    inline void output(){
        calcvmag();
        calcstress();
        calcVort();
        FILE *f = fopen(_filename1,"w");
        int ind1;
        if (f == nullptr) {
            printf("Error opening file!\n");
            exit(1);
        }
        fprintf(f, "TITLE=\"%s\" VARIABLES=\"x\", \"y\", \"u\", \"v\", \"vmag\", \"omegaxy\", \"vortz\" ZONE T=\"%s\" I=%d J=%d F=POINT\n", _filename1, _filename1,N,N);
        for (unsigned int i = 0; i < N; i++) {
            for (unsigned int j = 0; j < N ; j++) {
                ind1 = LU2(i,j);
                fprintf(f, "%.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f\n", _x[i], _x[j], _u1[ind1] / _Umax,_u2[ind1] / _Umax, _vmag[ind1], _stress[ind1], _vort[ind1]);
            }
        }
        fclose(f);
        FILE *f2 = fopen(_filename2,"w");
        if (f2 == nullptr) {
            printf("Error opening file!\n");
            exit(1);
        }
        for (unsigned int i = 0; i < _error.size(); i++) {
            if (_error[i] == 0.0)
                break;
            else
                fprintf(f2, "%d\t%.10e\n", 1000*i,_error[i]);
        }
        fclose(f2);
    }

private:
    vector<double> _f1;
    vector<double> _f2;
    vector<double> _fstar;
    vector<double> _u1;
    vector<double> _u2;
    vector<double> _rho;
    vector<double> _x;
    vector<double> _error;
    vector<double> _vmag;
    vector<double> _stress;
    vector<double> _vort;
    double _df, _df0, _OMEGA, _OMEGAm, _CS, _MACH, _rhobar;
    char _filename1[80];
    char _filename2[80];
    double _omega_e,_omega_eps,_omega_q,_omega_nu, _GS[9];
    double _Umax = max(max(Utop,Ubot),max(Vlef,Vrig));
    const double _forceX = 8.0 * NU * max(max(Utop, Ubot),max(Vlef, Vrig)) / N2, _forceY = 0.0;

    //  Left-right periodicity
    inline void virtualnode(){
#pragma omp parallel for
        for (unsigned int j = 1; j < Nm1; j++) {
            _fstar[LU3(0, j, 1)] = _fstar[LU3(Nm2, j, 1)];
            _fstar[LU3(0, j, 5)] = _fstar[LU3(Nm2, j, 5)];
            _fstar[LU3(0, j, 8)] = _fstar[LU3(Nm2, j, 8)];

            _fstar[LU3(Nm1, j, 3)] = _fstar[LU3(1, j, 3)];
            _fstar[LU3(Nm1, j, 6)] = _fstar[LU3(1, j, 6)];
            _fstar[LU3(Nm1, j, 7)] = _fstar[LU3(1, j, 7)];
        }

        //  Top Left
        _fstar[LU3(0, Nm1, 1)] = _fstar[LU3(Nm2, Nm1, 1)];
        _fstar[LU3(0, Nm1, 4)] = _fstar[LU3(Nm2, Nm1, 4)];
        _fstar[LU3(0, Nm1, 8)] = _fstar[LU3(Nm2, Nm1, 8)];

        //  Top Right
        _fstar[LU3(Nm1, Nm1, 3)] = _fstar[LU3(1, Nm1, 3)];
        _fstar[LU3(Nm1, Nm1, 7)] = _fstar[LU3(1, Nm1, 7)];
        _fstar[LU3(Nm1, Nm1, 4)] = _fstar[LU3(1, Nm1, 4)];

        //  Bottom Left
        _fstar[LU3(0, 0, 1)] = _fstar[LU3(Nm2, 0, 1)];
        _fstar[LU3(0, 0, 2)] = _fstar[LU3(Nm2, 0, 2)];
        _fstar[LU3(0, 0, 5)] = _fstar[LU3(Nm2, 0, 5)];

        //  Bottom Right
        _fstar[LU3(Nm1, 0, 3)] = _fstar[LU3(1, 0, 3)];
        _fstar[LU3(Nm1, 0, 2)] = _fstar[LU3(1, 0, 2)];
        _fstar[LU3(Nm1, 0, 6)] = _fstar[LU3(1, 0, 6)];
    }

    //  Intitializes x vector
    inline void linspace(vector<double> &x, const double _start, const double _end, const int _num){
        for (unsigned int i = 0; i < _num; i++)
            x[i] = _start + i * (_end - _start) / (_num - 1.0);
    }

    inline double calcfeq(const int k, const int ind, const int check){
        const double u1ij=_u1[ind], u2ij = _u2[ind];
        double cdotu{},feq{},u2{},rho0=_rho[ind];
        u2 = P2(u1ij) + P2(u2ij);
        cdotu = c[k][0] * u1ij + c[k][1] * u2ij;
        feq = w[k] * (_rho[ind] + rho0 * (3.0 * cdotu + (4.5 * P2(cdotu) - 1.5 * u2)));
        if (check == 1)
            checkfeq(feq);
        return feq;
    }

    inline double calcfeq(const int k, const double u1ij, const double u2ij, const double rhoij, const int check){
        double cdotu{},feq{},u2{}, rho0=rhoij;
        u2 = P2(u1ij) + P2(u2ij);
        cdotu = c[k][0] * u1ij + c[k][1] * u2ij;
        feq = w[k] * (rhoij + rho0 * (3.0 * cdotu + (4.5 * P2(cdotu) - 1.5 * u2)));
        if (check == 1)
            checkfeq(feq);
        return feq;
    }

    //  Checks for negative feq
    inline void checkfeq(const double value){
        if (value < 0) {
            printf("Error: negative feq.  Therefore, unstable.\n");
            exit(1);
        }
    }

    inline double rmsError(){
        double difference{};
#pragma omp parallel for reduction(+:difference)
        for (unsigned int i = 0; i < NQ; i++){
            difference += P2(_f2[i] - _f1[i]);
        }
        return sqrt(difference / NQ);
    }

    inline void calcmeq(vector<double> &meq, const double u1, const double u2, const double rho) {
        const double u12 = P2(u1);
        const double u22 = P2(u2);
        const double rho0 = rho;
        meq[0] = rho;                               //  rho
        meq[1] = -2 * rho + 3 * rho0 * (u12 + u22);  //  e
        meq[2] = rho - 3 * rho0 * (u12 + u22);       //  eps
        meq[3] = rho0 * u1;                          //  jx
        meq[4] = -meq[3];                           //  qx
        meq[5] = rho0 * u2;                          //  jy
        meq[6] = -meq[5];                           //  qy
        meq[7] = rho0 * (u12 - u22);                 //  pxx
        meq[8] = rho0 * u1 * u2;                     //  pxy
    }

    inline int LU2(const int i, const int j) {
        return N*j + i;
    }

    inline int LU3(const int i, const int j, const int k) {
        return N2*k + N*j + i;
    }

    inline double P2(const double value){
        return (value * value);
    }

    inline void calcvmag(){
#pragma omp parallel for
        for (unsigned int i = 0; i < N2; i++)
            _vmag[i] = sqrt(P2(_u1[i]) + P2(_u2[i]));
    }

    inline void calcstress(){
#pragma omp parallel
        {
        int ind1;
#pragma omp for collapse(2)
        for (unsigned int i = 0; i < N; i++){
            for (unsigned int j = 0; j < N; j++){
                ind1 = LU2(i,j);
                for (unsigned int k = 0; k < Q; k++){
                    _stress[ind1] -= (1.0 - 0.5 * _OMEGA) * c[k][0] * c[k][1] * (_f2[LU3(i,j,k)] - calcfeq(k,ind1,1));
                }
                if (BC == 1)
                    _stress[ind1] -= 0.5 * (1.0 - 0.5 * _OMEGA) * (_forceX * _u2[ind1] + _forceY * _u1[ind1]);
            }
        }
        }
    }

    inline void calcVort(){
        double dvdx{},dudy{};
        const double h2 = P2(_x[1] - _x[0]);
        int ind1;

#pragma omp parallel for private(ind1,dvdx,dudy), collapse(2)
        //  Internal region
        for (unsigned int i = 1; i < Nm1; i++) {
            for (unsigned int j = 1; j < Nm1; j++) {
                ind1 = LU2(i,j);
                dvdx = (_u2[LU2(i-1,j)] - 2.0 * _u2[ind1] + _u2[LU2(i+1,j)]) / (h2 * _Umax);
                dudy = (_u1[LU2(i,j-1)] - 2.0 * _u1[ind1] + _u1[LU2(i,j+1)]) / (h2 * _Umax);
                _vort[ind1] = dvdx - dudy;
            }
        }

#pragma omp parallel for private(dvdx,dudy)
        //  boundary
        for (unsigned int i = 1; i < N-1; i++) {

            //  Top
            dvdx = (_u2[LU2(i-1,Nm1)] - 2.0 * _u2[LU2(i-1,Nm1)] + _u2[LU2(i+1,Nm1)]) / (h2 * _Umax);
            dudy = (_u1[LU2(i,Nm1)] - 2.0 * _u1[LU2(i,Nm2)] + _u1[LU2(i,N - 3)]) / (h2 * _Umax);
            _vort[LU2(i,Nm1)] = dvdx - dudy;

            //  Bottom
            dvdx = (_u2[LU2(i-1,0)] - 2.0 * _u2[LU2(i-1,0)] + _u2[LU2(i+1,0)]) / (h2 * _Umax);
            dudy = (_u1[LU2(i,2)] - 2.0 * _u1[LU2(i,1)] + _u1[LU2(i,0)]) / (h2 * _Umax);
            _vort[LU2(i,0)] = dvdx - dudy;

            //  Left
            dvdx = (_u2[LU2(2,i)] - 2.0 * _u2[LU2(1,i)] + _u2[LU2(0,i)]) / (h2 * _Umax);
            dudy = (_u1[LU2(0,i+1)] - 2.0 * _u1[LU2(0,i)] + _u1[LU2(0,i-1)]) / (h2 * _Umax);
            _vort[LU2(0,i)] = dvdx - dudy;

            //  Right
            dvdx = (_u2[LU2(Nm1,i)] - 2.0 * _u2[LU2(Nm2,i)] + _u2[LU2(N - 3,i)]) / (h2 * _Umax);
            dudy = (_u1[LU2(Nm1,i+1)] - 2.0 * _u1[LU2(Nm1,i)] + _u1[LU2(Nm1,i-1)]) / (h2 * _Umax);
            _vort[LU2(0,i)] = dvdx - dudy;
        }

        //  Corners
        //  Bottom Left
        dvdx = (_u2[LU2(2,0)] - 2.0 * _u2[LU2(1,0)] + _u2[LU2(0,0)]) / (h2 * _Umax);
        dudy = (_u1[LU2(0,2)] - 2.0 * _u1[LU2(0,1)] + _u1[LU2(0,0)]) / (h2 * _Umax);
        _vort[LU2(0,0)] = dvdx - dudy;

        //  Bottom Right
        dvdx = (_u2[LU2(Nm1,0)] - 2.0 * _u2[LU2(Nm2,0)] + _u2[LU2(N - 3,0)]) / (h2 * _Umax);
        dudy = (_u1[LU2(Nm1,2)] - 2.0 * _u1[LU2(Nm1,1)] + _u1[LU2(Nm1,0)]) / (h2 * _Umax);
        _vort[LU2(Nm1,0)] = dvdx - dudy;

        //  Top Left
        dvdx = (_u2[LU2(2,Nm1)] - 2.0 * _u2[LU2(1,Nm1)] + _u2[LU2(0,Nm1)]) / (h2 * _Umax);
        dudy = (_u1[LU2(0,Nm1)] - 2.0 * _u1[LU2(0,Nm2)] + _u1[LU2(0,N - 3)]) / (h2 * _Umax);
        _vort[LU2(0,Nm1)] = dvdx - dudy;

        //  Top Right
        dvdx = (_u2[LU2(Nm1,Nm1)] - 2.0 * _u2[LU2(Nm2,Nm1)] + _u2[LU2(N - 3,Nm1)]) / (h2 * _Umax);
        dudy = (_u1[LU2(Nm1,Nm1)] - 2.0 * _u1[LU2(Nm1,Nm2)] + _u1[LU2(Nm1,N - 3)]) / (h2 * _Umax);
        _vort[LU2(Nm1,Nm1)] = dvdx - dudy;
    }


};
#endif //LIDDRIVENCAVITYLBM_LBMCLASS_H
