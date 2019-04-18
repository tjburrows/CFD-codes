#ifndef LIDDRIVENCAVITYLBM_LBMCLASS_H
#define LIDDRIVENCAVITYLBM_LBMCLASS_H
#include <iostream>
#include <vector>
#include <math.h>
#include <omp.h>
#include "Eigen/Dense"

using namespace std;

extern const double    Re;
extern const int       N;
extern const int       M;
extern const double    Utop,Vtop;
extern const double    Ubot,Vbot;
extern const double    Ulef,Vlef;
extern const double    Urig,Vrig;
extern const int       SAVETXT;
extern const double    THRESH;
extern const int       THREADS;
extern const int       MBounds;
extern const int       MCollide;
extern const int       PRECOND;
extern const double     GAMMA;
extern const int       INCOMP;
extern const int       BC;
extern const int               Q;
extern const double            MAXITER;
extern const double            NU;
extern const double            TAU;
extern const double            MAGIC;
extern const double            RHO0,U0,V0;
extern const int      			N2;
extern const int      NQ;
extern const double            w[9];
extern const int               c[9][2];
extern const int               opp[9];
extern const int               half[4];
extern const int               prntInt;
extern const int               Nm1,Mm1;
extern const int               Nm2,Mm2;
extern const int GM[9][9];
extern const double GMinv[9][9];
extern double start,stop;
extern const int IBN,IB;
extern const double IBcenter[2],IBradius;

class LBMClass{
public:
    LBMClass(): _f1(NQ,0.0), _f2(NQ,0.0), _fstar(NQ, 0.0),_u1(N2, U0) ,_u2(N2, V0), _rho(N2, RHO0),_p(N2, 0.0), _x(N, 0.0),_y(M, 0.0), _error(100, 0.0),_vmag(N2,0.0),
				_stress(N2, 0.0), _vort(N2, 0.0),_dudx(N2, 0.0),_dudy(N2, 0.0),_dvdx(N2, 0.0),_dvdy(N2, 0.0),_forceX(N2, 0.0),_forceY(N2, 0.0),
				_df(), _df0(),_MACHSTAR(0.0), _TAU_P(0.0), _OMEGA(0.0), _OMEGAm(0.0), _CS(0.0), _MACH(0.0), _rhobar(1.0),
				_filename1(),_filename2(),_omega_e(),_omega_eps(),_omega_q(),_omega_nu(), _GS{}, _Umax(),
				_IBrx(IBN,0.0), _IBry(IBN,0.0), _IBur(IBN,0.0), _IBvr(IBN,0.0), _IBFx(IBN,0.0), _IBFy(IBN,0.0), _IBub(IBN,0.0), _IBvb(IBN,0.0),
				_IBmatrixAx(IBN,IBN), _IBmatrixAy(IBN,IBN), _IBmatrixAxInv(IBN,IBN), _IBmatrixAyInv(IBN,IBN), _fBodyX(),_fBodyY(),_IBds(),_IBrxmx(),_IBrxmn(),_IBrymx(),_IBrymn()
    {
        //  Initialize x
        if (MBounds == 0){
            linspace(_x,(0.5 / N), (N - 0.5) / (double) N,N);
            linspace(_y,(0.5 / M), (M - 0.5) / (double) N,M);
        }

        else{
            linspace(_x,0.0,1.0 ,N);
            linspace(_y,0.0,(double) M / (double) N ,M);
        }
        _CS = 1.0 / sqrt(3.0);
        _MACH = Utop / _CS;
		_MACHSTAR = _MACH / sqrt(GAMMA);

        _TAU_P   = 0.5 + (TAU - 0.5) / GAMMA;
        _OMEGA   = 1.0 / _TAU_P;
        _OMEGAm  = 1.0 / (0.5 + ((MAGIC) / ((1.0 / (_OMEGA)) - 0.5)));
		_Umax = max(max(Utop,Ubot),max(Vlef,Vrig));
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

        //_GS = {0.0, _omega_e, _omega_eps, 0.0, _omega_q, 0.0, _omega_q, _omega_nu, _omega_nu};
        //  SRT
        //const double _GS[Q] = {0.0, _OMEGA, _OMEGA, 0.0, _OMEGA, 0.0, _OMEGA, _OMEGA, _OMEGA};

        //   Mohamad Textbook
        //const double _GS[Q] = {0.0, 1.4, 1.4, 0.0, 1.2, 0.0, 1.2, _OMEGA, _OMEGA};

        //  High Re from Zhen-Hua et al.
        //const double _GS[Q] = {0.0, 1.1, 1.0, 0.0, 1.2, 0.0, 1.2, _OMEGA, _OMEGA};

//        _GS[0] = 0.0;
//        _GS[1] = 1.1;
//        _GS[2] = 1.0;
//        _GS[3] = 0.0;
//        _GS[4] = 1.2;
//        _GS[5] = 0.0;
//        _GS[6] = 1.2;
//        _GS[7] = _omega_nu;
//        _GS[8] = _omega_nu;


        sprintf(_filename1,"Solution_n=%d_m=%d_Re=%.0f_BCM=%d_CM=%d_G=%0.2f_U=%0.2f.dat",N,M,Re,MBounds,MCollide, GAMMA, Utop);
        sprintf(_filename2,"Error_n=%d_m=%d_Re=%.0f_BCM=%d_CM=%d_G=%0.2f_U=%0.2f.txt",N,M,Re,MBounds,MCollide, GAMMA,Utop);

        //  Initialize f
        int ind1,ind2;
#pragma omp parallel for private(ind1,ind2) collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                ind1 = LU2(i, j);
                for (int k = 0; k < Q; k++) {
                    ind2 = LU3(i, j, k);
                    _f1[ind2] = calcfeq(k,ind1);
                    _f2[ind2] = _f1[ind2];
                }
            }
        }
		if (BC == 1){
			_fBodyX = 8.0 * NU * _Umax / (M*M);
		}
		
		if (IB==1) {
			//*****  Immersed Boundary Initialization  *****
			//  initialize position of points on boundary
			for (int i = 0; i < IBN; i++){
				_IBrx[i] = IBcenter[0] + IBradius * cos((i * 360.0 / (double) IBN) * (_pi/180.0));
				_IBry[i] = IBcenter[1] + IBradius * sin((i * 360.0 / (double) IBN) * (_pi/180.0));
			}
			
			//	Find IB region
			for (int i = 0; i < IBN; i++){
				if (i == 0){
					_IBrxmx = _IBrx[0];
					_IBrxmn = _IBrx[0];
					_IBrymx = _IBry[0];
					_IBrymn = _IBry[0];
				}
				else {
					if (_IBrx[i] < _IBrxmn)
						_IBrxmn = _IBrx[i];
					if (_IBrx[i] > _IBrxmx)
						_IBrxmx = _IBrx[i];
					if (_IBry[i] < _IBrymn)
						_IBrymn = _IBry[i];
					if (_IBry[i] > _IBrymx)
						_IBrymx = _IBry[i];
				}
			}
			
			//	Find loop bounds
			_IBrxmx = min(ceil(_IBrxmx + 2), (double) Nm1);
			_IBrymx = min(ceil(_IBrymx + 2), (double) Mm1);
			_IBrxmn = max(floor(_IBrxmn - 2), 0.0);
			_IBrymn = max(floor(_IBrymn - 2), 0.0);
			//_IBrxmx = Nm1;
			//_IBrymx = Mm1;
			//_IBrxmn = 0.0;
			//_IBrymn = 0.0;

			_IBds = 2.0 * _pi * IBradius / (double) IBN;
			printf("Points of Immersed Boundary:\n");
			for (int i = 0; i < IBN; i++)
				printf("(%.3f, %.3f)\n", _IBrx[i],_IBry[i]);
			printf("X IB (%d,%d)\n",(int)_IBrxmn,(int)_IBrxmx);
			printf("Y IB (%d,%d)\n",(int)_IBrymn,(int)_IBrymx);
			//	Velocity of boundary
			for (int i = 0; i < IBN; i++){
				_IBub[i] = 0.0;
				_IBub[i] = 0.0;
			}
			
			//  Calculate matrix A
			_IBmatrixAx = Eigen::MatrixXd::Zero(IBN,IBN);
			_IBmatrixAy = Eigen::MatrixXd::Zero(IBN,IBN);
			double Dirac1,Dirac2;
#pragma omp parallel for collapse(2) private(Dirac1,Dirac2)
			for (int i = 0; i < IBN; i++){
				for (int j = 0; j < IBN; j++){
					for (int k = (int)_IBrxmn; k <= (int)_IBrxmx; k++){
						for (int l = (int)_IBrymn; l <= (int)_IBrymx; l++){
							Dirac1 =  diracdelta(_IBrx[i] - k) * diracdelta(_IBry[i] - l);
							Dirac2 =  diracdelta(_IBrx[j] - k) * diracdelta(_IBry[j] - l);
							_IBmatrixAx(i,j) +=  _IBds * Dirac1 * Dirac2;
							_IBmatrixAy(i,j) +=  _IBds * Dirac1 * Dirac2;
						}
					}
				}
			}

			//	Calculate inverse of matrix A
			_IBmatrixAxInv = _IBmatrixAx.inverse();
			_IBmatrixAyInv = _IBmatrixAy.inverse();
		}
		
        //  Print parameters
        printf("Re =\t%.0f\n", Re);
        printf("U =\t%.3e\n", Utop);
        printf("M =\t%.3e\n", Utop * sqrt(3));
        printf("N =\t%d\n", N);
        printf("M =\t%d\n", M);
        printf("tau =\t%.3e\n", _TAU_P);
        printf("nu =\t%.3e\n", NU);
        printf("Gamma =\t%.3e\n", GAMMA);
        if (PRECOND == 1){
            printf("_MACH* =\t%.3e\n", _MACHSTAR);
        }
    }

    //  Collide Methods
    inline void collideSRT() {
        int ind1{}, ind2{};
        double Fsource{},fTotalx{},fTotaly{};
#pragma omp parallel for private(ind1, ind2,Fsource,fTotalx,fTotaly) collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                ind1 = LU2(i, j);
				fTotalx = _forceX[ind1] + _fBodyX;
				fTotaly = _forceY[ind1] + _fBodyY;
                for (int k = 0; k < Q; k++) {
                    ind2 = LU3(i, j, k);
                    _fstar[ind2] = (1.0 - _OMEGA) * _f1[ind2] + _OMEGA * calcfeq(k, ind1);
					Fsource = (1.0 - 0.5 * _OMEGA) * w[k] *
							  (3.0 * (c[k][0] - _u1[ind1]) + 9.0 * (c[k][0] * _u1[ind1] + c[k][1] * _u2[ind1]) * c[k][0]) * fTotalx +
							  (3.0 * (c[k][1] - _u2[ind1]) + 9.0 * (c[k][0] * _u1[ind1] + c[k][1] * _u2[ind1]) * c[k][1]) * fTotaly;
					_fstar[ind2] += Fsource;
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
        for (int i = 0; i < N; i++){
            for (int j = 0; j < M; j++){
                ind1 = LU2(i, j);
                for (int k = 0; k < Q; k++)
                    feq[k] = calcfeq(k,ind1);

                //  Rest population
                fplus = _f1[LU3(i,j,0)];
                feqplus = feq[0];
                _fstar[LU3(i, j, 0)] = _f1[LU3(i, j, 0)] - _OMEGA * (fplus - feqplus);

                for (int k = 0; k < 4; k++) {
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
            double _m{},fTotalx,fTotaly; // moments
#pragma omp for collapse(2)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
                    ind1 = LU2(i,j);
					fTotalx = _forceX[ind1] + _fBodyX;
					fTotaly = _forceY[ind1] + _fBodyY;
                    calcmeq(_meq, _u1[ind1], _u2[ind1], _rho[ind1],_p[ind1]);
                    for (int k = 0; k < Q; k++){
                        _m = GM[k][0] * _f1[LU3(i,j,0)] + GM[k][1] * _f1[LU3(i,j,1)] + GM[k][2] * _f1[LU3(i,j,2)] +GM[k][3] * _f1[LU3(i,j,3)] + GM[k][4] * _f1[LU3(i,j,4)]
                             + GM[k][5] * _f1[LU3(i,j,5)] + GM[k][6] * _f1[LU3(i,j,6)] + GM[k][7] * _f1[LU3(i,j,7)] + GM[k][8] * _f1[LU3(i,j,8)];
                        _mstar[k] = _m - _GS[k] * (_m - _meq[k]);
                    }
					
					//	Forces
					_mstar[1] += (1.0 - 0.5 * _GS[1]) * (6.0 * (fTotalx * _u1[ind1] + fTotaly * _u2[ind1]));
					_mstar[2] += (1.0 - 0.5 * _GS[2]) * (-6.0 * (fTotalx * _u1[ind1] + fTotaly * _u2[ind1]));
					_mstar[3] += (1.0 - 0.5 * _GS[3]) * (fTotalx);
					_mstar[4] += (1.0 - 0.5 * _GS[4]) * (-fTotalx);
					_mstar[5] += (1.0 - 0.5 * _GS[5]) * (fTotaly);
					_mstar[6] += (1.0 - 0.5 * _GS[6]) * (-fTotaly);
					_mstar[7] += (1.0 - 0.5 * _GS[7]) * (2.0 * (fTotalx * _u1[ind1] - fTotaly * _u2[ind1]));
					_mstar[8] += (1.0 - 0.5 * _GS[8]) * (fTotaly * _u1[ind1] + fTotalx * _u2[ind1]);
					
					
                    for (int k = 0; k < Q; k++){
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
#pragma omp for collapse(3)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
                    for (int k = 0; k < Q; k++) {
                        inew = i + c[k][0];
                        jnew = j + c[k][1];
                        if (inew < N && inew >= 0 && jnew < M && jnew >= 0)
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
#pragma omp for collapse(3)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
                    for (int k = 0; k < Q; k++) {
                        iold = i - c[k][0];
                        jold = j - c[k][1];
                        if (iold < N && iold >= 0 && jold < M && jold >= 0)
                            _f2[LU3(i, j, k)] = _fstar[LU3(iold, jold, k)];
                    }
                }
            }
        }
    }

    //  Bounday Condition Methods
	
	//	Wind tunnel simulation
	inline void uniformFlow(){
		double rho{1.0}, sixth=1.0/6.0,twothirds=2.0/3.0,twelfth=1.0/12.0,Ucorner{};
		
		//	Outflow, 2nd order polynomial extrapolation
		double uout{};
		for (int i = 1; i < Mm1 ;i++) {
			uout = -1.0 + _f2[LU3(Nm1, i, 0)] + _f2[LU3(Nm1, i, 2)] + _f2[LU3(Nm1, i, 4)] + 2.0 * (_f2[LU3(Nm1, i, 1)] + _f2[LU3(Nm1, i, 5)] + _f2[LU3(Nm1, i, 8)]);
			_f2[LU3(Nm1, i, 3)] = _f2[LU3(Nm1, i, 1)] - twothirds * uout;
			_f2[LU3(Nm1, i, 7)] = _f2[LU3(Nm1, i, 5)] + 0.5 * (_f2[LU3(Nm1, i, 2)]-_f2[LU3(Nm1, i, 4)]) - sixth * uout;
			_f2[LU3(Nm1, i, 6)] = _f2[LU3(Nm1, i, 8)] - 0.5 * (_f2[LU3(Nm1, i, 2)]-_f2[LU3(Nm1, i, 4)]) - sixth * uout;
		}
		
		//	Inflow, Dirichlet BC
		for (int i = 0; i < M ;i++) {
//			rho = _f2[LU3(0, i, 0)] + _f2[LU3(0, i, 2)] + _f2[LU3(0, i, 4)] + 2.0 * (_f2[LU3(0, i, 3)] + _f2[LU3(0, i, 6)] + _f2[LU3(0, i, 7)]);
			_f2[LU3(0, i, 1)] = _f2[LU3(0, i, 3)] + twothirds * rho * Ulef;
			_f2[LU3(0, i, 5)] = _f2[LU3(0, i, 7)] + sixth * rho * Ulef;
			_f2[LU3(0, i, 8)] = _f2[LU3(0, i, 6)] + sixth * rho * Ulef;
		}
		
		//	half-way Specular reflection on top and bottom
		for (int i = 0; i < N; i++) {
			//	Top
			_f2[LU3(i, Mm1, 4)] = _f2[LU3(i, Mm1, 2)];
			_f2[LU3(i, Mm1, 7)] = _f2[LU3(i, Mm1, 6)];
			_f2[LU3(i, Mm1, 8)] = _f2[LU3(i, Mm1, 5)];
			
			//	Bottom
			_f2[LU3(i, 0, 2)] = _f2[LU3(i, 0, 4)];
			_f2[LU3(i, 0, 5)] = _f2[LU3(i, 0, 8)];
			_f2[LU3(i, 0, 6)] = _f2[LU3(i, 0, 7)];
		}
		
		
		
		
		//  Top Left (0,Nm1) knowns: 1,4,8, unknowns: 0, 5, 7
//		rho = _rho[LU2(0,Mm1)];
//		Ucorner = Ulef;
//		_f2[LU3(0, Mm1, 1)] = _f2[LU3(0, Mm1, 3)] + twothirds * rho * Ucorner;
//		_f2[LU3(0, Mm1, 4)] = _f2[LU3(0, Mm1, 2)];
//		_f2[LU3(0, Mm1, 8)] = _f2[LU3(0, Mm1, 6)] + sixth * rho * (Ucorner);
//		_f2[LU3(0, Mm1, 5)] = twelfth * rho * (Ucorner);
//		_f2[LU3(0, Mm1, 7)] = -_f2[LU3(0, Mm1, 5)];
//		_f2[LU3(0, Mm1, 0)] = rho - (_f2[LU3(0, Mm1, 1)] + _f2[LU3(0, Mm1, 2)] + _f2[LU3(0, Mm1, 3)] + _f2[LU3(0, Mm1, 4)] + _f2[LU3(0, Mm1, 5)] + _f2[LU3(0, Mm1, 6)] + _f2[LU3(0, Mm1, 7)] + _f2[LU3(0, Mm1, 8)]);

//		rho = _rho[LU2(0,0)];
//		Ucorner = Ulef;
//		_f2[LU3(0, 0, 1)] = _f2[LU3(0, 0, 3)] + twothirds * rho * Ucorner;
//		_f2[LU3(0, 0, 2)] = _f2[LU3(0, 0, 4)];
//		_f2[LU3(0, 0, 5)] = _f2[LU3(0, 0, 7)] + sixth * rho * (Ucorner);
//		_f2[LU3(0, 0, 6)] = twelfth * rho * (Ucorner);
//		_f2[LU3(0, 0, 8)] = -_f2[LU3(0, 0, 6)];
//		_f2[LU3(0, 0, 0)] = rho - (_f2[LU3(0, 0, 1)] + _f2[LU3(0, 0, 2)] + _f2[LU3(0, 0, 3)] + _f2[LU3(0, 0, 4)] + _f2[LU3(0, 0, 5)] + _f2[LU3(0, 0, 6)] + _f2[LU3(0, 0, 7)] + _f2[LU3(0, 0, 8)]);
	}

    inline void NEBB(){
        switch (BC){
            case 2 : {  //  Lid-driven cavity flow
                const double sixth = 1.0 / 6.0, twothirds = 2.0 / 3.0, twelfth = 1.0 / 12.0;
                double rho{_rhobar},Ucorner{},Vcorner{};
#pragma omp parallel for private(rho)
                for (int i = 1; i < N - 1; i++) {
                    //  Top wall, general case
                    rho = _f2[LU3(i, Mm1, 0)] + _f2[LU3(i, Mm1, 1)] + _f2[LU3(i, Mm1, 3)] + 2.0 * (_f2[LU3(i, Mm1, 2)] + _f2[LU3(i, Mm1, 6)] + _f2[LU3(i, Mm1, 5)]);
                    _f2[LU3(i, Mm1, 4)] = _f2[LU3(i, Mm1, 2)] - twothirds * rho * Vtop;
                    _f2[LU3(i, Mm1, 7)] = _f2[LU3(i, Mm1, 5)] + 0.5 * (_f2[LU3(i, Mm1, 1)] - _f2[LU3(i, Mm1, 3)]) - 0.5 * rho * Utop - sixth * rho * Vtop;
                    _f2[LU3(i, Mm1, 8)] = _f2[LU3(i, Mm1, 6)] - 0.5 * (_f2[LU3(i, Mm1, 1)] - _f2[LU3(i, Mm1, 3)]) + 0.5 * rho * Utop - sixth * rho * Vtop;

                    //  Bottom wall, general case
                    rho = _f2[LU3(i, 0, 0)] + _f2[LU3(i, 0, 1)] + _f2[LU3(i, 0, 3)] + 2.0 * (_f2[LU3(i, 0, 4)] + _f2[LU3(i, 0, 7)] + _f2[LU3(i, 0, 8)]);
                    _f2[LU3(i, 0, 2)] = _f2[LU3(i, 0, 4)] + twothirds * rho * Vbot;
                    _f2[LU3(i, 0, 5)] = _f2[LU3(i, 0, 7)] - 0.5 * (_f2[LU3(i, 0, 1)] - _f2[LU3(i, 0, 3)]) + 0.5 * rho * Ubot + sixth * rho * Vbot;
                    _f2[LU3(i, 0, 6)] = _f2[LU3(i, 0, 8)] + 0.5 * (_f2[LU3(i, 0, 1)] - _f2[LU3(i, 0, 3)]) - 0.5 * rho * Ubot + sixth * rho * Vbot;
                }
#pragma omp parallel for private(rho)
                for (int i = 1; i < Mm1; i++) {
                    //  Left wall, general case
                    rho = _f2[LU3(0, i, 0)] + _f2[LU3(0, i, 2)] + _f2[LU3(0, i, 4)] + 2.0 * (_f2[LU3(0, i, 3)] + _f2[LU3(0, i, 6)] + _f2[LU3(0, i, 7)]);
                    _f2[LU3(0, i, 1)] = _f2[LU3(0, i, 3)] + twothirds * rho * Ulef;
                    _f2[LU3(0, i, 5)] = _f2[LU3(0, i, 7)] - 0.5 * (_f2[LU3(0, i, 2)] - _f2[LU3(0, i, 4)]) + 0.5 * rho * Vlef + sixth * rho * Ulef;
                    _f2[LU3(0, i, 8)] = _f2[LU3(0, i, 6)] + 0.5 * (_f2[LU3(0, i, 2)] - _f2[LU3(0, i, 4)]) - 0.5 * rho * Vlef + sixth * rho * Ulef;

                    //  Right wall, general case
                    rho = _f2[LU3(Nm1, i, 0)] + _f2[LU3(Nm1, i, 2)] + _f2[LU3(Nm1, i, 4)] + 2.0 * (_f2[LU3(Nm1, i, 1)] + _f2[LU3(Nm1, i, 5)] + _f2[LU3(Nm1, i, 8)]);
                    _f2[LU3(Nm1, i, 3)] = _f2[LU3(Nm1, i, 1)] - twothirds * rho * Urig;
                    _f2[LU3(Nm1, i, 7)] = _f2[LU3(Nm1, i, 5)] + 0.5 * (_f2[LU3(Nm1, i, 2)] - _f2[LU3(Nm1, i, 4)]) - 0.5 * rho * Vrig - sixth * rho * Urig;
                    _f2[LU3(Nm1, i, 6)] = _f2[LU3(Nm1, i, 8)] - 0.5 * (_f2[LU3(Nm1, i, 2)] - _f2[LU3(Nm1, i, 4)]) + 0.5 * rho * Vrig - sixth * rho * Urig;
                }

                // Corners
                rho = _rhobar;
                //  Bottom Left (0,0) knowns: 1,5,2, unknowns: 0,6,8
				rho = _rho[LU2(0,0)];
				Vcorner = max(Vbot, Vlef);
				Ucorner = max(Ubot, Ulef);
                _f2[LU3(0, 0, 1)] = _f2[LU3(0, 0, 3)] + twothirds * rho * Ucorner;
                _f2[LU3(0, 0, 2)] = _f2[LU3(0, 0, 4)] + twothirds * rho * Vcorner;
                _f2[LU3(0, 0, 5)] = _f2[LU3(0, 0, 7)] + sixth * rho * (Ucorner + Vcorner);
                _f2[LU3(0, 0, 6)] = twelfth * rho * (Vcorner - Ucorner);
                _f2[LU3(0, 0, 8)] = -_f2[LU3(0, 0, 6)];
                _f2[LU3(0, 0, 0)] = rho - (_f2[LU3(0, 0, 1)] + _f2[LU3(0, 0, 2)] + _f2[LU3(0, 0, 3)] + _f2[LU3(0, 0, 4)] + _f2[LU3(0, 0, 5)] + _f2[LU3(0, 0, 6)] + _f2[LU3(0, 0, 7)] + _f2[LU3(0, 0, 8)]);

                //  Bottom Right (Nm1,0) knowns: 2,3,6, unknowns: 0, 5, 7
				rho = _rho[LU2(Nm1,0)];
				Vcorner = max(Vbot, Vrig);
				Ucorner = max(Ubot, Urig);
                _f2[LU3(Nm1, 0, 2)] = _f2[LU3(Nm1, 0, 4)] + twothirds * rho * Vcorner;
                _f2[LU3(Nm1, 0, 3)] = _f2[LU3(Nm1, 0, 1)] - twothirds * rho * Ucorner;
                _f2[LU3(Nm1, 0, 6)] = _f2[LU3(Nm1, 0, 8)] + sixth * rho * (-Ucorner + Vcorner);
                _f2[LU3(Nm1, 0, 5)] = twelfth * rho * (Vcorner + Ucorner);
                _f2[LU3(Nm1, 0, 7)] = -_f2[LU3(Nm1, 0, 5)];
                _f2[LU3(Nm1, 0, 0)] = rho - (_f2[LU3(Nm1, 0, 1)] + _f2[LU3(Nm1, 0, 2)] + _f2[LU3(Nm1, 0, 3)] + _f2[LU3(Nm1, 0, 4)] + _f2[LU3(Nm1, 0, 5)] + _f2[LU3(Nm1, 0, 6)] + _f2[LU3(Nm1, 0, 7)] + _f2[LU3(Nm1, 0, 8)]);

                //  Top Left (0,Nm1) knowns: 1,4,8, unknowns: 0, 5, 7
				rho = _rho[LU2(0,Mm1)];
				Vcorner = max(Vtop, Vlef);
				Ucorner = max(Utop, Ulef);
                _f2[LU3(0, Mm1, 1)] = _f2[LU3(0, Mm1, 3)] + twothirds * rho * Ucorner;
                _f2[LU3(0, Mm1, 4)] = _f2[LU3(0, Mm1, 2)] - twothirds * rho * Vcorner;
                _f2[LU3(0, Mm1, 8)] = _f2[LU3(0, Mm1, 6)] + sixth * rho * (Ucorner - Vcorner);
                _f2[LU3(0, Mm1, 5)] = twelfth * rho * (Vcorner + Ucorner);
                _f2[LU3(0, Mm1, 7)] = -_f2[LU3(0, Mm1, 5)];
                _f2[LU3(0, Mm1, 0)] = rho - (_f2[LU3(0, Mm1, 1)] + _f2[LU3(0, Mm1, 2)] + _f2[LU3(0, Mm1, 3)] + _f2[LU3(0, Mm1, 4)] + _f2[LU3(0, Mm1, 5)] + _f2[LU3(0, Mm1, 6)] + _f2[LU3(0, Mm1, 7)] + _f2[LU3(0, Mm1, 8)]);

                //  Top Right (Nm1,Nm1) knowns: 3,7,4, unknowns: 0, 6, 8
				rho = _rho[LU2(Nm1,Mm1)];
				Vcorner = max(Vtop, Vrig);
				Ucorner = max(Utop, Urig);
                _f2[LU3(Nm1, Mm1, 4)] = _f2[LU3(Nm1, Mm1, 2)] - twothirds * rho * Vcorner;
                _f2[LU3(Nm1, Mm1, 3)] = _f2[LU3(Nm1, Mm1, 1)] - twothirds * rho * Ucorner;
                _f2[LU3(Nm1, Mm1, 7)] = _f2[LU3(Nm1, Mm1, 5)] - sixth * rho * (Ucorner + Vcorner);
                _f2[LU3(Nm1, Mm1, 6)] = twelfth * rho * (Vcorner - Ucorner);
                _f2[LU3(Nm1, Mm1, 8)] = -_f2[LU3(Nm1, Mm1, 6)];
                _f2[LU3(Nm1, Mm1, 0)] = rho - (_f2[LU3(Nm1, Mm1, 1)] + _f2[LU3(Nm1, Mm1, 2)] + _f2[LU3(Nm1, Mm1, 3)] + _f2[LU3(Nm1, Mm1, 4)] + _f2[LU3(Nm1, Mm1, 5)] + _f2[LU3(Nm1, Mm1, 6)] + _f2[LU3(Nm1, Mm1, 7)] + _f2[LU3(Nm1, Mm1, 8)]);
                break;
            }

            case 1: {   //  Poiseuille Flow
#pragma omp parallel for
                for (int i = 0; i < N; i++) {
                    //  Top wall, general case
                    _f2[LU3(i, Mm1, 4)] = _f2[LU3(i, Mm1, 2)];
                    _f2[LU3(i, Mm1, 7)] = _f2[LU3(i, Mm1, 5)] + 0.5 * (_f2[LU3(i, Mm1, 1)] - _f2[LU3(i, Mm1, 3)]);
                    _f2[LU3(i, Mm1, 8)] = _f2[LU3(i, Mm1, 6)] - 0.5 * (_f2[LU3(i, Mm1, 1)] - _f2[LU3(i, Mm1, 3)]);

                    //  Bottom wall, general case
                    _f2[LU3(i, 0, 2)] = _f2[LU3(i, 0, 4)];
                    _f2[LU3(i, 0, 5)] = _f2[LU3(i, 0, 7)] - 0.5 * (_f2[LU3(i, 0, 1)] - _f2[LU3(i, 0, 3)]);
                    _f2[LU3(i, 0, 6)] = _f2[LU3(i, 0, 8)] + 0.5 * (_f2[LU3(i, 0, 1)] - _f2[LU3(i, 0, 3)]);
                }
                break;
            }

            case 0: {   //  Couette Flow
                double rho{};
#pragma omp parallel for private(rho)
                for (int i = 0; i < N; i++) {
                    //  Top wall, general case
                    rho = _f2[LU3(i, Mm1, 0)] + _f2[LU3(i, Mm1, 1)] + _f2[LU3(i, Mm1, 3)] + 2.0 * (_f2[LU3(i, Mm1, 2)] + _f2[LU3(i, Mm1, 6)] + _f2[LU3(i, Mm1, 5)]);
                    _f2[LU3(i, Mm1, 4)] = _f2[LU3(i, Mm1, 2)];
                    _f2[LU3(i, Mm1, 7)] = _f2[LU3(i, Mm1, 5)] + 0.5 * (_f2[LU3(i, Mm1, 1)] - _f2[LU3(i, Mm1, 3)]) - 0.5 * rho * Utop;
                    _f2[LU3(i, Mm1, 8)] = _f2[LU3(i, Mm1, 6)] - 0.5 * (_f2[LU3(i, Mm1, 1)] - _f2[LU3(i, Mm1, 3)]) + 0.5 * rho * Utop;

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
                    double rhowall{_rhobar};
#pragma omp for
                    for (int i = 1; i < N-1; i++){
                        //  Bottom wall, (i, 0)
                        rhowall = _f2[LU3(i,0,0)] +_f2[LU3(i,0,1)] +_f2[LU3(i,0,3)] + 2.0 * (_f2[LU3(i,0,4)] +_f2[LU3(i,0,7)] +_f2[LU3(i,0,8)]);
                        rhof = _f2[LU3(i,1,0)] + _f2[LU3(i,1,1)] + _f2[LU3(i,1,2)] + _f2[LU3(i,1,3)] + _f2[LU3(i,1,4)] + _f2[LU3(i,1,5)]+ _f2[LU3(i,1,6)]+ _f2[LU3(i,1,7)]+ _f2[LU3(i,1,8)];
                        u1f = ((_f2[LU3(i,1,1)] + _f2[LU3(i,1,5)] + _f2[LU3(i,1,8)]) - (_f2[LU3(i,1,3)] + _f2[LU3(i,1,6)] + _f2[LU3(i,1,7)]));
                        u2f = ((_f2[LU3(i,1,2)] + _f2[LU3(i,1,5)] + _f2[LU3(i,1,6)]) - (_f2[LU3(i,1,4)] + _f2[LU3(i,1,7)] + _f2[LU3(i,1,8)]));
                        if (INCOMP != 1){
                            u1f /= rhof;
                            u2f /= rhof;
                        }
                        for (int k = 0; k < Q; k++)
                            _f2[LU3(i,0,k)] = calcfeq(k,Ubot,Vbot,rhowall) + (_f2[LU3(i,1,k)] - calcfeq(k, u1f, u2f, rhof));

                        //  Top Wall, (i,Nm1)
                        rhowall = _f2[LU3(i,Mm1,0)] +_f2[LU3(i,Mm1,1)] +_f2[LU3(i,Mm1,3)] + 2.0 * (_f2[LU3(i,Mm1,2)] +_f2[LU3(i,Mm1,6)] +_f2[LU3(i,Mm1,5)]);
                        rhof = _f2[LU3(i,Mm2,0)] + _f2[LU3(i,Mm2,1)] + _f2[LU3(i,Mm2,2)] + _f2[LU3(i,Mm2,3)] + _f2[LU3(i,Mm2,4)] + _f2[LU3(i,Mm2,5)]+ _f2[LU3(i,Mm2,6)]+ _f2[LU3(i,Mm2,7)]+ _f2[LU3(i,Mm2,8)];
                        u1f = ((_f2[LU3(i,Mm2,1)] + _f2[LU3(i,Mm2,5)] + _f2[LU3(i,Mm2,8)]) - (_f2[LU3(i,Mm2,3)] + _f2[LU3(i,Mm2,6)] + _f2[LU3(i,Mm2,7)]));
                        u2f = ((_f2[LU3(i,Mm2,2)] + _f2[LU3(i,Mm2,5)] + _f2[LU3(i,Mm2,6)]) - (_f2[LU3(i,Mm2,4)] + _f2[LU3(i,Mm2,7)] + _f2[LU3(i,Mm2,8)]));
                        if (INCOMP != 1){
                            u1f /= rhof;
                            u2f /= rhof;
                        }
                        for (int k = 0; k < Q; k++)
                            _f2[LU3(i,Mm1,k)] = calcfeq(k,Utop,Vtop,rhowall) + (_f2[LU3(i,Mm2,k)] - calcfeq(k, u1f, u2f, rhof));
                    }
#pragma omp for
                    for (int i = 1; i < Mm1; i++){
                        //  Left wall, (0,i)
                        rhowall = _f2[LU3(0,i,0)] +_f2[LU3(0,i,2)] +_f2[LU3(0,i,4)] + 2.0 * (_f2[LU3(0,i,3)] +_f2[LU3(0,i,6)] +_f2[LU3(0,i,7)]);
                        rhof = _f2[LU3(1,i,0)] + _f2[LU3(1,i,1)] + _f2[LU3(1,i,2)] + _f2[LU3(1,i,3)] + _f2[LU3(1,i,4)] + _f2[LU3(1,i,5)]+ _f2[LU3(1,i,6)]+ _f2[LU3(1,i,7)]+ _f2[LU3(1,i,8)];
                        u1f = ((_f2[LU3(1,i,1)] + _f2[LU3(1,i,5)] + _f2[LU3(1,i,8)]) - (_f2[LU3(1,i,3)] + _f2[LU3(1,i,6)] + _f2[LU3(1,i,7)]));
                        u2f = ((_f2[LU3(1,i,2)] + _f2[LU3(1,i,5)] + _f2[LU3(1,i,6)]) - (_f2[LU3(1,i,4)] + _f2[LU3(1,i,7)] + _f2[LU3(1,i,8)]));
                        if (INCOMP != 1){
                            u1f /= rhof;
                            u2f /= rhof;
                        }
                        for (int k = 0; k < Q; k++)
                            _f2[LU3(0,i,k)] = calcfeq(k,Ulef,Vlef,rhowall) + (_f2[LU3(1,i,k)] - calcfeq(k, u1f, u2f, rhof));

                        //  Right Wall (Nm1, i)
                        rhowall = _f2[LU3(Nm1,i,0)] +_f2[LU3(Nm1,i,2)] +_f2[LU3(Nm1,i,4)] + 2.0 * (_f2[LU3(Nm1,i,1)] +_f2[LU3(Nm1,i,5)] +_f2[LU3(Nm1,i,8)]);
                        rhof = _f2[LU3(Nm2,i,0)] + _f2[LU3(Nm2,i,1)] + _f2[LU3(Nm2,i,2)] + _f2[LU3(Nm2,i,3)] + _f2[LU3(Nm2,i,4)] + _f2[LU3(Nm2,i,5)]+ _f2[LU3(Nm2,i,6)]+ _f2[LU3(Nm2,i,7)]+ _f2[LU3(Nm2,i,8)];
                        u1f = ((_f2[LU3(Nm2,i,1)] + _f2[LU3(Nm2,i,5)] + _f2[LU3(Nm2,i,8)]) - (_f2[LU3(Nm2,i,3)] + _f2[LU3(Nm2,i,6)] + _f2[LU3(Nm2,i,7)]));
                        u2f = ((_f2[LU3(Nm2,i,2)] + _f2[LU3(Nm2,i,5)] + _f2[LU3(Nm2,i,6)]) - (_f2[LU3(Nm2,i,4)] + _f2[LU3(Nm2,i,7)] + _f2[LU3(Nm2,i,8)]));
                        if (INCOMP != 1){
                            u1f /= rhof;
                            u2f /= rhof;
                        }
                        for (int k = 0; k < Q; k++)
                            _f2[LU3(Nm1,i,k)] = calcfeq(k,Urig,Vrig,rhowall) + (_f2[LU3(Nm2,i,k)] - calcfeq(k, u1f, u2f, rhof));
                    }

                    //  Corners
                    //  Bottom left (0,0)
                    rhof = _f2[LU3(1,1,0)] + _f2[LU3(1,1,1)] + _f2[LU3(1,1,2)] + _f2[LU3(1,1,3)] + _f2[LU3(1,1,4)] + _f2[LU3(1,1,5)]+ _f2[LU3(1,1,6)]+ _f2[LU3(1,1,7)]+ _f2[LU3(1,1,8)];
                    u1f = ((_f2[LU3(1,1,1)] + _f2[LU3(1,1,5)] + _f2[LU3(1,1,8)]) - (_f2[LU3(1,1,3)] + _f2[LU3(1,1,6)] + _f2[LU3(1,1,7)]));
                    u2f = ((_f2[LU3(1,1,2)] + _f2[LU3(1,1,5)] + _f2[LU3(1,1,6)]) - (_f2[LU3(1,1,4)] + _f2[LU3(1,1,7)] + _f2[LU3(1,1,8)]));
					if (INCOMP != 1){
                            u1f /= rhof;
                            u2f /= rhof;
                    }
					rhowall = _rho[LU2(0,0)];
					rhowall = rhof;
#pragma omp for
                    for (int k = 0; k < Q; k++)
                        _f2[LU3(0,0,k)] = calcfeq(k,Ubot,Vlef,rhowall) + (_f2[LU3(1,1,k)] - calcfeq(k, u1f, u2f, rhof));

                    //  Bottom Right (Nm1,0)
                    rhof = _f2[LU3(Nm2,1,0)] + _f2[LU3(Nm2,1,1)] + _f2[LU3(Nm2,1,2)] + _f2[LU3(Nm2,1,3)] + _f2[LU3(Nm2,1,4)] + _f2[LU3(Nm2,1,5)]+ _f2[LU3(Nm2,1,6)]+ _f2[LU3(Nm2,1,7)]+ _f2[LU3(Nm2,1,8)];
                    u1f = ((_f2[LU3(Nm2,1,1)] + _f2[LU3(Nm2,1,5)] + _f2[LU3(Nm2,1,8)]) - (_f2[LU3(Nm2,1,3)] + _f2[LU3(Nm2,1,6)] + _f2[LU3(Nm2,1,7)]));
                    u2f = ((_f2[LU3(Nm2,1,2)] + _f2[LU3(Nm2,1,5)] + _f2[LU3(Nm2,1,6)]) - (_f2[LU3(Nm2,1,4)] + _f2[LU3(Nm2,1,7)] + _f2[LU3(Nm2,1,8)]));
					if (INCOMP != 1){
                            u1f /= rhof;
                            u2f /= rhof;
                    }
					rhowall = _rho[LU2(Nm1,0)];
					rhowall = rhof;
#pragma omp for
                    for (int k = 0; k < Q; k++)
                        _f2[LU3(Nm1,0,k)] = calcfeq(k,Ubot,Vrig,rhowall) + (_f2[LU3(Nm2,1,k)] - calcfeq(k, u1f, u2f, rhof));

                    //  Top Right (Nm1,Nm1)
                    rhof = _f2[LU3(Nm2,Mm2,0)] + _f2[LU3(Nm2,Mm2,1)] + _f2[LU3(Nm2,Mm2,2)] + _f2[LU3(Nm2,Mm2,3)] + _f2[LU3(Nm2,Mm2,4)] + _f2[LU3(Nm2,Mm2,5)]+ _f2[LU3(Nm2,Mm2,6)]+ _f2[LU3(Nm2,Mm2,7)]+ _f2[LU3(Nm2,Mm2,8)];
                    u1f = ((_f2[LU3(Nm2,Mm2,1)] + _f2[LU3(Nm2,Mm2,5)] + _f2[LU3(Nm2,Mm2,8)]) - (_f2[LU3(Nm2,Mm2,3)] + _f2[LU3(Nm2,Mm2,6)] + _f2[LU3(Nm2,Mm2,7)]));
                    u2f = ((_f2[LU3(Nm2,Mm2,2)] + _f2[LU3(Nm2,Mm2,5)] + _f2[LU3(Nm2,Mm2,6)]) - (_f2[LU3(Nm2,Mm2,4)] + _f2[LU3(Nm2,Mm2,7)] + _f2[LU3(Nm2,Mm2,8)]));
					if (INCOMP != 1){
                            u1f /= rhof;
                            u2f /= rhof;
                    }
					rhowall = _rho[LU2(Nm1,Mm1)];
					rhowall = rhof;
#pragma omp for
                    for (int k = 0; k < Q; k++)
                        _f2[LU3(Nm1,Mm1,k)] = calcfeq(k,Utop,Vrig,rhowall) + (_f2[LU3(Nm2,Mm2,k)] - calcfeq(k, u1f, u2f, rhof));

                    //  Top Left (0,Nm1)
                    rhof = _f2[LU3(1,Mm2,0)] + _f2[LU3(1,Mm2,1)] + _f2[LU3(1,Mm2,2)] + _f2[LU3(1,Mm2,3)] + _f2[LU3(1,Mm2,4)] + _f2[LU3(1,Mm2,5)]+ _f2[LU3(1,Mm2,6)]+ _f2[LU3(1,Mm2,7)]+ _f2[LU3(1,Mm2,8)];
                    u1f = ((_f2[LU3(1,Mm2,1)] + _f2[LU3(1,Mm2,5)] + _f2[LU3(1,Mm2,8)]) - (_f2[LU3(1,Mm2,3)] + _f2[LU3(1,Mm2,6)] + _f2[LU3(1,Mm2,7)]));
                    u2f = ((_f2[LU3(1,Mm2,2)] + _f2[LU3(1,Mm2,5)] + _f2[LU3(1,Mm2,6)]) - (_f2[LU3(1,Mm2,4)] + _f2[LU3(1,Mm2,7)] + _f2[LU3(1,Mm2,8)]));
					if (INCOMP != 1){
                            u1f /= rhof;
                            u2f /= rhof;
                    }
					rhowall = _rho[LU2(0,Mm1)];
					rhowall = rhof;
#pragma omp for
                    for (int k = 0; k < Q; k++)
                        _f2[LU3(0,Mm1,k)] = calcfeq(k,Utop,Vlef,rhowall) + (_f2[LU3(1,Mm2,k)] - calcfeq(k, u1f, u2f, rhof));
                };
                break;
            }

            case 1: {   //  Poiseuille Flow
#pragma omp parallel
                {
                    double rhof{},u1f{},u2f{};
                    double rhowall{1.0};
#pragma omp for
                    for (int i = 1; i < N-1; i++){
                        //  Bottom wall, (i, 0)
                        rhowall = _f2[LU3(i,0,0)] +_f2[LU3(i,0,1)] +_f2[LU3(i,0,3)] + 2.0 * (_f2[LU3(i,0,4)] +_f2[LU3(i,0,7)] +_f2[LU3(i,0,8)]);
                        rhof = _f2[LU3(i,1,0)] + _f2[LU3(i,1,1)] + _f2[LU3(i,1,2)] + _f2[LU3(i,1,3)] + _f2[LU3(i,1,4)] + _f2[LU3(i,1,5)]+ _f2[LU3(i,1,6)]+ _f2[LU3(i,1,7)]+ _f2[LU3(i,1,8)];
                        u1f = ((_f2[LU3(i,1,1)] + _f2[LU3(i,1,5)] + _f2[LU3(i,1,8)]) - (_f2[LU3(i,1,3)] + _f2[LU3(i,1,6)] + _f2[LU3(i,1,7)])) / rhof;
                        u2f = ((_f2[LU3(i,1,2)] + _f2[LU3(i,1,5)] + _f2[LU3(i,1,6)]) - (_f2[LU3(i,1,4)] + _f2[LU3(i,1,7)] + _f2[LU3(i,1,8)])) / rhof;
                        for (int k = 0; k < Q; k++)
                            _f2[LU3(i,0,k)] = calcfeq(k,0.0,0.0,rhowall) + (_f2[LU3(i,1,k)] - calcfeq(k, u1f, u2f, rhof));

                        //  Top Wall, (i,Nm1)
                        rhowall = _f2[LU3(i,Mm1,0)] +_f2[LU3(i,Mm1,1)] +_f2[LU3(i,Mm1,3)] + 2.0 * (_f2[LU3(i,Mm1,2)] +_f2[LU3(i,Mm1,6)] +_f2[LU3(i,Mm1,5)]);
                        rhof = _f2[LU3(i,Mm2,0)] + _f2[LU3(i,Mm2,1)] + _f2[LU3(i,Mm2,2)] + _f2[LU3(i,Mm2,3)] + _f2[LU3(i,Mm2,4)] + _f2[LU3(i,Mm2,5)]+ _f2[LU3(i,Mm2,6)]+ _f2[LU3(i,Mm2,7)]+ _f2[LU3(i,Mm2,8)];
                        u1f = ((_f2[LU3(i,Mm2,1)] + _f2[LU3(i,Mm2,5)] + _f2[LU3(i,Mm2,8)]) - (_f2[LU3(i,Mm2,3)] + _f2[LU3(i,Mm2,6)] + _f2[LU3(i,Mm2,7)])) / rhof;
                        u2f = ((_f2[LU3(i,Mm2,2)] + _f2[LU3(i,Mm2,5)] + _f2[LU3(i,Mm2,6)]) - (_f2[LU3(i,Mm2,4)] + _f2[LU3(i,Mm2,7)] + _f2[LU3(i,Mm2,8)])) / rhof;
                        for (int k = 0; k < Q; k++)
                            _f2[LU3(i,Mm1,k)] = calcfeq(k,0.0,0.0,rhowall) + (_f2[LU3(i,Mm2,k)] - calcfeq(k, u1f, u2f, rhof));
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
                    for (int i = 1; i < N-1; i++){
                        //  Bottom wall, (i, 0)
                        rhowall = _f2[LU3(i,0,0)] +_f2[LU3(i,0,1)] +_f2[LU3(i,0,3)] + 2.0 * (_f2[LU3(i,0,4)] +_f2[LU3(i,0,7)] +_f2[LU3(i,0,8)]);
                        rhof = _f2[LU3(i,1,0)] + _f2[LU3(i,1,1)] + _f2[LU3(i,1,2)] + _f2[LU3(i,1,3)] + _f2[LU3(i,1,4)] + _f2[LU3(i,1,5)]+ _f2[LU3(i,1,6)]+ _f2[LU3(i,1,7)]+ _f2[LU3(i,1,8)];
                        u1f = ((_f2[LU3(i,1,1)] + _f2[LU3(i,1,5)] + _f2[LU3(i,1,8)]) - (_f2[LU3(i,1,3)] + _f2[LU3(i,1,6)] + _f2[LU3(i,1,7)])) / rhof;
                        u2f = ((_f2[LU3(i,1,2)] + _f2[LU3(i,1,5)] + _f2[LU3(i,1,6)]) - (_f2[LU3(i,1,4)] + _f2[LU3(i,1,7)] + _f2[LU3(i,1,8)])) / rhof;
                        for (int k = 0; k < Q; k++)
                            _f2[LU3(i,0,k)] = calcfeq(k,Ubot,0.0,rhowall) + (_f2[LU3(i,1,k)] - calcfeq(k, u1f, u2f, rhof));

                        //  Top Wall, (i,Nm1)
                        rhowall = _f2[LU3(i,Mm1,0)] +_f2[LU3(i,Mm1,1)] +_f2[LU3(i,Mm1,3)] + 2.0 * (_f2[LU3(i,Mm1,2)] +_f2[LU3(i,Mm1,6)] +_f2[LU3(i,Mm1,5)]);
                        rhof = _f2[LU3(i,Mm2,0)] + _f2[LU3(i,Mm2,1)] + _f2[LU3(i,Mm2,2)] + _f2[LU3(i,Mm2,3)] + _f2[LU3(i,Mm2,4)] + _f2[LU3(i,Mm2,5)]+ _f2[LU3(i,Mm2,6)]+ _f2[LU3(i,Mm2,7)]+ _f2[LU3(i,Mm2,8)];
                        u1f = ((_f2[LU3(i,Mm2,1)] + _f2[LU3(i,Mm2,5)] + _f2[LU3(i,Mm2,8)]) - (_f2[LU3(i,Mm2,3)] + _f2[LU3(i,Mm2,6)] + _f2[LU3(i,Mm2,7)])) / rhof;
                        u2f = ((_f2[LU3(i,Mm2,2)] + _f2[LU3(i,Mm2,5)] + _f2[LU3(i,Mm2,6)]) - (_f2[LU3(i,Mm2,4)] + _f2[LU3(i,Mm2,7)] + _f2[LU3(i,Mm2,8)])) / rhof;
                        for (int k = 0; k < Q; k++)
                            _f2[LU3(i,Mm1,k)] = calcfeq(k,Utop,0.0,rhowall) + (_f2[LU3(i,Mm2,k)] - calcfeq(k, u1f, u2f, rhof));
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
        double rhowall = _rhobar;
        switch (BC){
            case 2: {    //  Lid-driven cavity flow
#pragma omp parallel for
                for (int i = 1; i < Mm1; i++){
                    //  Left wall
//					rhowall = _f2[LU3(0,i,0)] +_f2[LU3(0,i,2)] +_f2[LU3(0,i,4)] + 2.0 * (_f2[LU3(0,i,3)] +_f2[LU3(0,i,6)] +_f2[LU3(0,i,7)]);
                    _f2[LU3(0, i, 5)] = _fstar[LU3(0, i, 7)] - 2.0 * w[7] * rhowall * c[7][1] * Vlef * 3.0;
                    _f2[LU3(0, i, 1)] = _fstar[LU3(0, i, 3)];
                    _f2[LU3(0, i, 8)] = _fstar[LU3(0, i, 6)] - 2.0 * w[6] * rhowall * c[6][1] * Vlef * 3.0;

                    //  Right wall
//					 rhowall = _f2[LU3(Nm1,i,0)] +_f2[LU3(Nm1,i,2)] +_f2[LU3(Nm1,i,4)] + 2.0 * (_f2[LU3(Nm1,i,1)] +_f2[LU3(Nm1,i,5)] +_f2[LU3(Nm1,i,8)]);
                    _f2[LU3(Nm1, i, 7)] = _fstar[LU3(Nm1, i, 5)] - 2.0 * w[5] * rhowall * c[5][1] * Vrig * 3.0;
                    _f2[LU3(Nm1, i, 3)] = _fstar[LU3(Nm1, i, 1)];
                    _f2[LU3(Nm1, i, 6)] = _fstar[LU3(Nm1, i, 8)] - 2.0 * w[8] * rhowall * c[8][1] * Vrig * 3.0;
                }

#pragma omp parallel for
                for (int i = 1; i < Nm1; i++){
                    //  Bottom wall
//					 rhowall = _f2[LU3(i,0,0)] +_f2[LU3(i,0,1)] +_f2[LU3(i,0,3)] + 2.0 * (_f2[LU3(i,0,4)] +_f2[LU3(i,0,7)] +_f2[LU3(i,0,8)]);
                    _f2[LU3(i, 0, 6)] = _fstar[LU3(i, 0, 8)] - 2.0 * w[8] * rhowall * c[8][0] * Ubot * 3.0;
                    _f2[LU3(i, 0, 2)] = _fstar[LU3(i, 0, 4)];
                    _f2[LU3(i, 0, 5)] = _fstar[LU3(i, 0, 7)] - 2.0 * w[7] * rhowall * c[7][0] * Ubot * 3.0;

                    //  Top wall
//					rhowall = _f2[LU3(i,Mm1,0)] +_f2[LU3(i,Mm1,1)] +_f2[LU3(i,Mm1,3)] + 2.0 * (_f2[LU3(i,Mm1,2)] +_f2[LU3(i,Mm1,6)] +_f2[LU3(i,Mm1,5)]);
                    _f2[LU3(i, Mm1, 8)] = _fstar[LU3(i, Mm1, 6)] - 2.0 * w[6] * rhowall * c[6][0] * Utop * 3.0;
                    _f2[LU3(i, Mm1, 4)] = _fstar[LU3(i, Mm1, 2)];
                    _f2[LU3(i, Mm1, 7)] = _fstar[LU3(i, Mm1, 5)] - 2.0 * w[5] * rhowall * c[5][0] * Utop * 3.0;
                }

                //  Corners
                //  Top left
//				rhowall = _rho[LU2(0, Mm1)];
                _f2[LU3(0, Mm1, 1)] = _fstar[LU3(0, Mm1, 3)];
                _f2[LU3(0, Mm1, 8)] = _fstar[LU3(0, Mm1, 6)] - 2.0 * w[6] * rhowall * (c[6][0] * Utop + c[6][1] * Vlef) * 3.0;
                _f2[LU3(0, Mm1, 4)] = _fstar[LU3(0, Mm1, 2)];

                //  Bottom Left
//				rhowall = _rho[LU2(0, 0)];
                _f2[LU3(0, 0, 1)] = _fstar[LU3(0, 0, 3)];
                _f2[LU3(0, 0, 5)] = _fstar[LU3(0, 0, 7)] - 2.0 * w[7] * rhowall * (c[7][0] * Ubot + c[7][1] * Vlef) * 3.0;
                _f2[LU3(0, 0, 2)] = _fstar[LU3(0, 0, 4)];

                //  Top Right
//				rhowall = _rho[LU2(Nm1, Mm1)];
                _f2[LU3(Nm1, Mm1, 3)] = _fstar[LU3(Nm1, Mm1, 1)];
                _f2[LU3(Nm1, Mm1, 7)] = _fstar[LU3(Nm1, Mm1, 5)] - 2.0 * w[5] * rhowall * (c[5][0] * Utop + c[5][1] * Vrig) * 3.0;
                _f2[LU3(Nm1, Mm1, 4)] = _fstar[LU3(Nm1, Mm1, 2)];

                //  Bottom Right
//				rhowall = _rho[LU2(Nm1, 0)];
                _f2[LU3(Nm1, 0, 3)] = _fstar[LU3(Nm1, 0, 1)];
                _f2[LU3(Nm1, 0, 6)] = _fstar[LU3(Nm1, 0, 8)] - 2.0 * w[8] * rhowall * (c[8][0] * Ubot + c[8][1] * Vrig) * 3.0;
                _f2[LU3(Nm1, 0, 2)] = _fstar[LU3(Nm1, 0, 4)];
                break;
            }

            case 1: {   //  Poiseuille Flow
#pragma omp parallel for
                for (int i = 0; i < N; i++){
                    //  Bottom wall
                    _f2[LU3(i, 0, 6)] = _fstar[LU3(i, 0, 8)];
                    _f2[LU3(i, 0, 2)] = _fstar[LU3(i, 0, 4)];
                    _f2[LU3(i, 0, 5)] = _fstar[LU3(i, 0, 7)];

                    //  Top wall
                    _f2[LU3(i, Mm1, 8)] = _fstar[LU3(i, Mm1, 6)];
                    _f2[LU3(i, Mm1, 4)] = _fstar[LU3(i, Mm1, 2)];
                    _f2[LU3(i, Mm1, 7)] = _fstar[LU3(i, Mm1, 5)];
                }
                break;
            }

            case 0: {       //  Couette flow
#pragma omp parallel for
                for (int i = 0; i < N; i++){
                    //  Bottom wall
                    _f2[LU3(i, 0, 6)] = _fstar[LU3(i, 0, 8)] - 2.0 * w[8] * _rhobar * c[8][0] * Ubot * 3.0;
                    _f2[LU3(i, 0, 2)] = _fstar[LU3(i, 0, 4)] - 2.0 * w[4] * _rhobar * c[4][0] * Ubot * 3.0;
                    _f2[LU3(i, 0, 5)] = _fstar[LU3(i, 0, 7)] - 2.0 * w[7] * _rhobar * c[7][0] * Ubot * 3.0;

                    //  Top wall
                    _f2[LU3(i, Mm1, 8)] = _fstar[LU3(i, Mm1, 6)] - 2.0 * w[6] * _rhobar * c[6][0] * Utop * 3.0;
                    _f2[LU3(i, Mm1, 4)] = _fstar[LU3(i, Mm1, 2)] - 2.0 * w[2] * _rhobar * c[2][0] * Utop * 3.0;
                    _f2[LU3(i, Mm1, 7)] = _fstar[LU3(i, Mm1, 5)] - 2.0 * w[5] * _rhobar * c[5][0] * Utop * 3.0;
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
        double temp{},fTotalx{},fTotaly{};
        int ind1{};
        _rhobar = 0.0;
#pragma omp parallel for private(temp,ind1,fTotalx,fTotaly) collapse(2) reduction(+:_rhobar)
        for (int i = 0; i < N; i++){
            for (int j = 0; j < M; j++){
                ind1 = LU2(i,j);
				fTotalx = _forceX[ind1] + _fBodyX;
				fTotaly = _forceY[ind1] + _fBodyY;
                temp = _f2[LU3(i,j,0)] + _f2[LU3(i,j,1)] + _f2[LU3(i,j,2)] + _f2[LU3(i,j,3)] + _f2[LU3(i,j,4)] + _f2[LU3(i,j,5)] + _f2[LU3(i,j,6)] + _f2[LU3(i,j,7)] + _f2[LU3(i,j,8)];
                _rho[ind1] = temp;
                _rhobar += (temp / N2);
                _u1[ind1] = ((_f2[LU3(i,j,1)] + _f2[LU3(i,j,5)] + _f2[LU3(i,j,8)]) - (_f2[LU3(i,j,3)] + _f2[LU3(i,j,6)] + _f2[LU3(i,j,7)]));
                _u2[ind1] = ((_f2[LU3(i,j,2)] + _f2[LU3(i,j,5)] + _f2[LU3(i,j,6)]) - (_f2[LU3(i,j,4)] + _f2[LU3(i,j,7)] + _f2[LU3(i,j,8)]));
				if (INCOMP != 1){
					_u1[ind1] /= temp;
					_u2[ind1] /= temp;
					_p[ind1] = _rho[ind1] / 3.0;
				}
				else {
    				_p[ind1] = (1.0/(3.0*(1.0 - w[0]))) * (calcfeq(0,ind1)+calcfeq(1,ind1)+calcfeq(2,ind1)+calcfeq(3,ind1)+calcfeq(4,ind1)
                                      +calcfeq(5,ind1)+calcfeq(6,ind1)+calcfeq(7,ind1)+calcfeq(8,ind1) - 1.5 * (P2(_u1[ind1]) + P2(_u2[ind1])));
				}
				
				//	Forces
				_u1[ind1] += 0.5 * fTotalx / temp;
				_u2[ind1] += 0.5 * fTotaly / temp;
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
        fprintf(f, "TITLE=\"%s\" VARIABLES=\"x\", \"y\", \"u\", \"v\", \"vmag\", \"omegaxy\", \"vortz\" ZONE T=\"%s\" I=%d J=%d F=POINT\n", _filename1, _filename1,N,M);
        for (int j = 0; j < M; j++) {
            for (int i = 0; i < N; i++) {
                ind1 = LU2(i,j);
                fprintf(f, "%.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f\n", _x[i], _y[j], _u1[ind1] / _Umax,_u2[ind1] / _Umax, _vmag[ind1], _stress[ind1], _vort[ind1]);
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

    inline void immersedBoundary(){
        //  Calculate vector B
		Eigen::VectorXd matrixBx = Eigen::VectorXd::Zero(IBN),matrixBy = Eigen::VectorXd::Zero(IBN);
        double sumx{},sumy{},Dirac;
        for (int i = 0; i < IBN; i++){
            sumx = 0.0;
            sumy = 0.0;
#pragma omp parallel for collapse(2) private(Dirac) reduction(+:sumx,sumy)
            for (int k = (int)_IBrxmn; k <= (int)_IBrxmx; k++) {
                for (int l =(int) _IBrymn; l <= (int)_IBrymx; l++) {
					Dirac = diracdelta(_IBrx[i] - k) * diracdelta(_IBry[i] - l);
                    sumx += _u1[LU2(k,l)] * Dirac;
                    sumy += _u2[LU2(k,l)] * Dirac;
                }
            }
            matrixBx(i) = _IBub[i] - sumx;
            matrixBy(i) = _IBvb[i] - sumy;
        }
		
		//	Solve for velocity correction
		Eigen::VectorXd IBdu = Eigen::VectorXd::Zero(IBN),IBdv = Eigen::VectorXd::Zero(IBN);
		IBdu = _IBmatrixAxInv * matrixBx;
		IBdv = _IBmatrixAyInv * matrixBy;

		//	Spread du to LBM grid
		double du,dv,ind1;
#pragma omp parallel for collapse(2) private(du,dv,ind1,Dirac)
		for (int i = (int)_IBrxmn; i <= (int)_IBrxmx; i++){
			for (int j = (int)_IBrymn; j <= (int)_IBrymx; j++){
				du = 0.0; dv = 0.0; ind1 = LU2(i,j);_forceX[ind1] = 0.0;_forceY[ind1] = 0.0;
				for (int k = 0; k < IBN; k++){
					Dirac = diracdelta(_IBrx[k] - i) * diracdelta(_IBry[k] - j);
					du += IBdu[k] * Dirac;
					dv += IBdv[k] * Dirac;
				}
				_u1[ind1] += du;
				_u2[ind1] += dv;
				_forceX[ind1] += 2.0 * du * _rho[ind1];
				_forceY[ind1] += 2.0 * dv * _rho[ind1];
			}
		}
    }

private:
    vector<double> _f1,_f2,_fstar,_u1,_u2,_rho,_p,_x,_y,_error,_vmag,_stress,_vort,_dudx,_dudy,_dvdx,_dvdy,_forceX,_forceY;
    double _df, _df0, _MACHSTAR, _TAU_P, _OMEGA, _OMEGAm, _CS, _MACH, _rhobar;
    char _filename1[80];
    char _filename2[80];
    double _omega_e,_omega_eps,_omega_q,_omega_nu, _GS[9];
    double _Umax;
	double _pi = 3.1415926535897;
	vector<double> _IBrx, _IBry, _IBur, _IBvr, _IBFx, _IBFy, _IBub, _IBvb;
	Eigen::MatrixXd _IBmatrixAx, _IBmatrixAy, _IBmatrixAxInv, _IBmatrixAyInv;
	double _fBodyX,_fBodyY,_IBds,_IBrxmx,_IBrxmn,_IBrymx,_IBrymn;
	
    //  Left-right periodicity
    inline void virtualnode(){
#pragma omp parallel for
        for (int j = 1; j < Mm1; j++) {
            _fstar[LU3(0, j, 1)] = _fstar[LU3(Nm2, j, 1)];
            _fstar[LU3(0, j, 5)] = _fstar[LU3(Nm2, j, 5)];
            _fstar[LU3(0, j, 8)] = _fstar[LU3(Nm2, j, 8)];

            _fstar[LU3(Nm1, j, 3)] = _fstar[LU3(1, j, 3)];
            _fstar[LU3(Nm1, j, 6)] = _fstar[LU3(1, j, 6)];
            _fstar[LU3(Nm1, j, 7)] = _fstar[LU3(1, j, 7)];
        }

        //  Top Left
        _fstar[LU3(0, Mm1, 1)] = _fstar[LU3(Nm2, Mm1, 1)];
        _fstar[LU3(0, Mm1, 4)] = _fstar[LU3(Nm2, Mm1, 4)];
        _fstar[LU3(0, Mm1, 8)] = _fstar[LU3(Nm2, Mm1, 8)];

        //  Top Right
        _fstar[LU3(Nm1, Mm1, 3)] = _fstar[LU3(1, Mm1, 3)];
        _fstar[LU3(Nm1, Mm1, 7)] = _fstar[LU3(1, Mm1, 7)];
        _fstar[LU3(Nm1, Mm1, 4)] = _fstar[LU3(1, Mm1, 4)];

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
        for (int i = 0; i < _num; i++)
            x[i] = _start + i * (_end - _start) / (_num - 1.0);
    }

    inline double calcfeq(const int k, const int ind, const int check = 1){
        const double u1ij=_u1[ind], u2ij = _u2[ind];
        double cdotu{},feq{},u2{},rho0=_rho[ind];
		if (INCOMP==1){
			u2 = P2(u1ij) + P2(u2ij);
			const double s0 = w[0] * (-1.5 * u2);
			const double p = (1.0 / (3.0*(1.0 - w[0]))) * (_rho[ind] + s0);
			cdotu = c[k][0] * u1ij + c[k][1] * u2ij;
			const double s = w[k]*(3.0 * cdotu + 4.5 * P2(cdotu) - 1.5 * u2);
			if (k == 0) {
				feq = _rho[ind] - (1.0 - w[0]) * 3.0 * p + s0;
			}
			else {
				feq = w[k] * 3.0 * p + s;
			}
		}
		else{
			u2 = P2(u1ij) + P2(u2ij);
			cdotu = c[k][0] * u1ij + c[k][1] * u2ij;
			feq = w[k] * (_rho[ind] + rho0 * (3.0 * cdotu + (4.5 * P2(cdotu) - 1.5 * u2) / GAMMA));
			if (check == 1)
				checkfeq(feq,ind);
		}

        return feq;
    }

    inline double calcfeq(const int k, const double u1ij, const double u2ij, const double rhoij, const int check = 1){
        double cdotu{},feq{},u2{}, rho0=rhoij;
		if (INCOMP==1){
			u2 = P2(u1ij) + P2(u2ij);
			const double s0 = w[0] * (-1.5 * u2);
			const double p = (1.0 / (3.0*(1.0 - w[0]))) * (rhoij + s0);
			cdotu = c[k][0] * u1ij + c[k][1] * u2ij;
			const double s = w[k]*(3.0 * cdotu + 4.5 * P2(cdotu) - 1.5 * u2);
			if (k == 0) {
				feq = rhoij - (1.0 - w[0]) * 3.0 * p + s0;
			}
			else {
				feq = w[k] * 3.0 * p + s;
			}
		}
		else{
			u2 = P2(u1ij) + P2(u2ij);
			cdotu = c[k][0] * u1ij + c[k][1] * u2ij;
			feq = w[k] * (rhoij + rho0 * (3.0 * cdotu + (4.5 * P2(cdotu) - 1.5 * u2) / GAMMA));
			if (check == 1)
				checkfeq(feq,1);
		}
        return feq;
    }

    //  Checks for negative feq
    inline void checkfeq(const double value, const int index){
        if (value < 0) {
            printf("Error: negative feq at index %d.  Therefore, unstable.\n",index);
            exit(1);
        }
    }

    inline double rmsError(){
        double difference{};
#pragma omp parallel for reduction(+:difference)
        for (int i = 0; i < NQ; i++)
            difference += P2(_f2[i] - _f1[i]);
        return sqrt(difference / NQ);
    }

    inline void calcmeq(vector<double> &meq, const double u1, const double u2, const double rho, const double pres) {
        const double u12 = P2(u1);
        const double u22 = P2(u2);
        if (INCOMP==1) {
            const double alpha2 = 24.0, alpha3 = -36.0;
            const double c1 = -2.0, c2 = -2.0;
            const double gamma1 = 2.0/3.0,gamma2 = 18.0,gamma3=2.0/3.0,gamma4=-18.0;
            meq[0] = rho;                               //  rho
            meq[1] = 0.25*alpha2*pres + (gamma2/6.0)*(u12+u22);  //  e
            meq[2] = 0.25*alpha3*pres + (gamma4/6.0)*(u12+u22);       //  eps
            meq[3] = u1;                          //  jx
            meq[4] = 0.5*c1*u1;                           //  qx
            meq[5] = u2;                          //  jy
            meq[6] = 0.5*c2*u2;                           //  qy
            meq[7] = 1.5*gamma1*(u12-u22);                 //  pxx
            meq[8] = 1.5*gamma3*u1*u2;                     //  pxy
        }
        else {
            meq[0] = rho;                               //  rho
            meq[1] = -2 * rho + 3 * rho * (u12 + u22) / GAMMA;  //  e
            meq[2] = rho - 3 * rho * (u12 + u22) / GAMMA;       //  eps
            meq[3] = rho * u1;                          //  jx
            meq[4] = -meq[3];                           //  qx
            meq[5] = rho * u2;                          //  jy
            meq[6] = -meq[5];                           //  qy
            meq[7] = rho * (u12 - u22) / GAMMA;                 //  pxx
            meq[8] = rho * u1 * u2 / GAMMA;                     //  pxy
    }
        
        
        
    }

    inline int LU2(const int i, const int j){
		return N*j + i;
	}
        
    inline int LU3(const int i, const int j, const int k){
		return N2*k + N*j + i;
	}
        
    inline double P2(const double value){
		return (value * value);
	}
        
    inline void calcvmag(){
#pragma omp parallel for
        for (int i = 0; i < N2; i++)
            _vmag[i] = sqrt(P2(_u1[i]) + P2(_u2[i]));
    }

    inline void calcstress(){
#pragma omp parallel
        {
            int ind1;
#pragma omp for collapse(2)
            for (int i = 0; i < N; i++){
                for (int j = 0; j < M; j++){
                    ind1 = LU2(i,j);
                    for (int k = 0; k < Q; k++){
                        _stress[ind1] -= (1.0 - 0.5 * _OMEGA) * c[k][0] * c[k][1] * (_f2[LU3(i,j,k)] - calcfeq(k,ind1));
                    }
                    if (BC == 1)
                        _stress[ind1] -= 0.5 * (1.0 - 0.5 * _OMEGA) * (_forceX[ind1] * _u2[ind1] + _forceY[ind1] * _u1[ind1]);
                }
            }
        }
    }

    inline void calcVort(){
        calcDerivatives();
#pragma omp parallel for
        for (int i = 0; i < N2; i++)
            _vort[i] = _dvdx[i] - _dudy[i];
    }

    inline void calcDerivatives(){
        //  2rd order Finite Difference
        int indS{},indE{},ind{},indN{},indW{};
        double hinv = 1.0 / ((_x[1] - _x[0]) * _Umax);  //  scales derivatives in space and velocity scale
        //        //  d/dx central differences
        for (int i = 1; i < Nm1; i++){
            for (int j = 1; j < M; j++){
                ind = LU2(i,j);     //  Current point
                indE = LU2(i+1,j);  //  West
                indW = LU2(i-1,j);  //  East
                _dudx[ind] = 0.5 * (_u1[indE] - _u1[indW]) * hinv;
                _dvdx[ind] = 0.5 * (_u2[indE] - _u2[indW]) * hinv;
            }
        }

        //  d/dy central differences
        for (int i = 0; i < N; i++){
            for (int j = 1; j < Mm1; j++){
                ind = LU2(i,j);     //  Current point
                indN = LU2(i,j+1);  //  North
                indS = LU2(i,j-1);  //  South
                _dudy[ind] = 0.5 * (_u1[indN] - _u1[indS]) * hinv;
                _dvdy[ind] = 0.5 * (_u2[indN] - _u2[indS]) * hinv;
            }
        }

        int indSS{},indNN{},indEE{},indWW{};
        //  d/dx forward (i = 0) & backward (i = Nm1) differences
        for (int j = 0; j < M; j++){
            //  forward
            ind     = LU2(0,j);
            indE    = LU2(1,j);
            indEE   = LU2(2,j);
            _dudx[ind] = (-1.5 * _u1[ind] + 2.0 * _u1[indE] - 0.5 * _u1[indEE]) * hinv;
            _dvdx[ind] = (-1.5 * _u2[ind] + 2.0 * _u2[indE] - 0.5 * _u2[indEE]) * hinv;

            //  backward
            ind     = LU2(Nm1,j);
            indW    = LU2(Nm2,j);
            indWW   = LU2(N - 3,j);
            _dudx[ind] = (1.5 * _u1[ind] - 2.0 * _u1[indW] + 0.5 * _u1[indWW]) * hinv;
            _dvdx[ind] = (1.5 * _u2[ind] - 2.0 * _u2[indW] + 0.5 * _u2[indWW]) * hinv;
        }

        //  d/dy forward (j = 0) & backward (j = Mm1) differences
        for (int i = 0; i < N; i++){
            //  forward
            ind     = LU2(i,0);
            indN    = LU2(i,1);
            indNN   = LU2(i,2);
            _dudy[ind] = (-1.5 * _u1[ind] + 2.0 * _u1[indN] - 0.5 * _u1[indNN]) * hinv;
            _dvdy[ind] = (-1.5 * _u2[ind] + 2.0 * _u2[indN] - 0.5 * _u2[indNN]) * hinv;

            //  backward
            ind     = LU2(i,Mm1);
            indS    = LU2(i,Mm2);
            indSS   = LU2(i,M - 3);
            _dudy[ind] = (1.5 * _u1[ind] - 2.0 * _u1[indS] + 0.5 * _u1[indSS]) * hinv;
            _dvdy[ind] = (1.5 * _u2[ind] - 2.0 * _u2[indS] + 0.5 * _u2[indSS]) * hinv;
        }
    }

    inline double diracdelta(const double x, const int order = 4){
        const double absx = abs(x);
        double phi{};
        const double dx = 1.0;
        switch (order){
            case 2 : {
                if (absx < dx)
                    phi = 1.0 - absx;
                else
                    phi = 0.0;
                break;
            }
            case 3 : {
                if (absx <= 0.5 * dx)
                    phi = (1.0/3.0) * (1.0 + sqrt(1.0  - 3.0 * P2(absx)));
                else if (absx <= 1.5 * dx)
                    phi = (1.0/6.0) * (5.0 - 3.0 * absx - sqrt(-2.0 + 6.0 * absx - 3.0 * P2(absx)));
                else
                    phi = 0.0;
                break;
            }
            case 4 : {
                if (absx <= dx)
                    phi = (1.0/8.0) * (3.0 - 2.0 * absx + sqrt(1.0 + 4.0 * absx - 4.0 * P2(absx)));
                else if (absx <= (2.0 * dx))
                    phi = (1.0/8.0) * (5.0 - 2.0 * absx - sqrt(-7.0 + 12.0 * absx - 4.0 * P2(absx)));
                else
                    phi = 0.0;
                break;
            }
            default : {
                cout << "Invalid order for delta function.  Must be 2,3, or 4" << endl;
                exit(1);
            }
        }
        return phi;
    }

    bool diagdominant(const vector<vector<double>> &matrix, const int size){
        double diag{},sum{};
        bool dd = true;
        for (int i = 0; i < size; i++){
            diag = matrix[i][i];
            sum = 0.0;
            for (int j = 0; j < size; j++){
                if (j != i)
                    sum += matrix[i][j];
            }
            if (sum > diag){
                dd = false;
                break;
            }
        }
        return dd;
    }
};
#endif //LIDDRIVENCAVITYLBM_LBMCLASS_H
