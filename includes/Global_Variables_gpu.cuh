#ifndef Global_Variables_GPU_CUH
#define Global_Variables_GPU_CUH
#include "externLib.h"
#include "solver_precision.h"
#include "Module_extern.h"
__constant__ T_P PI_gpu;

__constant__ I_INT  NXG1_D;
__constant__ I_INT  NYG1_D;
__constant__ I_INT  NZG1_D;

/* Total number of grid nodes */
__constant__ I_INT  NGRID1_D;
__constant__ I_INT  NGRID0_D;

/* the number of grids on different faces of the domain (interior point) */
__constant__ I_INT  XYG0_D;
__constant__ I_INT  XZG0_D;
__constant__ I_INT  YZG0_D;
/* the number of grids on different faces of the domain (one ghost layer) */
__constant__ I_INT  XYG1_D;		
__constant__ I_INT  XZG1_D;
__constant__ I_INT  YZG1_D;

__constant__ T_P dx_LBM_gpu, dt_LBM_gpu, c_LBM_gpu, overc_LBM_gpu; //the size of grid step, the size of time step; dx_LBM/dt_LBM; prc(1.)/c_LBM

__constant__ int ex_gpu[NDIR], ey_gpu[NDIR], ez_gpu[NDIR];
__constant__ T_P w_equ_gpu[NDIR];
__constant__ I_INT nxGlobal_gpu, nyGlobal_gpu, nzGlobal_gpu;

__constant__ I_INT nAbsorbingL_gpu;  //number of Absorbing Layers

__constant__ T_P Lid_velocity_gpu, Density_gpu, SRT_OverTau_gpu;   //lid driven boundary

__constant__ T_P Frequency_lbm_gpu, AnguFreq_lbm_gpu;   //frequency and of Angular frequency sound wave in LBM system
__constant__ T_P Velocity_Bound_gpu;  //velocity in boundary for gpu

__constant__ T_P body_accelerate_gpu[3];       //body force in gpu

__constant__ T_P Sphere_radius_gpu;   //the radius of sphere particle

__constant__ I_INT Length_gpu[NDIR];
__constant__ int Reverse_gpu[NDIR] ; // reverse directions
// The index of velocity direction required to bounce on the boundary
__constant__ int BB_xy_top_gpu[5]   ; // The axis direction need in the first one
__constant__ int BB_yz_front_gpu[5] ; // The axis direction need in the first one
__constant__ int BB_xz_Right_gpu[5] ; // The axis direction need in the first one
// the transformation matrix M  and it's inverse in gpu
__constant__ int MRT_Trans_M_gpu[NDIR*NDIR];
__constant__ T_P MRT_Trans_M_inverse_gpu[NDIR*NDIR];
__constant__ int MRT_Trans_M_inverse_int_gpu[NDIR*NDIR];   //MRT_Trans_M_inverse*47880
__constant__ T_P MRT_C_inverse_int_gpu;   //1./47880
__constant__ T_P MRT_Collision_M_gpu[NDIR*NDIR];  // The collision matrix of MRT = inverse(M)*S*M
__constant__ T_P MRT_S_gpu[NDIR];
#endif