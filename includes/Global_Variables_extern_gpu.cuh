// // #ifndef Global_Variables_Extern_GPU_CUH
// // #define Global_Variables_Extern_GPU_CUH
// #include "externLib.h"
// #include "solver_precision.h"
// #include "Module_extern.h"

// extern __constant__ I_INT  NXG1_D;
// extern __constant__ I_INT  NYG1_D;
// extern __constant__ I_INT  NZG1_D;

// /* Total number of grid nodes */
// extern __constant__ I_INT  NGRID1_D;
// extern __constant__ I_INT  NGRID0_D;

// /* the number of grids on different faces of the domain (interior point) */
// extern __constant__ I_INT  XYG0_D;
// extern __constant__ I_INT  XZG0_D;
// extern __constant__ I_INT  YZG0_D;
// /* the number of grids on different faces of the domain (one ghost layer) */
// extern __constant__ I_INT  XYG1_D;		
// extern __constant__ I_INT  XZG1_D;
// extern __constant__ I_INT  YZG1_D;

// extern __constant__ int ex_gpu[NDIR], ey_gpu[NDIR], ez_gpu[NDIR];
// extern __constant__ T_P w_equ_gpu[NDIR];
// extern __constant__ I_INT nxGlobal_gpu, nyGlobal_gpu, nzGlobal_gpu;

// extern __constant__ I_INT nAbsorbingL_gpu;  //number of Absorbing Layers

// extern __constant__ T_P Lid_velocity_gpu, Density_gpu, SRT_OverTau_gpu;   //lid driven boundary

// extern __constant__ I_INT Length_gpu[NDIR];
// extern __constant__ int Reverse_gpu[NDIR] ; // reverse directions
// // The index of velocity direction required to bounce on the boundary
// extern __constant__ int BB_xy_top_gpu[5]   ; // The axis direction need in the first one
// extern __constant__ int BB_yz_front_gpu[5] ; // The axis direction need in the first one
// extern __constant__ int BB_xz_Right_gpu[5] ; // The axis direction need in the first one
// // the transformation matrix M  and it's inverse in gpu
// extern __constant__ int MRT_Trans_M_gpu[NDIR*NDIR];
// extern __constant__ T_P MRT_Trans_M_inverse_gpu[NDIR*NDIR];
// extern __constant__ int MRT_Trans_M_inverse_int_gpu[NDIR*NDIR];   //MRT_Trans_M_inverse*47880
// extern __constant__ T_P MRT_C_inverse_int_gpu;   //1./47880
// extern __constant__ T_P MRT_Collision_M_gpu[NDIR*NDIR];  // The collision matrix of MRT = inverse(M)*S*M
// extern __constant__ T_P MRT_S_gpu[NDIR];

// // #endif