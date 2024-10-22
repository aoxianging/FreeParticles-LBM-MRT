#ifndef INDEX_GPU_CUH
#define INDEX_GPU_CUH
#include "externLib_CUDA.cuh"
#include "Global_Variables_gpu.cuh"

/* Total number of array nodes after adding the ghost nodes */
// zero ghost layers
#define NXG0_D	(nxGlobal_gpu)
#define NYG0_D	(nyGlobal_gpu)
#define NZG0_D	(nzGlobal_gpu)
// // one ghost layer
// #define NXG1_D	(nxGlobal_gpu + 2)
// #define NYG1_D	(nyGlobal_gpu + 2)
// #define NZG1_D	(nzGlobal_gpu + 2)

// /* Total number of grid nodes */
// #define NGRID1_D	(NXG1_D * NYG1_D * NZG1_D)
// #define NGRID0_D	(NXG0_D * NYG0_D * NZG0_D)

// /* the number of grids on different faces of the domain (interior point) */
// #define XYG0_D	(NXG0_D*NYG0_D)			
// #define XZG0_D	(NXG0_D*NZG0_D)
// #define YZG0_D	(NYG0_D*NZG0_D)
// /* the number of grids on different faces of the domain (one ghost layer) */
// #define XYG1_D	(NXG1_D*NYG1_D)			
// #define XZG1_D	(NXG1_D*NZG1_D)
// #define YZG1_D	(NYG1_D*NZG1_D)

/* indexing functions for arrays */
// scalar array with one ghost layer
#define p_index_D(x,y,z) ((x) + (NXG1_D) * ((y) + (NYG1_D) * (z)))

// Calculates the position of the index in the array after the corresponding time step
//#define p_step_index_D(p, i_f, nt) (((p + LENGTH_D(i_f) * nt)<0) ? ((p + LENGTH_D(i_f) * nt)%NGRID1_D + NGRID1_D) : ((p + LENGTH_D(i_f) * nt)%NGRID1_D))
//#define p_step_index_D(p_step) ((p_step<0) ? (p_step%NGRID1_D + NGRID1_D) : (p_step%NGRID1_D))

//#define p_step_index_D(p_step) (((p_step)%(NGRID1_D) + (NGRID1_D)) % (NGRID1_D))
#define p_step_index_D(p_step) (p_step)
#endif