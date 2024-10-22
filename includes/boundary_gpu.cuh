// #ifndef BOUNDARY_GPU_CUH
// #define BOUNDARY_GPU_CUH

// #include "Global_Variables_extern_gpu.cuh"

// #endif
/* boundary condition */
#define BEFORE_COLLSION (1)
#define AFTER_COLLSION (2)
/* Select Non-equilibrium extrapolation method: BEFORE_COLLSION / AFTER_COLLSION */
#define GUO_EXTRAPOLATION (AFTER_COLLSION)

#define NONE_NRBC (0) 
#define ABC (1)
// Select FLAG_NRBC method: NONE_NRBC / ABC  (which means none NRBC / absorbing boundary conditions)
#define FLAG_NRBC (NONE_NRBC) 