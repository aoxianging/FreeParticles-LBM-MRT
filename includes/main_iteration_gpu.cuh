#ifndef main_iteration_GPU_CUH
#define main_iteration_GPU_CUH

//#include "Global_Variables_extern_gpu.cuh"


//void main_iteration_kernel_GPU();

/* copy constant data to GPU */
void copyConstantData();


void main_iteration_kernel_GPU();

void get_macro_vars_cpu();

#endif