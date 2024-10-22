#ifndef MemAllocate_GPU_CUH
#define MemAllocate_GPU_CUH

/* copy constant data to GPU */
void copyConstantData();
/* initialization basic - CUDA */
void initialization_GPU();

// ************* fluid flow related memory allocate/deallocate ******************************
void MemAllocate_fluid_GPU(int flag);
//************* particle boundary related memory allocate/deallocate in gpu******************************
void MemAllocate_particle_GPU(int flag);

// Transfer information from the GPU to the CPU (DeviceToHost)
void copy_old_fluid_DeviceToHost();

#endif