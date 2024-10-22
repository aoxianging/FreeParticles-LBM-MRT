#include "index_cpu.h"
//#include "boundary_gpu.cuh"
#include "externLib.h"
#include "solver_precision.h"
#include "externLib_CUDA.cuh"
#include "main_iteration_gpu.cuh"
#include "Fluid_field_extern.h"
#include "Module_extern.h"
//#include "utils.h"
#include "utils_gpu.cuh"
//#include "Global_Variables_extern_gpu.cuh"

#include "MemAllocate_gpu.cuh"



/* initialization basic - CUDA */
void initialization_GPU() {
    cout << "***************************** GPU Specifications **********************************" << endl;
    const int kb = 1024;
    const int mb = kb * kb;
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    cout << 0 << ": " << props.name << ": " << props.major << "." << props.minor << endl;
    cout << "  Global memory:   " << props.totalGlobalMem / mb << " mb" << endl;
    cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << " kb" << endl;
    cout << "  Constant memory: " << props.totalConstMem / kb << " kb" << endl;
    cout << "  Block registers: " << props.regsPerBlock << endl;
    cout << "  Number of SMs: " << props.multiProcessorCount << endl;
    cout << "  Clock frequencey: " << props.clockRate / 1e3 << " MHz" << endl;
    cout << "  Warp size:         " << props.warpSize << endl;
    cout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
    cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << endl;
    cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << endl;
    cout << "***************************** MemAllocate GPU **********************************" << endl;

    //************* fluid flow related memory allocate/deallocate ******************************
    MemAllocate_fluid_GPU(1);    
    //************* particle boundary related memory allocate/deallocate in gpu******************************
    MemAllocate_particle_GPU(1);

    cout << "Estimate the total amount of global memory needed on GPU (GB) = " << (memory_gpu / double(1024 * 1024 * 1024)) << endl;
}

// ************* fluid flow related memory allocate/deallocate ******************************
void MemAllocate_fluid_GPU(int flag) {
    int FLAG = flag;
    if (FLAG == 1) {
        cudaErrorCheck(cudaMalloc(&Velocity_ux_gpu, mem_cells_s1_TP)); memory_gpu += mem_cells_s1_TP; 
        cudaErrorCheck(cudaMemcpy(Velocity_ux_gpu, Velocity_ux, mem_cells_s1_TP, cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&Velocity_uy_gpu, mem_cells_s1_TP)); memory_gpu += mem_cells_s1_TP; 
        cudaErrorCheck(cudaMemcpy(Velocity_uy_gpu, Velocity_uy, mem_cells_s1_TP, cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&Velocity_uz_gpu, mem_cells_s1_TP)); memory_gpu += mem_cells_s1_TP; 
        cudaErrorCheck(cudaMemcpy(Velocity_uz_gpu, Velocity_uz, mem_cells_s1_TP, cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&Density_rho_gpu, mem_cells_s1_TP)); memory_gpu += mem_cells_s1_TP; 
        cudaErrorCheck(cudaMemcpy(Density_rho_gpu, Density_rho, mem_cells_s1_TP, cudaMemcpyHostToDevice));

        size_t width = NGRID1;    //row  //grid number
        size_t height = NDIR;    //column //number of discrete velocities
        //size_t pitch;

        cudaErrorCheck(cudaMallocPitch((void **)&pdf_gpu, &pitch, sizeof(T_P)*width, height));
        cudaErrorCheck(cudaMemcpy2D(pdf_gpu, pitch, pdf, sizeof(T_P)*width, sizeof(T_P)*width, height, cudaMemcpyHostToDevice));
        memory_gpu = memory_gpu + sizeof(T_P)*width * height;
        cudaErrorCheck(cudaMallocPitch((void **)&pdf_old_gpu, &pitch_old, sizeof(T_P)*width, height));
        cudaErrorCheck(cudaMemcpy2D(pdf_old_gpu, pitch_old, pdf, sizeof(T_P)*width, sizeof(T_P)*width, height, cudaMemcpyHostToDevice));
        memory_gpu = memory_gpu + sizeof(T_P)*width * height;

        cudaErrorCheck(cudaMalloc(&Boundary_xz0_gpu, XZG0*5*sizeof(T_P))); memory_gpu += (XZG0*5*sizeof(T_P)); 
        cudaErrorCheck(cudaMemcpy(Boundary_xz0_gpu, Boundary_xz0, (XZG0*5*sizeof(T_P)), cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&Boundary_xz1_gpu, (XZG0*5*sizeof(T_P)))); memory_gpu += (XZG0*5*sizeof(T_P)); 
        cudaErrorCheck(cudaMemcpy(Boundary_xz1_gpu, Boundary_xz1, (XZG0*5*sizeof(T_P)), cudaMemcpyHostToDevice));        
        cudaErrorCheck(cudaMalloc(&Boundary_xy0_gpu, XYG0*5*sizeof(T_P))); memory_gpu += (XYG0*5*sizeof(T_P)); 
        cudaErrorCheck(cudaMemcpy(Boundary_xy0_gpu, Boundary_xy0, (XYG0*5*sizeof(T_P)), cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&Boundary_xy1_gpu, (XYG0*5*sizeof(T_P)))); memory_gpu += (XYG0*5*sizeof(T_P)); 
        cudaErrorCheck(cudaMemcpy(Boundary_xy1_gpu, Boundary_xy1, (XYG0*5*sizeof(T_P)), cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&Boundary_yz0_gpu, YZG0*5*sizeof(T_P))); memory_gpu += (YZG0*5*sizeof(T_P)); 
        cudaErrorCheck(cudaMemcpy(Boundary_yz0_gpu, Boundary_yz0, (YZG0*5*sizeof(T_P)), cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&Boundary_yz1_gpu, (YZG0*5*sizeof(T_P)))); memory_gpu += (YZG0*5*sizeof(T_P)); 
        cudaErrorCheck(cudaMemcpy(Boundary_yz1_gpu, Boundary_yz1, (YZG0*5*sizeof(T_P)), cudaMemcpyHostToDevice));
    }
    else {
        cudaErrorCheck(cudaFree(Velocity_ux_gpu));
        cudaErrorCheck(cudaFree(Velocity_uy_gpu));
        cudaErrorCheck(cudaFree(Velocity_uz_gpu));
        cudaErrorCheck(cudaFree(Density_rho_gpu));
                
        cudaErrorCheck(cudaFree(pdf_gpu));
        cudaErrorCheck(cudaFree(pdf_old_gpu));

        cudaErrorCheck(cudaFree(Boundary_xz0_gpu));
        cudaErrorCheck(cudaFree(Boundary_xz1_gpu));
        cudaErrorCheck(cudaFree(Boundary_xy0_gpu));
        cudaErrorCheck(cudaFree(Boundary_xy1_gpu));
        cudaErrorCheck(cudaFree(Boundary_yz0_gpu));
        cudaErrorCheck(cudaFree(Boundary_yz1_gpu));
    }
}

//************* particle boundary related memory allocate/deallocate in gpu******************************
void MemAllocate_particle_GPU(int flag) {
    int FLAG = flag;
    if (FLAG == 1) {
        for (int i_p=0; i_p<Num_Sphere; i_p++){
            cudaErrorCheck(cudaMalloc(&spheres_gpu[i_p].Coords, mem_size_particle_3D_TP)); memory_gpu += mem_size_particle_3D_TP; 
            cudaErrorCheck(cudaMemcpy(spheres_gpu[i_p].Coords, spheres[i_p].Coords, mem_size_particle_3D_TP, cudaMemcpyHostToDevice));
            cudaErrorCheck(cudaMalloc(&spheres_gpu[i_p].Velos, mem_size_particle_3D_TP)); memory_gpu += mem_size_particle_3D_TP; 
            cudaErrorCheck(cudaMemcpy(spheres_gpu[i_p].Velos, spheres[i_p].Velos, mem_size_particle_3D_TP, cudaMemcpyHostToDevice));
            cudaErrorCheck(cudaMalloc(&spheres_gpu[i_p].AngulVelos, mem_size_particle_3D_TP)); memory_gpu += mem_size_particle_3D_TP; 
            cudaErrorCheck(cudaMemcpy(spheres_gpu[i_p].AngulVelos, spheres[i_p].AngulVelos, mem_size_particle_3D_TP, cudaMemcpyHostToDevice));

            cudaErrorCheck(cudaMalloc(&Particles_gpu[i_p].BC_I, mem_particle_BC_max_long)); memory_gpu += mem_particle_BC_max_long; 
            cudaErrorCheck(cudaMemcpy(Particles_gpu[i_p].BC_I, Particles[i_p].BC_I, mem_particle_BC_max_long, cudaMemcpyHostToDevice));
            cudaErrorCheck(cudaMalloc(&Particles_gpu[i_p].BC_DV, mem_particle_BC_max_int)); memory_gpu += mem_particle_BC_max_int; 
            cudaErrorCheck(cudaMemcpy(Particles_gpu[i_p].BC_DV, Particles[i_p].BC_DV, mem_particle_BC_max_int, cudaMemcpyHostToDevice));
            cudaErrorCheck(cudaMalloc(&Particles_gpu[i_p].BC_q, mem_particle_BC_max_TP)); memory_gpu += mem_particle_BC_max_TP; 
            cudaErrorCheck(cudaMemcpy(Particles_gpu[i_p].BC_q, Particles[i_p].BC_q, mem_particle_BC_max_TP, cudaMemcpyHostToDevice));
            // The set of distribution functions passed to the particle
            cudaErrorCheck(cudaMalloc(&Particles_gpu[i_p].BC_fOld, mem_particle_BC_max_TP)); memory_gpu += mem_particle_BC_max_TP; 
            cudaErrorCheck(cudaMemcpy(Particles_gpu[i_p].BC_fOld, Particles[i_p].BC_fOld, mem_particle_BC_max_TP, cudaMemcpyHostToDevice));
            // Index of the new fluid point due to the particle boundary moving
            cudaErrorCheck(cudaMalloc(&Particles_gpu[i_p].Refill_Point_I, mem_refill_max_long)); memory_gpu += mem_refill_max_long; 
            cudaErrorCheck(cudaMemcpy(Particles_gpu[i_p].Refill_Point_I, Particles[i_p].Refill_Point_I, mem_refill_max_long, cudaMemcpyHostToDevice));
            // Index of the direction of discrete velocity for Refill process
            cudaErrorCheck(cudaMalloc(&Particles_gpu[i_p].Refill_Point_DV, mem_refill_max_int)); memory_gpu += mem_refill_max_int; 
            cudaErrorCheck(cudaMemcpy(Particles_gpu[i_p].Refill_Point_DV, Particles[i_p].Refill_Point_DV, mem_refill_max_int, cudaMemcpyHostToDevice));
        }
    }
    else {
        for (int i_p=0; i_p<Num_Sphere; i_p++){
            cudaErrorCheck(cudaFree(spheres_gpu[i_p].Coords));
            cudaErrorCheck(cudaFree(spheres_gpu[i_p].Velos));
            cudaErrorCheck(cudaFree(spheres_gpu[i_p].AngulVelos));

            cudaErrorCheck(cudaFree(Particles_gpu[i_p].BC_I));
            cudaErrorCheck(cudaFree(Particles_gpu[i_p].BC_DV));
            cudaErrorCheck(cudaFree(Particles_gpu[i_p].BC_q));
            cudaErrorCheck(cudaFree(Particles_gpu[i_p].BC_fOld));
            cudaErrorCheck(cudaFree(Particles_gpu[i_p].Refill_Point_I));
            cudaErrorCheck(cudaFree(Particles_gpu[i_p].Refill_Point_DV));
        }
    }
}
// Transfer information from the GPU to the CPU (DeviceToHost)
void copy_old_fluid_DeviceToHost() {
    size_t width = NGRID1;    //row  //grid number
    size_t height = NDIR;    //column //number of discrete velocities
    cudaErrorCheck(cudaMemcpy2D(pdf, sizeof(T_P)*width, pdf_gpu, pitch, sizeof(T_P)*width, height, cudaMemcpyDeviceToHost));    
    cudaErrorCheck(cudaMemcpy(Boundary_xz0, Boundary_xz0_gpu, (XZG0*5*sizeof(T_P)), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(Boundary_xz1, Boundary_xz1_gpu, (XZG0*5*sizeof(T_P)), cudaMemcpyDeviceToHost)); 
    cudaErrorCheck(cudaMemcpy(Boundary_xy0, Boundary_xy0_gpu, (XYG0*5*sizeof(T_P)), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(Boundary_xy1, Boundary_xy1_gpu, (XYG0*5*sizeof(T_P)), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(Boundary_yz0, Boundary_yz0_gpu, (YZG0*5*sizeof(T_P)), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(Boundary_yz1, Boundary_yz1_gpu, (YZG0*5*sizeof(T_P)), cudaMemcpyDeviceToHost));    
    // particles info
    for (int i_p=0; i_p<Num_Sphere; i_p++){
        cudaErrorCheck(cudaMemcpy(spheres[i_p].Coords, spheres_gpu[i_p].Coords, mem_size_particle_3D_TP, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(spheres[i_p].Velos, spheres_gpu[i_p].Velos, mem_size_particle_3D_TP, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(spheres[i_p].AngulVelos, spheres_gpu[i_p].AngulVelos, mem_size_particle_3D_TP, cudaMemcpyDeviceToHost));

        cudaErrorCheck(cudaMemcpy(Particles[i_p].BC_I, Particles_gpu[i_p].BC_I, mem_particle_BC_max_long, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(Particles[i_p].BC_DV, Particles_gpu[i_p].BC_DV, mem_particle_BC_max_int, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(Particles[i_p].BC_q, Particles_gpu[i_p].BC_q, mem_particle_BC_max_TP, cudaMemcpyDeviceToHost));        
        cudaErrorCheck(cudaMemcpy(Particles[i_p].BC_fOld, Particles_gpu[i_p].BC_fOld, mem_particle_BC_max_TP, cudaMemcpyDeviceToHost));
        // Index of the new fluid point due to the particle boundary moving    
        cudaErrorCheck(cudaMemcpy(Particles[i_p].Refill_Point_I, Particles_gpu[i_p].Refill_Point_I, mem_refill_max_long, cudaMemcpyDeviceToHost));
        // Index of the direction of discrete velocity for Refill process
        cudaErrorCheck(cudaMemcpy(Particles[i_p].Refill_Point_DV, Particles_gpu[i_p].Refill_Point_DV, mem_refill_max_int, cudaMemcpyDeviceToHost));
    }
    
    
}