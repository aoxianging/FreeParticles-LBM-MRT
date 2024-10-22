#include "index_gpu.cuh"
#include "index_cpu.h"
#include "externLib.h"
#include "solver_precision.h"
#include "externLib_CUDA.cuh"
#include "main_iteration_gpu.cuh"
#include "Fluid_field_extern.h"
#include "Module_extern.h"
#include "utils.h"
#include "utils_gpu.cuh"
#include "Misc.h"
//#include "Global_Variables_extern_gpu.cuh"
#include "Global_Variables_gpu.cuh"

#include "boundary_gpu.cuh"
void updata_information_moving_particle(int i_Nsphe);

/* copy constant data to GPU */
void copyConstantData() {
    cudaErrorCheck(cudaMemcpyToSymbol(PI_gpu, &PI, sizeof(T_P)));  //pi
    
    cudaErrorCheck(cudaMemcpyToSymbol(dx_LBM_gpu, &dx_LBM, sizeof(T_P)));  //the size of grid
    cudaErrorCheck(cudaMemcpyToSymbol(dt_LBM_gpu, &dt_LBM, sizeof(T_P)));  //the size of time step
    cudaErrorCheck(cudaMemcpyToSymbol(c_LBM_gpu, &c_LBM, sizeof(T_P)));  //c_LBM = dx_LBM/dt_LBM
    cudaErrorCheck(cudaMemcpyToSymbol(overc_LBM_gpu, &overc_LBM, sizeof(T_P)));  //overc_LBM = prc(1.)/c_LBM
    cudaErrorCheck(cudaMemcpyToSymbol(w_equ_gpu, w_equ, NDIR * sizeof(T_P)));
    
    cudaErrorCheck(cudaMemcpyToSymbol(ex_gpu, ex, NDIR * sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(ey_gpu, ey, NDIR * sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(ez_gpu, ez, NDIR * sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(nxGlobal_gpu, &nxGlobal, sizeof(I_INT)));
    cudaErrorCheck(cudaMemcpyToSymbol(nyGlobal_gpu, &nyGlobal, sizeof(I_INT)));
    cudaErrorCheck(cudaMemcpyToSymbol(nzGlobal_gpu, &nzGlobal, sizeof(I_INT)));
    cudaErrorCheck(cudaMemcpyToSymbol(nAbsorbingL_gpu, &nAbsorbingL, sizeof(I_INT)));
    I_INT temp;
    temp = (nxGlobal + 2);  cudaErrorCheck(cudaMemcpyToSymbol(NXG1_D, &temp, sizeof(I_INT)));
    temp = (nyGlobal + 2);  cudaErrorCheck(cudaMemcpyToSymbol(NYG1_D, &temp, sizeof(I_INT)));
    temp = (nzGlobal + 2);  cudaErrorCheck(cudaMemcpyToSymbol(NZG1_D, &temp, sizeof(I_INT)));
    temp = (NXG1 * NYG1 * NZG1);  cudaErrorCheck(cudaMemcpyToSymbol(NGRID1_D, &temp, sizeof(I_INT)));
    temp = (NXG0 * NYG0 * NZG0);  cudaErrorCheck(cudaMemcpyToSymbol(NGRID0_D, &temp, sizeof(I_INT)));
    temp = (NXG0*NYG0);  cudaErrorCheck(cudaMemcpyToSymbol(XYG0_D, &temp, sizeof(I_INT)));
    temp = (NXG0*NZG0);  cudaErrorCheck(cudaMemcpyToSymbol(XZG0_D, &temp, sizeof(I_INT)));
    temp = (NYG0*NZG0);  cudaErrorCheck(cudaMemcpyToSymbol(YZG0_D, &temp, sizeof(I_INT)));
    temp = (NXG1*NYG1);  cudaErrorCheck(cudaMemcpyToSymbol(XYG1_D, &temp, sizeof(I_INT)));
    temp = (NXG1*NZG1);  cudaErrorCheck(cudaMemcpyToSymbol(XZG1_D, &temp, sizeof(I_INT)));
    temp = (NYG1*NZG1);  cudaErrorCheck(cudaMemcpyToSymbol(YZG1_D, &temp, sizeof(I_INT)));
    
    cudaErrorCheck(cudaMemcpyToSymbol(Length_gpu, Length, NDIR * sizeof(I_INT)));
    cudaErrorCheck(cudaMemcpyToSymbol(Reverse_gpu, Reverse, NDIR * sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(BB_xy_top_gpu, BB_xy_top, 5* sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(BB_yz_front_gpu, BB_yz_front, 5 * sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(BB_xz_Right_gpu, BB_xz_Right, 5 * sizeof(int)));

    cudaErrorCheck(cudaMemcpyToSymbol(Lid_velocity_gpu, &Lid_velocity, sizeof(T_P) ));
    cudaErrorCheck(cudaMemcpyToSymbol(Frequency_lbm_gpu, &Frequency_lbm, sizeof(T_P) ));
    cudaErrorCheck(cudaMemcpyToSymbol(AnguFreq_lbm_gpu, &AnguFreq_lbm, sizeof(T_P) ));
    cudaErrorCheck(cudaMemcpyToSymbol(Velocity_Bound_gpu, &Lid_velocity, sizeof(T_P) ));    
    cudaErrorCheck(cudaMemcpyToSymbol(Density_gpu, &Density, sizeof(T_P) ));
    cudaErrorCheck(cudaMemcpyToSymbol(SRT_OverTau_gpu, &SRT_OverTau, sizeof(T_P) ));
    
    cudaErrorCheck(cudaMemcpyToSymbol(body_accelerate_gpu, body_accelerate, 3*sizeof(T_P) ));

    T_P temp_r = (T_P)(spheres[0].GridNum_D)/prc(2.) ; cudaErrorCheck(cudaMemcpyToSymbol(Sphere_radius_gpu, &temp_r, sizeof(T_P) ));
    //parameters for MRT
    cudaErrorCheck(cudaMemcpyToSymbol(MRT_Trans_M_gpu, MRT_Trans_M, NDIR*NDIR* sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(MRT_Trans_M_inverse_gpu, MRT_Trans_M_inverse, NDIR*NDIR * sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(MRT_Trans_M_inverse_int_gpu, MRT_Trans_M_inverse_int, NDIR*NDIR * sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(MRT_C_inverse_int_gpu, &MRT_C_inverse_int, sizeof(T_P)));    
    cudaErrorCheck(cudaMemcpyToSymbol(MRT_Collision_M_gpu, MRT_Collision_M, NDIR*NDIR * sizeof(T_P)));    
    cudaErrorCheck(cudaMemcpyToSymbol(MRT_S_gpu, MRT_S, NDIR * sizeof(T_P)));
}

//#pragma region (kernel_fluidphase)
//=====================================================================================================================================
//----------------------OSI kernel----------------------
// complete streaming steps use one-step index (OSI) algorithm from Ma et al., 2023 
//=====================================================================================================================================
__global__ void kernel_OSI_GPU(I_INT imax, T_P* pdf_old_gpu, T_P* pdf_gpu, I_INT pitch_old, I_INT pitch, int ntime) {

    // Indexing (Thread) Represents the position in the grid that needs to be solved
    I_INT i_Thr = blockIdx.x * blockDim.x + threadIdx.x ;

    T_P *p_f;   //the point to rowHead(fi in first grid)
    T_P f[NDIR], feq[NDIR];    
    T_P rho, ux, uy, uz;
    T_P edotu, udotu;

    T_P  delta_ABC = 0, temp_ABC = 0;
    T_P sigma_ABC = 0.;
    T_P feq_w[NDIR];

    if (i_Thr < imax ) {
        //++++++++ + OSI pull step++++++++++++
        I_INT i_z = i_Thr/XYG0_D + 1;
        I_INT i_y = i_Thr%XYG0_D / NXG0_D + 1;
        I_INT i_x = i_Thr%XYG0_D % NXG0_D + 1;
        I_INT i_Golbal = p_index_D(i_x,i_y,i_z);  //the index of point in the domain with one ghost layer
        
        sigma_ABC = 0;
        rho = Density_gpu;
        ux=0.; uz=0.; 
        uy = Velocity_Bound_gpu;
        udotu = ux * ux + uy * uy + uz * uz;
        #pragma unroll
        for (int i_f=0; i_f<NDIR; i_f++){
            edotu = ((T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz);          
            feq_w[i_f]= rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);
        }
        #if (FLAG_NRBC == ABC)
        if (i_x <= nAbsorbingL_gpu){
            delta_ABC = nAbsorbingL_gpu - i_x + prc(0.5);            
        }else{
            delta_ABC = nAbsorbingL_gpu - (1 + NXG0_D - i_x) + prc(0.5);            
        }
        if (i_z <= nAbsorbingL_gpu){
            delta_ABC = nAbsorbingL_gpu - i_z + prc(0.5);            
        }else{
            delta_ABC = nAbsorbingL_gpu - (1 + NZG0_D - i_z) + prc(0.5);            
        }
        // if (i_y <= nAbsorbingL_gpu){
        //     delta_ABC = nAbsorbingL_gpu - i_y + prc(0.5);            
        // }else{
        //     delta_ABC = nAbsorbingL_gpu - (1 + NYG0_D - i_y) + prc(0.5);            
        // }        
        if (delta_ABC > 0 ) {
            T_P AS = 0.3;
            sigma_ABC = AS*pow(((T_P)delta_ABC/nAbsorbingL_gpu),2);
            // T_P AS = 12.207;
            // sigma_ABC = AS*(nAbsorbingL_gpu - delta_ABC)*pow(((T_P)delta_ABC/nAbsorbingL_gpu),4)/(nAbsorbingL_gpu);
            #pragma unroll
            for (int i_f=0; i_f<NDIR; i_f++){
                // feq_w[i_f]= Density_gpu * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*(T_P)ey_gpu[i_f]*Lid_velocity_gpu + prc(4.5)*(T_P)ey_gpu[i_f]*Lid_velocity_gpu*(T_P)ey_gpu[i_f]*Lid_velocity_gpu - prc(1.5)* Lid_velocity_gpu*Lid_velocity_gpu);
                feq_w[i_f]= Density_gpu * w_equ_gpu[i_f] * (prc(1.));
            }                
        }
        //printf("error");
        #endif
        
        rho = 0;
        ux = 0; uy = 0; uz = 0;
        #pragma unroll
        for (int i_f=0; i_f<NDIR; i_f++){
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch_old);            
            f[i_f] = p_f[i_Golbal];
            rho = rho + f[i_f];
            ux = ux + f[i_f]* (T_P)ex_gpu[i_f];
            uy = uy + f[i_f]* (T_P)ey_gpu[i_f];
            uz = uz + f[i_f]* (T_P)ez_gpu[i_f];
        }
        ux = (ux/rho + prc(0.5)*body_accelerate_gpu[0] );
        uy = (uy/rho + prc(0.5)*body_accelerate_gpu[1] );
        uz = (uz/rho + prc(0.5)*body_accelerate_gpu[2] );         
        // // // SRT        
        // udotu = ux * ux + uy * uy + uz * uz;
        // #pragma unroll
        // for (int i_f=0; i_f<NDIR; i_f++){
        //     edotu = ((T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz);
        //     feq[i_f]= rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);
        //     p_f = (T_P*)((char*)pdf_gpu + i_f * pitch);
        //     p_f[i_Golbal + Length_gpu[i_f]] = f[i_f] - SRT_OverTau_gpu * (f[i_f] - feq[i_f]) + sigma_ABC  * (feq_w[i_f] - feq[i_f]) ;   //SRT_OverTau = 1/tau            
        // }
        // MRT
        T_P fnew[NDIR];
        T_P temp_F[NDIR];
        T_P MRT_Force_gpu[NDIR];
        MRT_Force_gpu[0] = prc(0.); MRT_Force_gpu[2] = prc(0.); MRT_Force_gpu[4] = prc(0.); MRT_Force_gpu[6] = prc(0.); MRT_Force_gpu[8] = prc(0.); MRT_Force_gpu[10] = prc(0.);
        MRT_Force_gpu[12] = prc(0.); MRT_Force_gpu[16] = prc(0.); MRT_Force_gpu[17] = prc(0.); MRT_Force_gpu[18] = prc(0.);
        MRT_Force_gpu[1] = prc(19.)*(prc(2.)- MRT_S_gpu[1])*(ux*body_accelerate_gpu[0]+uy*body_accelerate_gpu[1]+uz*body_accelerate_gpu[2]);
        MRT_Force_gpu[3] = (prc(1.)- prc(0.5)*MRT_S_gpu[3])*body_accelerate_gpu[0]; MRT_Force_gpu[5] = (prc(1.)- prc(0.5)*MRT_S_gpu[5])*body_accelerate_gpu[1]; MRT_Force_gpu[7] = (prc(1.)- prc(0.5)*MRT_S_gpu[7])*body_accelerate_gpu[2];
        MRT_Force_gpu[9] = (prc(2.)- MRT_S_gpu[9]) * (prc(2.)*ux*body_accelerate_gpu[0] - uy*body_accelerate_gpu[1] - uz*body_accelerate_gpu[2]);
        MRT_Force_gpu[11] = (prc(2.)- MRT_S_gpu[11]) * (uy*body_accelerate_gpu[1] - uz*body_accelerate_gpu[2]);
        MRT_Force_gpu[13] = (prc(1.)- prc(0.5)*MRT_S_gpu[13]) * (ux*body_accelerate_gpu[1] + uy*body_accelerate_gpu[0]); MRT_Force_gpu[14] = (prc(1.)- prc(0.5)*MRT_S_gpu[14]) * (uy*body_accelerate_gpu[2] + uz*body_accelerate_gpu[1]); MRT_Force_gpu[15] = (prc(1.)- prc(0.5)*MRT_S_gpu[15]) * (ux*body_accelerate_gpu[2] + uz*body_accelerate_gpu[0]);
        udotu = ux * ux + uy * uy + uz * uz;
        #pragma unroll
        for (int i_f=0; i_f<NDIR; i_f++){
            edotu = (T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz;
            feq[i_f]= rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);
        }
        #pragma unroll
        for (int i_f=0; i_f<NDIR; i_f++){
            fnew[i_f] = prc(0.);
            temp_F[i_f] = prc(0.);
            #pragma unroll
            for (int i_m=0; i_m<NDIR; i_m++){
                fnew[i_f] = fnew[i_f] + MRT_Collision_M_gpu[i_f*NDIR + i_m] * (f[i_m]-feq[i_m]);
                temp_F[i_f] = temp_F[i_f] + MRT_Trans_M_inverse_gpu[i_f*NDIR + i_m] * MRT_Force_gpu[i_m];
            }            
            temp_ABC = sigma_ABC * (feq_w[i_f] - feq[i_f]);
            p_f = (T_P*)((char*)pdf_gpu + i_f * pitch);
            p_f[i_Golbal + Length_gpu[i_f]] = f[i_f] - fnew[i_f] + rho * temp_F[i_f] + temp_ABC;            
        }

    }
}
//=====================================================================================================================================
//----------------------point_source----------------------
// add point source in the given point of compuite domain with different method: BB: The distribution function of the point sound source near the grid point is obtained by bounce back; equi: The distribution function of the point sound source near the grid point is obtained by Equilibrium distribution function
//=====================================================================================================================================
__global__ void acoustic_point_source_BB(I_INT imax, T_P* pdf_old_gpu, I_INT pitch_old, int ntime) {
    // Indexing (Thread) Represents the position in the grid that needs to be solved
    I_INT i_Thr = blockIdx.x * blockDim.x + threadIdx.x ;

    T_P *p_f;   //the point to rowHead(fi in first grid)
    T_P f[NDIR];
    // T_P feq[NDIR];
    // T_P edotu, udotu;    
    T_P rho;
    T_P u_sound;
    
    if (i_Thr < imax ) {
        //++++++++ + OSI pull step++++++++++++
        I_INT i_z = NXG0_D/2 + 1;
        I_INT i_y = NYG0_D/2 + 1;
        I_INT i_x = NZG0_D/2 + 1;
        I_INT i_Golbal = p_index_D(i_x,i_y,i_z);  //the index of point in the domain with one ghost layer
        
        // // point source 
        
        u_sound = Velocity_Bound_gpu;
        rho = 0.;//Density_gpu * (1. + sqrt(3.)*Velocity_Bound_gpu);
        #pragma unroll
        for (int i_f=0; i_f<NDIR; i_f++){
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch_old);            
            f[i_f] = p_f[i_Golbal];
            rho = rho + f[i_f];
        }

        //udotu = ux * ux + uy * uy + uz * uz;
        #pragma unroll
        for (int i_f=0; i_f<NDIR; i_f++){
            //edotu = ((T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz);
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch_old);
            p_f[i_Golbal + Length_gpu[i_f]] = f[ Reverse_gpu[i_f] ] + prc(6.) * w_equ_gpu[i_f] * rho * u_sound;  //- prc(6.) * Density_gpu * w_equ_gpu[i_f] * prc(0.5)*body_accelerate_gpu[2] * (T_P)ez_gpu[i_f];
        }
    }
}
__global__ void acoustic_point_source_equi(I_INT imax, T_P* pdf_old_gpu, I_INT pitch_old, int ntime) {
    // Indexing (Thread) Represents the position in the grid that needs to be solved
    I_INT i_Thr = blockIdx.x * blockDim.x + threadIdx.x ;

    T_P *p_f;   //the point to rowHead(fi in first grid)
    T_P feq[NDIR];    
    T_P rho, ux, uy, uz;
    T_P edotu, udotu;

    if (i_Thr < imax ) {
        //++++++++ + OSI pull step++++++++++++
        I_INT i_z = NXG0_D/2 + 1;
        I_INT i_y = NYG0_D/2 + 1;
        I_INT i_x = NZG0_D/2 + 1;
        I_INT i_Golbal = p_index_D(i_x,i_y,i_z);  //the index of point in the domain with one ghost layer
        
        // // point source 
        rho = Density_gpu * (1. + sqrt(3.)*Velocity_Bound_gpu);//Density_gpu * (1. + Lid_velocity_gpu*sqrt(3.)*sin(AnguFreq_lbm_gpu * ntime * dt_LBM_gpu));//rhobb;
        ux=0; uz=0; uy = 0;
   
        udotu = ux * ux + uy * uy + uz * uz;
        #pragma unroll
        for (int i_f=0; i_f<NDIR; i_f++){
            edotu = ((T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz);
            feq[i_f]= rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch_old);
            p_f[i_Golbal + Length_gpu[i_f]] = feq[i_f];  //f[i_f] - prc(6.) * Density_gpu * w_equ_gpu[i_f] * prc(0.5)*body_accelerate_gpu[2] * (T_P)ez_gpu[i_f];        
        }
    }
}

//#pragma endregion (kernel_fluidphase)

//================================================================================================================================================================= =
//----------------------off-wall bounce back for wall boundary----------------------
//currently, only the domain boundary is no-slip wall, there is no particle in the fluid domain
//for one-step index (OSI) algorithm
//================================================================================================================================================================= =
__global__ void wall_BB_BC_GPU_face_xy(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime) {
    int i_f ;  //column //index of discrete velocities
    //int i_grid ;    //row  //index of grid position coordinates
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    I_INT i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT ix, iy, iz;
    I_INT i_Global;//the index of point in the global distribution function array
    //Represents the position in the grid that needs to be solved
    //int position;
    T_P f[NDIR];
    if (i < imax ) {
        //z direction (top wall)
        ix = i % (NXG0_D) + 1;
        iy = i / (NXG0_D) + 1;
        iz = NZG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = BB_xy_top_gpu[i3];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            f[i_f] = p_f[p_step_index_D((i_Global + Length_gpu[i_f]) )]; 
            p_f = (T_P*)((char*)pdf_old_gpu + Reverse_gpu[i_f] * pitch);
            p_f[p_step_index_D(i_Global )] = f[i_f] - prc(6.)* w_equ_gpu[i_f]* Density_gpu * Lid_velocity_gpu * (T_P)ey_gpu[i_f];// - prc(6.) * Density_gpu * w_equ_gpu[i_f] * prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] * (T_P)ez_gpu[i_f];
        }
        //__syncthreads();
        //z direction (bottom wall)
        iz = 1;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = Reverse_gpu[BB_xy_top_gpu[i3]];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            f[i_f] = p_f[p_step_index_D((i_Global + Length_gpu[i_f]) )];
            p_f = (T_P*)((char*)pdf_old_gpu + Reverse_gpu[i_f] * pitch);
            p_f[p_step_index_D(i_Global )] = f[i_f];//- prc(6.) * Density_gpu * w_equ_gpu[i_f] * prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] * (T_P)ez_gpu[i_f];
        }
    }
}
__global__ void wall_BB_BC_GPU_face_yz(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime) {

    int i_f ;  //column //index of discrete velocities
    //int i_grid ;    //row  //index of grid position coordinates
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    I_INT  i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT  ix, iy, iz;
    I_INT i_Global;//the index of point in the global distribution function array
    //Represents the position in the grid that needs to be solved
    //int position; 
    T_P f[NDIR];
    if (i < imax ) {
        //x direction (front wall)
        iy = i % (NYG0_D) + 1;
        iz = i / (NYG0_D) + 1;
        ix = NXG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = BB_yz_front_gpu[i3];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            f[i_f] = p_f[p_step_index_D((i_Global + Length_gpu[i_f]) )];
            p_f = (T_P*)((char*)pdf_old_gpu + Reverse_gpu[i_f] * pitch);
            p_f[p_step_index_D(i_Global )] = f[i_f];//- prc(6.) * Density_gpu * w_equ_gpu[i_f] * prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] * (T_P)ez_gpu[i_f];
        }
        //x direction (back wall)        
        ix = 1;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = Reverse_gpu[BB_yz_front_gpu[i3]];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            f[i_f] = p_f[p_step_index_D((i_Global + Length_gpu[i_f]) )];
            p_f = (T_P*)((char*)pdf_old_gpu + Reverse_gpu[i_f] * pitch);
            p_f[p_step_index_D(i_Global )] = f[i_f];//- prc(6.) * Density_gpu * w_equ_gpu[i_f] * prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] * (T_P)ez_gpu[i_f];
        }
    }
}
__global__ void wall_BB_BC_GPU_face_xz(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime) {

    int i_f ;  //column //index of discrete velocities
    //int i_grid ;    //row  //index of grid position coordinates
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    I_INT  i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT  ix, iy, iz;
    I_INT i_Global;//the index of point in the global distribution function array
    //Represents the position in the grid that needs to be solved
    //int position; 
    T_P f[NDIR];
    if (i < imax ) {
        //y direction (right wall)
        ix = i % (NXG0_D) + 1;
        iz = i / (NXG0_D) + 1;
        iy = NYG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = BB_xz_Right_gpu[i3];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            f[i_f] = p_f[p_step_index_D((i_Global + Length_gpu[i_f]) )];
            p_f = (T_P*)((char*)pdf_old_gpu + Reverse_gpu[i_f] * pitch);
            p_f[p_step_index_D(i_Global )] = f[i_f];//- prc(6.) * Density_gpu * w_equ_gpu[i_f] * prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] * (T_P)ez_gpu[i_f];
        }
        //y direction (left wall)        
        iy = 1;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = Reverse_gpu[BB_xz_Right_gpu[i3]];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            f[i_f] = p_f[p_step_index_D((i_Global + Length_gpu[i_f]) )];
            p_f = (T_P*)((char*)pdf_old_gpu + Reverse_gpu[i_f] * pitch);
            p_f[p_step_index_D(i_Global )] = f[i_f];//- prc(6.) * Density_gpu * w_equ_gpu[i_f] * prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] * (T_P)ez_gpu[i_f];
        }
    }
}
//================================================================================================================================================================= =
//----------------------periodic boundary  ----------------------
//for one-step index (OSI) algorithm
//================================================================================================================================================================= =
__global__ void periodic_BC_GPU_face_xy(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime) {
    // Indexing (Thread) 
    I_INT i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT ix, iy, iz;
    I_INT i_Global, i_Global_peri; //the index of point in the global distribution function array
    int i_f;  //column //index of discrete velocities        
    T_P * p_f;   //the point to rowHead(fi in first grid)
    if (i < imax ) {
        //z direction (top wall)
        ix = i % (NXG0_D) + 1;
        iy = i / (NXG0_D) + 1;
        iz = NZG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        i_Global_peri = p_index_D(ix, iy, 0);
        //printf("___ %d , %d, %d, %d, %d___", int(BB_xy_top_gpu[0]), int(BB_xy_top_gpu[1]), int(BB_xy_top_gpu[2]), int(BB_xy_top_gpu[3]), int(BB_xy_top_gpu[4]));        
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = Reverse_gpu[BB_xy_top_gpu[i3]];            
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);            
            p_f[p_step_index_D(i_Global )] = p_f[p_step_index_D(i_Global_peri )];            
        }
        //__syncthreads();
        //z direction (bottom wall)
        iz = 1;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        i_Global_peri = p_index_D(ix, iy, NZG0_D+1);
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = BB_xy_top_gpu[i3];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);            
            p_f[p_step_index_D(i_Global )] = p_f[p_step_index_D(i_Global_peri )];           
        }
    }
}
__global__ void periodic_BC_GPU_face_yz(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime) {
    // Indexing (Thread) 
    I_INT i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT ix, iy, iz;    
    I_INT i_Global, i_Global_peri; //the index of point in the global distribution function array
    int i_f ;  //column //index of discrete velocities        
    T_P *p_f;   //the point to rowHead(fi in first grid)

    if (i < imax ) {        
        //x direction (front wall)
        iy = i % (NYG0_D) + 1;
        iz = i / (NYG0_D) + 1;
        ix = NXG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        i_Global_peri = p_index_D(0, iy, iz);
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = Reverse_gpu[BB_yz_front_gpu[i3]];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            p_f[p_step_index_D(i_Global )] = p_f[p_step_index_D(i_Global_peri )];
        }
        //x direction (back wall)        
        ix = 1;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        i_Global_peri = p_index_D(NXG0_D+1, iy, iz);
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = BB_yz_front_gpu[i3];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            p_f[p_step_index_D(i_Global )] = p_f[p_step_index_D(i_Global_peri )];
        }
    }
}
__global__ void periodic_BC_GPU_face_xz(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime) {    
    // Indexing (Thread) 
    I_INT  i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT  ix, iy, iz;
    I_INT i_Global, i_Global_peri; //the index of point in the global distribution function array
    int i_f ;  //column //index of discrete velocities
    T_P *p_f;   //the point to rowHead(fi in first grid)

    if (i < imax ) {        
        //y direction (Right wall)
        ix = i % (NXG0_D) + 1;
        iz = i / (NXG0_D) + 1;
        iy = NYG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        i_Global_peri = p_index_D(ix, 0, iz);
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = Reverse_gpu[BB_xz_Right_gpu[i3]];
            //__syncthreads();
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            p_f[p_step_index_D(i_Global )] = p_f[p_step_index_D(i_Global_peri )];
        }
        //x direction (Left wall)        
        iy = 1;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        i_Global_peri = p_index_D(ix, NYG0_D+1, iz);
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = BB_xz_Right_gpu[i3];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            p_f[p_step_index_D(i_Global )] = p_f[p_step_index_D(i_Global_peri )];
        }
    }
}
__global__ void periodic_BC_GPU_edge_y(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime) {

    int i_f ;  //column //index of discrete velocities
    //int i_grid ;    //row  //index of grid position coordinates
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int ix,iy,iz;

    //T_P f[NDIR];

    if (i < imax ) {        
        // (1,iy,1)
        iy = i + 1;
        ix = 1;
        iz = 1;
        i_f = 11;
        p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);            
        p_f[p_step_index_D(p_index_D(ix, iy, iz) )] = 
        p_f[p_step_index_D(p_index_D(NXG0_D+1, iy, NZG0_D+1) )];

        // (NXG0_D,iy,1)          
        iy = i + 1;
        ix = NXG0_D;
        iz = 1;
        i_f = 12;
        p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);            
        p_f[p_step_index_D(p_index_D(ix, iy, iz) )] = 
        p_f[p_step_index_D(p_index_D(0, iy, NZG0_D+1) )];

        // (1,iy,NZG0_D)          
        iy = i + 1;
        ix = 1;
        iz = NZG0_D;
        i_f = 13;
        p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);            
        p_f[p_step_index_D(p_index_D(ix, iy, iz) )] = 
        p_f[p_step_index_D(p_index_D(NXG0_D+1, iy, 0) )];

        // (NXG0_D,iy,NZG0_D)          
        iy = i + 1;
        ix = NXG0_D;
        iz = NZG0_D;
        i_f = 14;
        p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);            
        p_f[p_step_index_D(p_index_D(ix, iy, iz) )] = 
        p_f[p_step_index_D(p_index_D(0, iy, 0) )];        
    }
}
//================================================================================================================================================================= =
//----------------------Zhao-Li et al., 2002, Non-equilibrium extrapolation method for boundary condition for inlet/outlet boundary----------------------
//for one-step index (OSI) algorithm
//================================================================================================================================================================= =
__global__ void BC_Guo2002_GPU_face_xz_save(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime, T_P* Boundary_xz0_gpu, T_P* Boundary_xz1_gpu) { // Zhao-Li et al., 2002, Non-equilibrium extrapolation method
    int i_f ;  //column //index of discrete velocities
    //int i_grid ;    //row  //index of grid position coordinates
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    I_INT  i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT  ix, iy, iz;
    //Represents the position in the grid that needs to be solved
    //int position;    
    if (i < imax ) {
        //y direction (left : inlet)
        ix = i % (NXG0_D) + 1;
        iy = 1;
        iz = i / (NXG0_D) + 1;
        I_INT i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array

        T_P fbb[NDIR], rhobb, uxbb, uybb, uzbb;   //fluid node which neighbour of boundary node
        T_P feqbb, edotubb, udotubb;
        T_P rho = Density_gpu;
        T_P ux=0, uz=0, uy = 0.;    //boundary node
        T_P feqb, edotu, udotu;
        
        rhobb = 0; uxbb = 0; uybb = 0; uzbb = 0;
        #pragma unroll
        for (int i3=0; i3<NDIR; i3++ ){
            i_f = i3;
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            #if (GUO_EXTRAPOLATION == BEFORE_COLLSION)
            fbb[i_f] = p_f[p_step_index_D((i_Global ) )];
            #elif (GUO_EXTRAPOLATION == AFTER_COLLSION)
            fbb[i_f] = p_f[p_step_index_D((i_Global ) + Length_gpu[i_f] )];
            #endif  
            rhobb = rhobb + fbb[i_f];
            uxbb  = uxbb + fbb[i_f] * (T_P)ex_gpu[i_f];
            uybb  = uybb + fbb[i_f] * (T_P)ey_gpu[i_f];
            uzbb  = uzbb + fbb[i_f] * (T_P)ez_gpu[i_f];
        }
        uxbb  = uxbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[0] *overc_LBM_gpu;
        uybb  = uybb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[1] *overc_LBM_gpu;
        uzbb  = uzbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] *overc_LBM_gpu;
        udotubb = uxbb * uxbb + uybb * uybb + uzbb * uzbb;

        // rho = Density_gpu * (1. + sqrt(3.)*Velocity_Bound_gpu);//rhobb; 
        //ux=uxbb; uz=uybb; uy = uzbb;        
        rho = rhobb;//Density_gpu ;//rhobb; 
        ux=0.*overc_LBM_gpu; uz=0.*overc_LBM_gpu; 
        uy = Velocity_Bound_gpu;//-Lid_velocity_gpu + (prc(2.)*Lid_velocity_gpu*(iz-ez_gpu[i_f]))/(NZG0_D+1);
        udotu = ux * ux + uy * uy + uz * uz;
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = BB_xz_Right_gpu[i3];            
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            edotubb = (T_P)ex_gpu[i_f] * uxbb + (T_P)ey_gpu[i_f] * uybb + (T_P)ez_gpu[i_f] * uzbb;            
            feqbb = rhobb * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotubb + prc(4.5)*edotubb*edotubb - prc(1.5)* udotubb);
            edotu = (T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz;            
            feqb= rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);
            #if (GUO_EXTRAPOLATION == BEFORE_COLLSION)
            Boundary_xz0_gpu[i * 5 + i3] = feqb + (1.-SRT_OverTau_gpu) *(fbb[i_f] - feqbb);
            #elif (GUO_EXTRAPOLATION == AFTER_COLLSION)
            p_f[p_step_index_D((i_Global ) )] = feqb + (fbb[i_f] - feqbb);
            #endif
        }

        //y direction (right : outlet)
        iy = NYG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        rhobb = 0; uxbb = 0; uybb = 0; uzbb = 0;
        #pragma unroll
        for (int i3=0; i3<NDIR; i3++ ){
            i_f = i3;
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            #if (GUO_EXTRAPOLATION == BEFORE_COLLSION)
            fbb[i_f] = p_f[p_step_index_D((i_Global ) )];
            #elif (GUO_EXTRAPOLATION == AFTER_COLLSION)
            fbb[i_f] = p_f[p_step_index_D((i_Global ) + Length_gpu[i_f] )];
            #endif  
            rhobb = rhobb + fbb[i_f];
            uxbb  = uxbb + fbb[i_f] * (T_P)ex_gpu[i_f];
            uybb  = uybb + fbb[i_f] * (T_P)ey_gpu[i_f];
            uzbb  = uzbb + fbb[i_f] * (T_P)ez_gpu[i_f];
        }
        uxbb  = uxbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[0] *overc_LBM_gpu;
        uybb  = uybb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[1] *overc_LBM_gpu;
        uzbb  = uzbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] *overc_LBM_gpu;
        udotubb = uxbb * uxbb + uybb * uybb + uzbb * uzbb;

        rho = rhobb;//rhobb;//Density_gpu;
        //ux=uxbb; uz=uybb; uy = uzbb;
        //udotu = ux * ux + uy * uy + uz * uz;
        ux=0.*overc_LBM_gpu; uz=0.*overc_LBM_gpu; 
        uy = Velocity_Bound_gpu;//-Lid_velocity_gpu + (prc(2.)*Lid_velocity_gpu*(iz-ez_gpu[i_f]))/(NZG0_D+1);
        udotu = ux * ux + uy * uy + uz * uz;
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = Reverse_gpu[BB_xz_Right_gpu[i3]];            
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            edotubb = (T_P)ex_gpu[i_f] * uxbb + (T_P)ey_gpu[i_f] * uybb + (T_P)ez_gpu[i_f] * uzbb;            
            feqbb = rhobb * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotubb + prc(4.5)*edotubb*edotubb - prc(1.5)* udotubb);
            edotu = (T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz;            
            feqb= rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);            
            #if (GUO_EXTRAPOLATION == BEFORE_COLLSION)
            Boundary_xz1_gpu[i * 5 + i3] = feqb + (1.-SRT_OverTau_gpu) *(fbb[i_f] - feqbb);
            #elif (GUO_EXTRAPOLATION == AFTER_COLLSION)
            p_f[p_step_index_D((i_Global ) )] = feqb + (fbb[i_f] - feqbb);
            #endif
            //feqb + (fbb[i_f] - feqbb);// feqb + (1.-SRT_OverTau_gpu) *(fbb[i_f] - feqbb);
            //printf("\n\t i_f = %f", p_f[p_step_index_D((i_Global) )]); 
        }
    }
}
__global__ void BC_Guo2002_GPU_face_xz_load(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime, T_P* Boundary_xz0_gpu, T_P* Boundary_xz1_gpu) { // Zhao-Li et al., 2002, Non-equilibrium extrapolation method

    int i_f ;  //column //index of discrete velocities
    //int i_grid ;    //row  //index of grid position coordinates
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    I_INT  i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT  ix, iy, iz;
    I_INT i_Global;  //the index of point in the global distribution function array
    //Represents the position in the grid that needs to be solved
    //int position;
    
    if (i < imax ) {
        ix = i % (NXG0_D) + 1;
        iz = i / (NXG0_D) + 1;
        T_P fw;
        //y direction (left : inlet)        
        iy = 1;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){            
            i_f = BB_xz_Right_gpu[i3];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);            
            fw = Boundary_xz0_gpu[(i * 5 + i3)];
            p_f[p_step_index_D((i_Global ) )] = fw;
        }

        //y direction (right : outlet)
        iy = NYG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){            
            i_f = Reverse_gpu[BB_xz_Right_gpu[i3]];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);            
            fw = Boundary_xz1_gpu[(i * 5 + i3)];
            p_f[p_step_index_D((i_Global ) )] = fw;
        }
    }
}
__global__ void BC_Guo2002_GPU_face_xy_save(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime, T_P* Boundary_xy0_gpu, T_P* Boundary_xy1_gpu) { // Zhao-Li et al., 2002, Non-equilibrium extrapolation method

    int i_f ;  //column //index of discrete velocities
    //int i_grid ;    //row  //index of grid position coordinates
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    I_INT i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT ix, iy, iz;
    I_INT i_Global;  //the index of point in the global distribution function array
    //Represents the position in the grid that needs to be solved
    //int position;
    
    if (i < imax ) {        
        ix = i % (NXG0_D) + 1;
        iy = i / (NXG0_D) + 1;
        iz = 1;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        
        T_P fbb[NDIR], rhobb, uxbb, uybb, uzbb;   //fluid node which neighbour of boundary node
        T_P feqbb, edotubb, udotubb;
        T_P rho = Density_gpu;
        T_P ux=0, uz=0, uy = 0.;    //boundary node
        T_P feqb, edotu, udotu;

        // //z direction (bottom)
        rhobb = 0; uxbb = 0; uybb = 0; uzbb = 0;
        #pragma unroll
        for (int i3=0; i3<NDIR; i3++ ){
            i_f = i3;
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            #if (GUO_EXTRAPOLATION == BEFORE_COLLSION)
            fbb[i_f] = p_f[p_step_index_D((i_Global ) )];
            #elif (GUO_EXTRAPOLATION == AFTER_COLLSION)
            fbb[i_f] = p_f[p_step_index_D((i_Global ) + Length_gpu[i_f] )];
            #endif  
            rhobb = rhobb + fbb[i_f];
            uxbb  = uxbb + fbb[i_f] * (T_P)ex_gpu[i_f];
            uybb  = uybb + fbb[i_f] * (T_P)ey_gpu[i_f];
            uzbb  = uzbb + fbb[i_f] * (T_P)ez_gpu[i_f];
        }
        uxbb  = uxbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[0] *overc_LBM_gpu;
        uybb  = uybb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[1] *overc_LBM_gpu;
        uzbb  = uzbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] *overc_LBM_gpu;
        udotubb = uxbb * uxbb + uybb * uybb + uzbb * uzbb;

        rho = rhobb;//Density_gpu * (1. + sqrt(3.)*Velocity_Bound_gpu);//rhobb;
        ux=0.; uz=0.; 
        uy = Velocity_Bound_gpu;
        udotu = ux * ux + uy * uy + uz * uz;
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = BB_xy_top_gpu[i3];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            edotubb = (T_P)ex_gpu[i_f] * uxbb + (T_P)ey_gpu[i_f] * uybb + (T_P)ez_gpu[i_f] * uzbb;            
            feqbb = rhobb * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotubb + prc(4.5)*edotubb*edotubb - prc(1.5)* udotubb);
            edotu = (T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz;            
            feqb= rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);            
            #if (GUO_EXTRAPOLATION == BEFORE_COLLSION)
            Boundary_xy0_gpu[i * 5 + i3] = feqb + (1.-SRT_OverTau_gpu) *(fbb[i_f] - feqbb);
            #elif (GUO_EXTRAPOLATION == AFTER_COLLSION)
            p_f[p_step_index_D((i_Global ) )] = feqb + (fbb[i_f] - feqbb);
            #endif
        }

        //z direction (top)
        iz = NZG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        rhobb = 0; uxbb = 0; uybb = 0; uzbb = 0;
        #pragma unroll
        for (int i3=0; i3<NDIR; i3++ ){
            i_f = i3;
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            #if (GUO_EXTRAPOLATION == BEFORE_COLLSION)
            fbb[i_f] = p_f[p_step_index_D((i_Global ) )];
            #elif (GUO_EXTRAPOLATION == AFTER_COLLSION)
            fbb[i_f] = p_f[p_step_index_D((i_Global ) + Length_gpu[i_f] )];
            #endif  
            rhobb = rhobb + fbb[i_f];
            uxbb  = uxbb + fbb[i_f] * (T_P)ex_gpu[i_f];
            uybb  = uybb + fbb[i_f] * (T_P)ey_gpu[i_f];
            uzbb  = uzbb + fbb[i_f] * (T_P)ez_gpu[i_f];
        }
        uxbb  = uxbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[0] *overc_LBM_gpu;
        uybb  = uybb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[1] *overc_LBM_gpu;
        uzbb  = uzbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] *overc_LBM_gpu;
        udotubb = uxbb * uxbb + uybb * uybb + uzbb * uzbb;
        
        rho = Density_gpu;
        //ux=uxbb; uz=uybb; uy = uzbb;//Lid_velocity_gpu; 
		ux=prc(0.);
        uz=prc(0.); 
        uy = Velocity_Bound_gpu;	
        udotu = ux * ux + uy * uy + uz * uz;
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = Reverse_gpu[BB_xy_top_gpu[i3]];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            edotubb = (T_P)ex_gpu[i_f] * uxbb + (T_P)ey_gpu[i_f] * uybb + (T_P)ez_gpu[i_f] * uzbb;            
            feqbb = rhobb * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotubb + prc(4.5)*edotubb*edotubb - prc(1.5)* udotubb);
            edotu = (T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz;            
            feqb= rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);            
            #if (GUO_EXTRAPOLATION == BEFORE_COLLSION)
            Boundary_xy1_gpu[i * 5 + i3] = feqb + (1.-SRT_OverTau_gpu) *(fbb[i_f] - feqbb);
            #elif (GUO_EXTRAPOLATION == AFTER_COLLSION)
            p_f[p_step_index_D((i_Global ) )] = feqb + (fbb[i_f] - feqbb);
            #endif
            //feqb + (fbb[i_f] - feqbb);// feqb + (1.-SRT_OverTau_gpu) *(fbb[i_f] - feqbb);
            //printf("\n\t i_f = %f", p_f[p_step_index_D((i_Global) )]); 
        }
    }
}
__global__ void BC_Guo2002_GPU_face_xy_load(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime, T_P* Boundary_xy0_gpu, T_P* Boundary_xy1_gpu) { // Zhao-Li et al., 2002, Non-equilibrium extrapolation method

    int i_f ;  //column //index of discrete velocities
    //int i_grid ;    //row  //index of grid position coordinates
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    I_INT  i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT  ix, iy, iz;
    I_INT i_Global;  //the index of point in the global distribution function array
    //Represents the position in the grid that needs to be solved
    //int position;
    
    if (i < imax ) {
        ix = i % (NXG0_D) + 1;
        iy = i / (NXG0_D) + 1;
        T_P fw;
        //z direction (bottom)        
        iz = 1;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){            
            i_f = BB_xy_top_gpu[i3];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            fw = Boundary_xy0_gpu[(i * 5 + i3)]; 
            p_f[p_step_index_D((i_Global ) )] = fw;
        }

        //z direction (top)
        iz = NZG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){            
            i_f = Reverse_gpu[BB_xy_top_gpu[i3]];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);            
            fw = Boundary_xy1_gpu[(i * 5 + i3)]; 
            p_f[p_step_index_D((i_Global ) )] = fw;
        }
    }    
}
__global__ void BC_Guo2002_GPU_face_yz_save(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime, T_P* Boundary_yz0_gpu, T_P* Boundary_yz1_gpu) { // Zhao-Li et al., 2002, Non-equilibrium extrapolation method

    int i_f ;  //column //index of discrete velocities
    //int i_grid ;    //row  //index of grid position coordinates
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    I_INT  i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT  ix, iy, iz;
    I_INT i_Global;  //the index of point in the global distribution function array
    //Represents the position in the grid that needs to be solved
    //int position;
    
    if (i < imax ) {
        //x direction (back)
        ix = 1;
        iy = i % (NYG0_D) + 1;
        iz = i / (NYG0_D) + 1;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array

        T_P fbb[NDIR], rhobb, uxbb, uybb, uzbb;   //fluid node which neighbour of boundary node
        T_P feqbb, edotubb, udotubb;
        T_P rho = Density_gpu;
        T_P ux=0, uz=0, uy = 0.;    //boundary node
        T_P feqb, edotu, udotu;
        
        rhobb = 0; uxbb = 0; uybb = 0; uzbb = 0;
        #pragma unroll
        for (int i3=0; i3<NDIR; i3++ ){
            i_f = i3;
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            #if (GUO_EXTRAPOLATION == BEFORE_COLLSION)
            fbb[i_f] = p_f[p_step_index_D((i_Global ) )];
            #elif (GUO_EXTRAPOLATION == AFTER_COLLSION)
            fbb[i_f] = p_f[p_step_index_D((i_Global ) + Length_gpu[i_f] )];
            #endif  
            rhobb = rhobb + fbb[i_f];
            uxbb  = uxbb + fbb[i_f] * (T_P)ex_gpu[i_f];
            uybb  = uybb + fbb[i_f] * (T_P)ey_gpu[i_f];
            uzbb  = uzbb + fbb[i_f] * (T_P)ez_gpu[i_f];
        }
        uxbb  = uxbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[0] *overc_LBM_gpu;
        uybb  = uybb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[1] *overc_LBM_gpu;
        uzbb  = uzbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] *overc_LBM_gpu;
        udotubb = uxbb * uxbb + uybb * uybb + uzbb * uzbb;

        // rho = Density_gpu * (1. + sqrt(3.)*Velocity_Bound_gpu);//rhobb;
        // ux=0; uz=0; uy = 0;
        rho = rhobb ;//rhobb; 
        ux=0.; uz=0.; 
        uy = Velocity_Bound_gpu;
        udotu = ux * ux + uy * uy + uz * uz;     
        #pragma unroll  
        for (int i3=0; i3<5; i3++ ){
            i_f = BB_yz_front_gpu[i3];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            edotubb = (T_P)ex_gpu[i_f] * uxbb + (T_P)ey_gpu[i_f] * uybb + (T_P)ez_gpu[i_f] * uzbb;            
            feqbb = rhobb * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotubb + prc(4.5)*edotubb*edotubb - prc(1.5)* udotubb);
            edotu = (T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz;            
            feqb= rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);            
            #if (GUO_EXTRAPOLATION == BEFORE_COLLSION)
            Boundary_yz0_gpu[i * 5 + i3] = feqb + (1.-SRT_OverTau_gpu) *(fbb[i_f] - feqbb);
            #elif (GUO_EXTRAPOLATION == AFTER_COLLSION)
            p_f[p_step_index_D((i_Global ) )] = feqb + (fbb[i_f] - feqbb);
            #endif
        }

        //x direction (front)
        ix = NXG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        rhobb = 0; uxbb = 0; uybb = 0; uzbb = 0;
        #pragma unroll
        for (int i3=0; i3<NDIR; i3++ ){
            i_f = i3;
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            #if (GUO_EXTRAPOLATION == BEFORE_COLLSION)
            fbb[i_f] = p_f[p_step_index_D((i_Global ) )];
            #elif (GUO_EXTRAPOLATION == AFTER_COLLSION)
            fbb[i_f] = p_f[p_step_index_D((i_Global ) + Length_gpu[i_f] )];
            #endif            
            rhobb = rhobb + fbb[i_f];
            uxbb  = uxbb + fbb[i_f] * (T_P)ex_gpu[i_f];
            uybb  = uybb + fbb[i_f] * (T_P)ey_gpu[i_f];
            uzbb  = uzbb + fbb[i_f] * (T_P)ez_gpu[i_f];
        }
        uxbb  = uxbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[0] *overc_LBM_gpu;
        uybb  = uybb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[1] *overc_LBM_gpu;
        uzbb  = uzbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] *overc_LBM_gpu;
        udotubb = uxbb * uxbb + uybb * uybb + uzbb * uzbb;
        
        rho = rhobb ;//rhobb;
        //ux=0; uz=0; uy = -Lid_velocity_gpu+2*Lid_velocity_gpu*(iz)/(NZG0_D+1) ;
        ux=0.; uz=0.; 
        uy = Velocity_Bound_gpu;
        udotu = ux * ux + uy * uy + uz * uz;
        //udotu = ux * ux + uy * uy + uz * uz;
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = Reverse_gpu[BB_yz_front_gpu[i3]];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            edotubb = (T_P)ex_gpu[i_f] * uxbb + (T_P)ey_gpu[i_f] * uybb + (T_P)ez_gpu[i_f] * uzbb;            
            feqbb = rhobb * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotubb + prc(4.5)*edotubb*edotubb - prc(1.5)* udotubb);
            edotu = (T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz;            
            feqb= rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);            
            #if (GUO_EXTRAPOLATION == BEFORE_COLLSION)
            Boundary_yz1_gpu[i * 5 + i3] = feqb + (1.-SRT_OverTau_gpu) *(fbb[i_f] - feqbb);
            #elif (GUO_EXTRAPOLATION == AFTER_COLLSION)
            p_f[p_step_index_D((i_Global ) )] = feqb + (fbb[i_f] - feqbb);
            #endif
            //feqb + (fbb[i_f] - feqbb);// feqb + (1.-SRT_OverTau_gpu) *(fbb[i_f] - feqbb);
            //printf("\n\t i_f = %f", p_f[p_step_index_D((i_Global) )]); 
        }
    }
}
__global__ void BC_Guo2002_GPU_face_yz_load(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime, T_P* Boundary_yz0_gpu, T_P* Boundary_yz1_gpu) { // Zhao-Li et al., 2002, Non-equilibrium extrapolation method

    int i_f ;  //column //index of discrete velocities
    //int i_grid ;    //row  //index of grid position coordinates
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    I_INT  i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT  ix, iy, iz;
    I_INT i_Global;  //the index of point in the global distribution function array
    //Represents the position in the grid that needs to be solved
    //int position;
    
    if (i < imax ) {
        iy = i % (NYG0_D) + 1;
        iz = i / (NYG0_D) + 1;
        T_P fw;
        //z direction (back)        
        ix = 1;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = BB_yz_front_gpu[i3];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            fw = Boundary_yz0_gpu[(i * 5 + i3)];
            p_f[p_step_index_D((i_Global ) )] = fw;
        }

        //z direction (front)
        ix = NXG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){            
            i_f = Reverse_gpu[BB_yz_front_gpu[i3]];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            fw = Boundary_yz1_gpu[(i * 5 + i3)];
            p_f[p_step_index_D((i_Global ) )] = fw;
        }
    }
}
__global__ void BC_NEEM_AfterColl_GPU_face_xz(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime, T_P* Boundary_xz0_gpu, T_P* Boundary_xz1_gpu) { // Non-equilibrium extrapolation method (NEEM) with the disturabtion after collision
    int i_f ;  //column //index of discrete velocities    
    T_P *p_f;   //the point to rowHead(fi in first grid)
    // Indexing (Thread) 
    I_INT  i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT  ix, iy, iz;
    I_INT i_Global;  //the index of point in the global distribution function array
    //Represents the position in the grid that needs to be solved
    //int position;
    
    if (i < imax ) {
        //y direction (left : inlet)
        ix = i % (NXG0_D) + 1;
        iy = 1;
        iz = i / (NXG0_D) + 1;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array

        T_P fbb[NDIR], rhobb, uxbb, uybb, uzbb;   //fluid node which neighbour of boundary node
        T_P feqbb, edotubb, udotubb;
        T_P rho = Density_gpu;
        T_P ux=0, uz=0, uy = Lid_velocity_gpu;    //boundary node
        T_P feqb, edotu, udotu;
        
        rhobb = 0; uxbb = 0; uybb = 0; uzbb = 0;
        #pragma unroll
        for (int i3=0; i3<NDIR; i3++ ){
            i_f = i3;
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            fbb[i_f] = p_f[p_step_index_D((i_Global )+ Length_gpu[i_f])];
            rhobb = rhobb + fbb[i_f];
            uxbb  = uxbb + fbb[i_f] * (T_P)ex_gpu[i_f];
            uybb  = uybb + fbb[i_f] * (T_P)ey_gpu[i_f];
            uzbb  = uzbb + fbb[i_f] * (T_P)ez_gpu[i_f];
        }
        uxbb  = uxbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[0] *overc_LBM_gpu;
        uybb  = uybb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[1] *overc_LBM_gpu;
        uzbb  = uzbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] *overc_LBM_gpu;
        udotubb = uxbb * uxbb + uybb * uybb + uzbb * uzbb;

        // rho = Density_gpu * (1. + sqrt(3.)*Velocity_Bound_gpu);//rhobb;        
        rho = rhobb;//Density_gpu ;//rhobb;        
        ux=0; uz=0; uy = Velocity_Bound_gpu;
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){
            i_f = BB_xz_Right_gpu[i3];
            ux=0. *overc_LBM_gpu; uz=0. *overc_LBM_gpu; uy = 0. *overc_LBM_gpu;//-Lid_velocity_gpu + (prc(2.)*Lid_velocity_gpu*(iz-ez_gpu[i_f]))/(NZG0_D+1);
            udotu = ux * ux + uy * uy + uz * uz;
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            edotubb = (T_P)ex_gpu[i_f] * uxbb + (T_P)ey_gpu[i_f] * uybb + (T_P)ez_gpu[i_f] * uzbb;            
            feqbb = rhobb * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotubb + prc(4.5)*edotubb*edotubb - prc(1.5)* udotubb);
            edotu = (T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz;            
            feqb= rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);
            p_f[p_step_index_D((i_Global ) )] = feqb + (fbb[i_f] - feqbb);
        }

        //y direction (right : outlet)
        iy = NYG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        rhobb = 0; uxbb = 0; uybb = 0; uzbb = 0;
        #pragma unroll
        for (int i3=0; i3<NDIR; i3++ ){
            i_f = i3;
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            fbb[i_f] = p_f[p_step_index_D((i_Global ) + Length_gpu[i_f] )];
            rhobb = rhobb + fbb[i_f];
            uxbb  = uxbb + fbb[i_f] * (T_P)ex_gpu[i_f];
            uybb  = uybb + fbb[i_f] * (T_P)ey_gpu[i_f];
            uzbb  = uzbb + fbb[i_f] * (T_P)ez_gpu[i_f];
        }
        uxbb  = uxbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[0] *overc_LBM_gpu;
        uybb  = uybb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[1] *overc_LBM_gpu;
        uzbb  = uzbb / rhobb + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] *overc_LBM_gpu;
        udotubb = uxbb * uxbb + uybb * uybb + uzbb * uzbb;

        rho = rhobb;//rhobb;//Density_gpu;
        //ux=uxbb; uz=uybb; uy = uzbb;
        //udotu = ux * ux + uy * uy + uz * uz;      
        #pragma unroll  
        for (int i3=0; i3<5; i3++ ){
            i_f = Reverse_gpu[BB_xz_Right_gpu[i3]];
            ux=0. *overc_LBM_gpu; uz=0. *overc_LBM_gpu; uy = 0.* overc_LBM_gpu;//-Lid_velocity_gpu + (prc(2.)*Lid_velocity_gpu*(iz-ez_gpu[i_f]))/(NZG0_D+1);
            udotu = ux * ux + uy * uy + uz * uz;
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            edotubb = (T_P)ex_gpu[i_f] * uxbb + (T_P)ey_gpu[i_f] * uybb + (T_P)ez_gpu[i_f] * uzbb;            
            feqbb = rhobb * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotubb + prc(4.5)*edotubb*edotubb - prc(1.5)* udotubb);
            edotu = (T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz;            
            feqb= rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);
            p_f[p_step_index_D((i_Global ) )] = feqb + (fbb[i_f] - feqbb);            
        }
    }
}
__global__ void Out_ABC_xz(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime, T_P* Boundary_xz0_gpu, T_P* Boundary_xz1_gpu) { // Zhao-Li et al., 2002, Non-equilibrium extrapolation method

    int i_f ;  //column //index of discrete velocities
    //int i_grid ;    //row  //index of grid position coordinates
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    I_INT  i = threadIdx.x + blockIdx.x * blockDim.x;
    I_INT  ix, iy, iz;
    I_INT i_Global;  //the index of point in the global distribution function array
    //Represents the position in the grid that needs to be solved
    //int position;
    
    if (i < imax ) {
        //y direction (right : outlet)                
        ix = i % (NXG0_D) + 1;
        iz = i / (NXG0_D) + 1;        
        T_P fw;
        iy = NYG0_D;
        i_Global = p_index_D(ix, iy, iz);  //the index of point in the global distribution function array
        #pragma unroll
        for (int i3=0; i3<5; i3++ ){            
            i_f = Reverse_gpu[BB_xz_Right_gpu[i3]];
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            //fw = Density_gpu * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*(T_P)ey_gpu[i_f]*Lid_velocity_gpu + prc(4.5)*(T_P)ey_gpu[i_f]*Lid_velocity_gpu*(T_P)ey_gpu[i_f]*Lid_velocity_gpu - prc(1.5)* Lid_velocity_gpu*Lid_velocity_gpu);
            fw = Density_gpu * w_equ_gpu[i_f] * (prc(1.) );
            p_f[p_step_index_D((i_Global ) )] = fw;
        }
    }
}
//===================================================================================================================================== =
//----------------------compute particle boundary in GPU ----------------------
//===================================================================================================================================== =
__global__ void particle_BB_BC_GPU_point(I_INT imax, T_P* pdf_old_gpu, I_INT * Particl_BC_I_gpu, int * Particl_BC_DV_gpu, I_INT pitch, int ntime) {
    int i_f;  //column //index of discrete velocities    
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    I_INT i = threadIdx.x + blockIdx.x * blockDim.x;
    //Represents the position in the grid that needs to be solved
    I_INT position;

    T_P f[NDIR];

    if (i < imax ) {
        i_f = Particl_BC_DV_gpu[i];
        p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
        position = Particl_BC_I_gpu[i];
        f[i_f] = p_f[p_step_index_D((position + Length_gpu[i_f]) )];
        p_f = (T_P*)((char*)pdf_old_gpu + Reverse_gpu[i_f] * pitch);        
        p_f[p_step_index_D(position )] = f[i_f];
    }
}
__global__ void particle_IBB_BC_GPU_point(I_INT imax, T_P* pdf_old_gpu, I_INT * Particl_BC_I_gpu, int * Particl_BC_DV_gpu, T_P * Particl_BC_q_gpu, I_INT pitch, int ntime, T_P * Force_block_gpu, T_P * MForce_block_gpu) {
    int i_f;  //column //index of discrete velocities    
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    int tid = threadIdx.x;
    I_INT i = threadIdx.x + blockIdx.x * blockDim.x;
    //Represents the position in the grid that needs to be solved
    I_INT position;

    T_P f1r,f2r, f2, f1tp;
    T_P q;
    T_P ux=0., uy=0., uz=0.;
    //T_P edotu;
    
    __shared__ T_P sForce[3], sMForce[3];
    T_P forcex,forcey,forcez;
    if (tid == 0 ){
        #pragma unroll
        for (int ff = 0; ff < 3; ff++){
            sForce[ff] = prc(0.);
            //sForcex = 0.;sForcey = 0.;sForcez = 0.;
            sMForce[ff] = prc(0.);
        }
    }
    __syncthreads();
    if (i < imax ) {        
        // =============Bouzidi,2001 2nd order=========>>
        T_P f3r, f3;
        i_f = Particl_BC_DV_gpu[i];
        p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
        position = Particl_BC_I_gpu[i];
        q = Particl_BC_q_gpu[i];
        f1r = p_f[p_step_index_D((position + Length_gpu[i_f]) )];        
        //edotu = (T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz;
        if (q<=0.5){
            f2r = p_f[p_step_index_D((position ) )];
            f3r = p_f[p_step_index_D((position - Length_gpu[i_f]) )];            
            p_f = (T_P*)((char*)pdf_old_gpu + Reverse_gpu[i_f] * pitch);
            f1tp = q* (2.*q+1.)* f1r + (2.*q+1.)*(1.-2.*q)*f2r - q*(1.-2.*q) * f3r;
        }else{
            p_f = (T_P*)((char*)pdf_old_gpu + Reverse_gpu[i_f] * pitch);
            f2 = p_f[p_step_index_D((position + Length_gpu[Reverse_gpu[i_f]]) )];
            f3 = p_f[p_step_index_D((position + 2*Length_gpu[Reverse_gpu[i_f]]) )];
            f1tp = f1r/(q*(2.*q+1.)) + (2.*q-1.)/(q) * f2 + (1.-2.*q)/(2.*q+1.) * f3;
        }
        p_f[p_step_index_D(position )]  = f1tp;
        
        forcex = (f1r * (ex_gpu[i_f] - ux) - f1tp * (ex_gpu[Reverse_gpu[i_f]] - ux));
        forcey = (f1r * (ey_gpu[i_f] - uy) - f1tp * (ey_gpu[Reverse_gpu[i_f]] - uy));
        forcez = (f1r * (ez_gpu[i_f] - uz) - f1tp * (ez_gpu[Reverse_gpu[i_f]] - uz));

        atomicAdd(&sForce[0], forcex);
        atomicAdd(&sForce[1], forcey);
        atomicAdd(&sForce[2], forcez);

        __syncthreads();

        if (tid == 0) {
            #pragma unroll
            for (int ff = 0; ff < 3; ff++){
                atomicAdd(Force_block_gpu + gridDim.x * ff + blockIdx.x, sForce[ff]);
                atomicAdd(MForce_block_gpu + gridDim.x * ff + blockIdx.x, sMForce[ff]);
                //printf("\n\t i_f = %d", gridDim.x); 
            }
        }
    }
}
__global__ void particle_IBB_BC_Force_GPU(I_INT imax, T_P* pdf_old_gpu, I_INT * Particl_BC_I_gpu, int * Particl_BC_DV_gpu, T_P * Particl_BC_q_gpu, I_INT pitch, int ntime, T_P * Force_block_gpu, T_P * MForce_block_gpu, T_P * Particl_BC_fOld_gpu, T_P * Particl_Coords_gpu, T_P * Particl_Velos_gpu, T_P * Particl_AngulVelos_gpu) {
    int i_f;  //column //index of discrete velocities    
    T_P *p_f;   //the point to rowHead(fi in first grid)

    // Indexing (Thread) 
    int tid = threadIdx.x;
    I_INT i = threadIdx.x + blockIdx.x * blockDim.x;
    //Represents the position in the grid that needs to be solved
    I_INT position;

    T_P f1, f1r, f1rO, f1tp;
    T_P q;
    T_P uwx=0., uwy=0., uwz=0.;
    T_P Bx,By,Bz;
    T_P edotuw;
    
    __shared__ T_P sForce[3], sMForce[3];
    T_P forcex,forcey,forcez;
    T_P Mforcex,Mforcey,Mforcez;
    if (tid == 0 ){
        #pragma unroll
        for (int ff = 0; ff < 3; ff++){
            sForce[ff] = prc(0.);            
            sMForce[ff] = prc(0.);
        }
    }
    __syncthreads();
    if (i < imax ) {        
        // =============Zhao and Yong, 2017 2nd order=========>>
        i_f = Reverse_gpu[Particl_BC_DV_gpu[i]];
        position = Particl_BC_I_gpu[i];
        q = Particl_BC_q_gpu[i];
        Bx = position%XYG1_D % NXG1_D + q * (T_P)ex_gpu[i_f] - Particl_Coords_gpu[0];
        By = position%XYG1_D / NXG1_D + q * (T_P)ey_gpu[i_f] - Particl_Coords_gpu[1];
        Bz = position/XYG1_D          + q * (T_P)ez_gpu[i_f] - Particl_Coords_gpu[2];
        Bx = Bx * dx_LBM_gpu; By = By * dx_LBM_gpu; Bz = Bz * dx_LBM_gpu;
        uwx = Particl_Velos_gpu[0] + (Particl_AngulVelos_gpu[1]*Bz - Particl_AngulVelos_gpu[2]*By);
        uwy = Particl_Velos_gpu[1] + (Particl_AngulVelos_gpu[2]*Bx - Particl_AngulVelos_gpu[0]*Bz);
        uwz = Particl_Velos_gpu[2] + (Particl_AngulVelos_gpu[0]*By - Particl_AngulVelos_gpu[1]*Bx);
        
        p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
        f1rO = Particl_BC_fOld_gpu[i];
        f1 = p_f[p_step_index_D((position + Length_gpu[i_f]) )];
        edotuw = (T_P)ex_gpu[i_f] * uwx + (T_P)ey_gpu[i_f] * uwy + (T_P)ez_gpu[i_f] * uwz;  edotuw = edotuw * overc_LBM_gpu;
        f1tp = 2.*q/(1.+2.*q) * f1 + 1./(1.+2.*q) * f1rO + 6./(1.+2.*q)*Density_gpu*w_equ_gpu[i_f]*edotuw;
        p_f[p_step_index_D(position )]  = f1tp;

        i_f = Reverse_gpu[i_f];
        p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
        f1r = p_f[p_step_index_D((position + Length_gpu[i_f]) )];
        //Particl_BC_fOld_gpu[i] = p_f[p_step_index_D((position ) )]; //This step may need to do only when the particles are fixed, and this function cannot be implemented here when the particles are moving.
        
        forcex = (f1r * (ex_gpu[i_f] * c_LBM_gpu - uwx) - f1tp * (ex_gpu[Reverse_gpu[i_f]] * c_LBM_gpu - uwx));
        forcey = (f1r * (ey_gpu[i_f] * c_LBM_gpu - uwy) - f1tp * (ey_gpu[Reverse_gpu[i_f]] * c_LBM_gpu - uwy));
        forcez = (f1r * (ez_gpu[i_f] * c_LBM_gpu - uwz) - f1tp * (ez_gpu[Reverse_gpu[i_f]] * c_LBM_gpu - uwz));
        atomicAdd(&sForce[0], forcex);
        atomicAdd(&sForce[1], forcey);
        atomicAdd(&sForce[2], forcez);

        Mforcex = (By*forcez - Bz*forcey);
        Mforcey = (Bz*forcex - Bx*forcez);
        Mforcez = (Bx*forcey - By*forcex);
        atomicAdd(&sMForce[0], Mforcex);
        atomicAdd(&sMForce[1], Mforcey);
        atomicAdd(&sMForce[2], Mforcez);

        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int ff = 0; ff < 3; ff++){
                // atomicAdd(Force_block_gpu + gridDim.x * ff + blockIdx.x, sForce[ff]);
                // atomicAdd(MForce_block_gpu + gridDim.x * ff + blockIdx.x, sMForce[ff]);
                Force_block_gpu[blockIdx.x + gridDim.x * ff]  = sForce[ff];
                MForce_block_gpu[blockIdx.x + gridDim.x * ff] = sMForce[ff];
            }
            //printf("\n\t %d, %f", blockIdx.x + gridDim.x * 2, sForce[2]);
        }
    }
}
__global__ void particle_IBB_BC_save_GPU(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime, I_INT * Particl_BC_I_gpu, int * Particl_BC_DV_gpu, T_P * Particl_BC_fOld_gpu) {
    int i_f;  //column //index of discrete velocities    
    T_P *p_f;   //the point to rowHead(fi in first grid)

    I_INT i = threadIdx.x + blockIdx.x * blockDim.x;
    //Represents the position in the grid that needs to be solved
    I_INT position;

    if (i < imax ) {
        // =============save info of before collision distribution in last time step. (Zhao and Yong, 2017 2nd order)=========>>
        i_f = Particl_BC_DV_gpu[i];
        position = Particl_BC_I_gpu[i];
        p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
        Particl_BC_fOld_gpu[i] = p_f[p_step_index_D((position ) )]; //This step may need to do only when the particles are fixed, and this function cannot be implemented here when the particles are moving.
    }
}
//===================================================================================================================================== =
//----------------------Refill process in GPU ----------------------
//===================================================================================================================================== =
__global__ void Refill_new_point_moving_particles_gpu(I_INT imax, T_P* pdf_old_gpu, I_INT pitch, int ntime, I_INT * Refill_Point_I_gpu, int * Refill_Point_DV_gpu, T_P* Particl_Velos_gpu, T_P* Particl_AngulVelos_gpu, T_P* Particl_Coords_gpu) {
    // Indexing (Thread) 
    I_INT i = threadIdx.x + blockIdx.x * blockDim.x;
    //Represents the position in the grid that needs to be solved    
    if (i < imax ) {        
        T_P *p_f;   //the point to rowHead(fi in first grid)
        I_INT ip_N, ip_F;  //the index of new points and nearest fluid point
        T_P rho, ux, uy, uz;  //macrovariables of the nearest fluid point
        T_P udotu, edotu;
        T_P f[NDIR], feq[NDIR], feq_w[NDIR]; //distribution of the nearest fluid point
        T_P uwx, uwy, uwz;
        T_P uwdotuw, edotuw;
        ip_N = Refill_Point_I_gpu[i];        
        ip_F = ip_N + Length_gpu[Refill_Point_DV_gpu[i]];      
        rho = 0;
        ux = 0; uy = 0; uz = 0;
        #pragma unroll
        for (int i_f=0; i_f<NDIR; i_f++){
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);            
            f[i_f] = p_f[p_step_index_D(ip_F )];
            rho = rho + f[i_f];
            ux = ux + f[i_f]* (T_P)ex_gpu[i_f];
            uy = uy + f[i_f]* (T_P)ey_gpu[i_f];
            uz = uz + f[i_f]* (T_P)ez_gpu[i_f];
        }
        ux = (ux/rho + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[0] * overc_LBM_gpu); 
        uy = (uy/rho + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[1] * overc_LBM_gpu); 
        uz = (uz/rho + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2] * overc_LBM_gpu); 
        udotu = ux * ux + uy * uy + uz * uz;

        T_P Bx, By, Bz;
        // int i_f = Reverse_gpu[Refill_Point_DV_gpu[i]];
        // Bx = ip_N%XYG1_D % NXG1_D + q * (T_P)ex_gpu[i_f] - Particl_Coords_gpu[0];
        // By = ip_N%XYG1_D / NXG1_D + q * (T_P)ey_gpu[i_f] - Particl_Coords_gpu[1];
        // Bz = ip_N/XYG1_D          + q * (T_P)ez_gpu[i_f] - Particl_Coords_gpu[2];
        Bx = ip_N%XYG1_D % NXG1_D - Particl_Coords_gpu[0];
        By = ip_N%XYG1_D / NXG1_D - Particl_Coords_gpu[1];
        Bz = ip_N/XYG1_D          - Particl_Coords_gpu[2];
        Bx = Bx * dx_LBM_gpu; By = By * dx_LBM_gpu; Bz = Bz * dx_LBM_gpu;
        uwx = Particl_Velos_gpu[0] + (Particl_AngulVelos_gpu[1]*Bz - Particl_AngulVelos_gpu[2]*By);
        uwy = Particl_Velos_gpu[1] + (Particl_AngulVelos_gpu[2]*Bx - Particl_AngulVelos_gpu[0]*Bz);
        uwz = Particl_Velos_gpu[2] + (Particl_AngulVelos_gpu[0]*By - Particl_AngulVelos_gpu[1]*Bx);
        uwx = uwx * overc_LBM_gpu; uwy = uwy * overc_LBM_gpu; uwz = uwz * overc_LBM_gpu;
        uwdotuw = uwx * uwx + uwy * uwy + uwz * uwz;
        #pragma unroll
        for (int i_f=0; i_f<NDIR; i_f++){
            p_f = (T_P*)((char*)pdf_old_gpu + i_f * pitch);
            edotu = (T_P)ex_gpu[i_f] * ux + (T_P)ey_gpu[i_f] * uy + (T_P)ez_gpu[i_f] * uz;
            edotuw = (T_P)ex_gpu[i_f] * uwx + (T_P)ey_gpu[i_f] * uwy + (T_P)ez_gpu[i_f] * uwz;
            feq[i_f] = rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);
            feq_w[i_f]= rho * w_equ_gpu[i_f] * (prc(1.) + prc(3.)*edotuw + prc(4.5)*edotuw*edotuw - prc(1.5)* uwdotuw);            
            p_f[p_step_index_D(ip_N )] = feq_w[i_f] + f[i_f] - feq[i_f];
        }
    }
   
}
//===================================================================================================================================== =
//----------------------compute macroscopic varaibles from PDFs in GPU ----------------------
//===================================================================================================================================== =
__global__ void compute_macro_vars_OSI_gpu(I_INT imax, T_P* pdf_gpu, T_P* Density_rho_gpu, T_P* Velocity_ux_gpu, T_P* Velocity_uy_gpu, T_P* Velocity_uz_gpu, int pitch, int ntime) { 
    T_P f[NDIR], rho, ux, uy, uz;
    int i_f;

    T_P *p_f;   //the point to rowHead(fi in first grid)

    I_INT i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < imax ) {
        rho = 0;
        ux = 0; uy = 0; uz = 0;
        #pragma unroll
        for (i_f=0; i_f < NDIR; i_f++ ){
            p_f = (T_P*)((char*)pdf_gpu + i_f * pitch);            
            f[i_f] = p_f[ p_step_index_D(i ) ]; 

            rho = rho + f[i_f];
            ux = ux + f[i_f] * ex_gpu[i_f] * c_LBM_gpu;   
            uy = uy + f[i_f] * ey_gpu[i_f] * c_LBM_gpu;
            uz = uz + f[i_f] * ez_gpu[i_f] * c_LBM_gpu;
        }
        ux = (ux/rho + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[0]) ; uy = (uy/rho + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[1]); uz = (uz/rho + prc(0.5)*dt_LBM_gpu*body_accelerate_gpu[2]); 
        Density_rho_gpu[i] = rho;
        Velocity_ux_gpu[i] = ux;
        Velocity_uy_gpu[i] = uy;
        Velocity_uz_gpu[i] = uz;
    }
}

void get_macro_vars_cpu(){
    cout << "Calculate macro variables and Copying results back to the host side .... ";
    dim3 block;
    dim3 grid;
    block.x = block_Threads_X;    
    grid.x = (int(NGRID1) + block.x - 1) / block.x;
    compute_macro_vars_OSI_gpu << <grid, block >> > (NGRID1, pdf_gpu, Density_rho_gpu, Velocity_ux_gpu, Velocity_uy_gpu, Velocity_uz_gpu, pitch, ntime);
    cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    
    cudaErrorCheck(cudaMemcpy(Velocity_ux, Velocity_ux_gpu, mem_cells_s1_TP, cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(Velocity_uy, Velocity_uy_gpu, mem_cells_s1_TP, cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(Velocity_uz, Velocity_uz_gpu, mem_cells_s1_TP, cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(Density_rho, Density_rho_gpu, mem_cells_s1_TP, cudaMemcpyDeviceToHost));
    
    cudaErrorCheck(cudaMemcpy2D(pdf, sizeof(T_P)*NGRID1, pdf_gpu, pitch, sizeof(T_P)*NGRID1, NDIR, cudaMemcpyDeviceToHost));

    cout << "\n sphere 1 Force:"<<spheres[0].Force[0] << "\t" << spheres[0].Force[1] << "\t" << spheres[0].Force[2] ;
    cout << "\n sphere 2 Force:"<<spheres[1].Force[0] << "\t" << spheres[1].Force[1] << "\t" << spheres[1].Force[2] << endl;
    cout << " Complete " << endl;
}

//===================================================================================================================================== =
//----------------------main iteration kernel in GPU ----------------------
//===================================================================================================================================== =
void main_iteration_kernel_GPU() {
    //get the velocity of boundary
    T_P temp_VB;
    temp_VB = Lid_velocity * sin(AnguFreq_lbm * ntime * dt_LBM);   
    //cout <<  temp_VB<<endl;
    cudaErrorCheck(cudaMemcpyToSymbol(Velocity_Bound_gpu, &temp_VB, sizeof(T_P) ));

    /* define grid structure */
    dim3 block;
    dim3 grid;
    block.x = block_Threads_X;  block.y = 1;    block.z = 1;
    grid.y = 1; grid.z = 1; 

    /* ************************** one step ***************************************** */    
    // // // // periodic boundary condition
    // grid.x = (XYG0 + block.x - 1) / block.x;
    // periodic_BC_GPU_face_xy << <grid, block >> > (XYG0, pdf_old_gpu, pitch_old, ntime);
    // cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    // grid.x = (YZG0 + block.x - 1) / block.x;
    // periodic_BC_GPU_face_yz << <grid, block >> > (YZG0, pdf_old_gpu, pitch_old, ntime);
    // cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    // // // // grid.x = (NYG0 + block.x - 1) / block.x;
    // // // // periodic_BC_GPU_edge_y << <grid, block >> > (NYG0, pdf_old_gpu, pitch_old, ntime);    
    // // // // cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    // grid.x = (XZG0 + block.x - 1) / block.x;
    // periodic_BC_GPU_face_xz << <grid, block >> > (XZG0, pdf_old_gpu, pitch_old, ntime);
    // cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    
    // // bounce back
    // block.x = block_Threads_X; 
    // grid.x = ((XYG0) + block.x - 1) / block.x;
    // wall_BB_BC_GPU_face_xy << <grid, block >> > ((XYG0), pdf_old_gpu, pitch_old, ntime);
    // cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    // grid.x = ((YZG0) + block.x - 1) / block.x;
    // wall_BB_BC_GPU_face_yz << <grid, block >> > ((YZG0), pdf_old_gpu, pitch_old, ntime);
    // cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    // grid.x = ((XZG0) + block.x - 1) / block.x;
    // wall_BB_BC_GPU_face_xz << <grid, block >> > ((XZG0), pdf_old_gpu, pitch_old, ntime);
    // cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());

    // // Non-equilibrium extrapolation method for velocity and pressure boundary conditions, Guo,2002 (Take care to consider the order of boundary implementation and collision implementation)
    // #if (GUO_EXTRAPOLATION == BEFORE_COLLSION)
    // grid.x = (XZG0 + block.x - 1) / block.x;
    // BC_Guo2002_GPU_face_xz_load << <grid, block >> > (XZG0, pdf_old_gpu, pitch_old, ntime, Boundary_xz0_gpu, Boundary_xz1_gpu);
    // cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    // grid.x = (XYG0 + block.x - 1) / block.x;
    // BC_Guo2002_GPU_face_xy_load << <grid, block >> > (XYG0, pdf_old_gpu, pitch_old, ntime, Boundary_xy0_gpu, Boundary_xy1_gpu);
    // cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());    
    // grid.x = (YZG0 + block.x - 1) / block.x;
    // BC_Guo2002_GPU_face_yz_load << <grid, block >> > (YZG0, pdf_old_gpu, pitch_old, ntime, Boundary_yz0_gpu, Boundary_yz1_gpu);
    // cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    // #endif
    // // save boundary info for Guo's 2002, Non-equilibrium extrapolation method
    grid.x = (XZG0 + block.x - 1) / block.x;
    BC_Guo2002_GPU_face_xz_save << <grid, block >> > (XZG0, pdf_old_gpu, pitch_old, ntime, Boundary_xz0_gpu, Boundary_xz1_gpu);
    cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    grid.x = (XYG0 + block.x - 1) / block.x;
    BC_Guo2002_GPU_face_xy_save << <grid, block >> > (XYG0, pdf_old_gpu, pitch_old, ntime, Boundary_xy0_gpu, Boundary_xy1_gpu);
    cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    grid.x = (YZG0 + block.x - 1) / block.x;
    BC_Guo2002_GPU_face_yz_save << <grid, block >> > (YZG0, pdf_old_gpu, pitch_old, ntime, Boundary_yz0_gpu, Boundary_yz1_gpu);
    cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());

    //Insertion sphere particle
    //cout <<Time_Step_Sphere_Join<<"\t"<<Num_Sphere<<endl;
    if (ntime >= Time_Step_Sphere_Join  && Num_Sphere > 0){
        if (ntime == Time_Step_Sphere_Join){
            for (int i_Nsphe = 0; i_Nsphe < Num_Sphere; i_Nsphe++){
                calculate_boundary_sphere((T_P)(spheres[i_Nsphe].GridNum_D), spheres[i_Nsphe].Coords, i_Nsphe);                
                Num_Sphere_Boundary[i_Nsphe] = num_Sphere_Boundary; Num_refill_point[i_Nsphe] = 0;
                 
                cudaErrorCheck(cudaMemcpy(spheres_gpu[i_Nsphe].Coords, spheres[i_Nsphe].Coords, mem_size_particle_3D_TP, cudaMemcpyHostToDevice));                
                cudaErrorCheck(cudaMemcpy(spheres_gpu[i_Nsphe].Velos, spheres[i_Nsphe].Velos, mem_size_particle_3D_TP, cudaMemcpyHostToDevice));
                cudaErrorCheck(cudaMemcpy(spheres_gpu[i_Nsphe].AngulVelos, spheres[i_Nsphe].AngulVelos, mem_size_particle_3D_TP, cudaMemcpyHostToDevice));
                cudaErrorCheck(cudaMemcpy(Particles_gpu[i_Nsphe].BC_I, Particles[i_Nsphe].BC_I, mem_particle_BC_max_long, cudaMemcpyHostToDevice));
                cudaErrorCheck(cudaMemcpy(Particles_gpu[i_Nsphe].BC_DV, Particles[i_Nsphe].BC_DV, mem_particle_BC_max_int, cudaMemcpyHostToDevice));
                cudaErrorCheck(cudaMemcpy(Particles_gpu[i_Nsphe].BC_q, Particles[i_Nsphe].BC_q, mem_particle_BC_max_TP, cudaMemcpyHostToDevice));
                // cout <<spheres[i_Nsphe].Coords[0]<<"\t"<<spheres[i_Nsphe].Coords[1]<<"\t"<<spheres[i_Nsphe].Coords[2]<<"\t"<<endl;
                grid.x = (Num_Sphere_Boundary[i_Nsphe] + block.x - 1) / block.x;
                particle_IBB_BC_save_GPU << <grid, block >> > (Num_Sphere_Boundary[i_Nsphe], pdf_old_gpu, pitch_old, ntime, Particles_gpu[i_Nsphe].BC_I, Particles_gpu[i_Nsphe].BC_DV, Particles_gpu[i_Nsphe].BC_fOld);
                cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
            }
        }
        for (int i_Nsphe = 0; i_Nsphe < Num_Sphere; i_Nsphe++){            
            // particle boudary (Take care to consider the order of boundary implementation and collision implementation)
            grid.x = (Num_Sphere_Boundary[i_Nsphe] + block.x - 1) / block.x; 
            I_INT nblock = grid.x;//(Num_Sphere_Boundary + block_Threads_X - 1) / block_Threads_X;
            T_P *Force_block_gpu, *MForce_block_gpu;    
            I_INT mem_force_block_TP = nblock * mem_force_TP;
            cudaErrorCheck(cudaMalloc(&Force_block_gpu, mem_force_block_TP));
            cudaErrorCheck(cudaMalloc(&MForce_block_gpu, mem_force_block_TP));
            cudaErrorCheck(cudaMemset(Force_block_gpu, 0., mem_force_block_TP));
            cudaErrorCheck(cudaMemset(MForce_block_gpu, 0., mem_force_block_TP));            
            particle_IBB_BC_Force_GPU << <grid, block >> > (Num_Sphere_Boundary[i_Nsphe], pdf_old_gpu, Particles_gpu[i_Nsphe].BC_I, Particles_gpu[i_Nsphe].BC_DV, Particles_gpu[i_Nsphe].BC_q, pitch_old, ntime, Force_block_gpu, MForce_block_gpu, Particles_gpu[i_Nsphe].BC_fOld, spheres_gpu[i_Nsphe].Coords, spheres_gpu[i_Nsphe].Velos, spheres_gpu[i_Nsphe].AngulVelos);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
            
            T_P * Force_block = (T_P*)malloc(mem_force_block_TP);
            T_P * MForce_block = (T_P*)malloc(mem_force_block_TP);
            cudaErrorCheck(cudaMemcpy(Force_block, Force_block_gpu, mem_force_block_TP, cudaMemcpyDeviceToHost));
            cudaErrorCheck(cudaMemcpy(MForce_block, MForce_block_gpu, mem_force_block_TP, cudaMemcpyDeviceToHost));

            spheres[i_Nsphe].Force[0] = prc(0.);  spheres[i_Nsphe].Force[1] = prc(0.);  spheres[i_Nsphe].Force[2] = prc(0.);
            spheres[i_Nsphe].MomentOfForce[0] = prc(0.);  spheres[i_Nsphe].MomentOfForce[1] = prc(0.);  spheres[i_Nsphe].MomentOfForce[2] = prc(0.);
            for (int bb=0; bb < nblock; bb++){
                spheres[i_Nsphe].Force[0] = spheres[i_Nsphe].Force[0] + Force_block[bb];
                spheres[i_Nsphe].Force[1] = spheres[i_Nsphe].Force[1] + Force_block[nblock+bb];
                spheres[i_Nsphe].Force[2] = spheres[i_Nsphe].Force[2] + Force_block[2*nblock+bb];
                spheres[i_Nsphe].MomentOfForce[0] = spheres[i_Nsphe].MomentOfForce[0] + MForce_block[bb];
                spheres[i_Nsphe].MomentOfForce[1] = spheres[i_Nsphe].MomentOfForce[1] + MForce_block[nblock+bb];
                spheres[i_Nsphe].MomentOfForce[2] = spheres[i_Nsphe].MomentOfForce[2] + MForce_block[2*nblock+bb];
            }
            for (int i_d=0; i_d < 3; i_d++){
                spheres[i_Nsphe].Force[i_d] = spheres[i_Nsphe].Force[i_d] / dt_LBM;
                spheres[i_Nsphe].MomentOfForce[i_d] = spheres[i_Nsphe].MomentOfForce[i_d] / dt_LBM;
            }
            free(Force_block);
            free(MForce_block);
            cudaErrorCheck(cudaFree(Force_block_gpu));
            cudaErrorCheck(cudaFree(MForce_block_gpu)); 
                       
            updata_information_moving_particle(i_Nsphe);   //update coor of particle, particle boundary point, refill point due to particle moving
            if (Num_Sphere_Boundary[i_Nsphe]<=0)
                cout <<"particle move out the compute domain or error with particle moving"<<endl;
            //// refill process (before collision)
            if (Num_refill_point[i_Nsphe] > 0){
                grid.x = (Num_refill_point[i_Nsphe] + block.x - 1) / block.x;
                Refill_new_point_moving_particles_gpu << <grid, block >> > (Num_refill_point[i_Nsphe], pdf_old_gpu, pitch_old, ntime, Particles_gpu[i_Nsphe].Refill_Point_I, Particles_gpu[i_Nsphe].Refill_Point_DV, spheres_gpu[i_Nsphe].Velos, spheres_gpu[i_Nsphe].AngulVelos, spheres_gpu[i_Nsphe].Coords);
                cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
            }
            grid.x = (Num_Sphere_Boundary[i_Nsphe] + block.x - 1) / block.x;
            
            particle_IBB_BC_save_GPU << <grid, block >> > (Num_Sphere_Boundary[i_Nsphe], pdf_old_gpu, pitch_old, ntime, Particles_gpu[i_Nsphe].BC_I, Particles_gpu[i_Nsphe].BC_DV, Particles_gpu[i_Nsphe].BC_fOld);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
            
        }
        if ( Num_Sphere == 2){
            T_P temp_dist = 0; //distance of two particles
            for (int iii=0; iii<3; iii++){
                temp_dist = temp_dist + (spheres[0].Coords[iii]-spheres[1].Coords[iii])*(spheres[0].Coords[iii]-spheres[1].Coords[iii]);
            }
            if (temp_dist <= (spheres[0].GridNum_D + spheres[1].GridNum_D)*(spheres[0].GridNum_D + spheres[1].GridNum_D)){
                simulation_end_indicator = 1;
            }
        }
    }

    // add point source in the field 
    // grid.x = (1 + block.x - 1) / block.x;
    // acoustic_point_source_equi << <grid, block >> > (1, pdf_old_gpu, pitch_old, ntime);
    
    // streaming and collision
    //block.x = block_Threads_X;
    grid.x = (NGRID0 + block.x - 1) / block.x;
    kernel_OSI_GPU << <grid, block >> > (NGRID0, pdf_old_gpu, pdf_gpu, pitch_old, pitch, ntime);
    cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    /* ************************** one step end***************************************** */
    
    //Swap an address pointer for a old pdf and new pdf    
    swap(pdf_old_gpu, pdf_gpu);
    // T_P * temp_swap;
    // temp_swap = pdf_old_gpu;
    // pdf_old_gpu = pdf_gpu;
    // pdf_gpu = temp_swap;
}

//===================================================================================================================================== =
//------------Update the particle force, speed, position and other related information changes in the CPU----------------------
//===================================================================================================================================== =
void updata_information_moving_particle(int i_Nsphe) {
    T_P velo[3], angulV[3], coord[3];
    for (int i = 0; i < 3; i++){
        velo[i] = spheres[i_Nsphe].Velos[i] + (spheres[i_Nsphe].Force[i]/spheres[i_Nsphe].Mass + body_accelerate[i] + accelerate_particle_force_LBM[i]) * dt_LBM;
        angulV[i] = spheres[i_Nsphe].AngulVelos[i] + spheres[i_Nsphe].MomentOfForce[i] / spheres[i_Nsphe].MomentOfInertia * dt_LBM;
        coord[i] = spheres[i_Nsphe].Coords[i] + 0.5* (spheres[i_Nsphe].Velos[i] + velo[i]) * overc_LBM;//dt_LBM / dx_LBM;   //Position in the index system
    }
    // Resets variables to 0 to prevent data contamination
    // memset(Particles_gpu[i_Nsphe].BC_I, 0, mem_particle_BC_max_long);
    // memset(Particles_gpu[i_Nsphe].BC_DV, 0, mem_particle_BC_max_int);
    // memset(Particles_gpu[i_Nsphe].BC_q, 0, mem_particle_BC_max_TP);
    // memset(Particles_gpu[i_Nsphe].Refill_Point_I, 0, mem_refill_max_long);
    // memset(Particles_gpu[i_Nsphe].Refill_Point_DV, 0, mem_refill_max_int);

    // updata prticle boundary    
    calculate_boundary_sphere_move(spheres[i_Nsphe].GridNum_D, coord, spheres[i_Nsphe].Coords, i_Nsphe);
    Num_Sphere_Boundary[i_Nsphe] = num_Sphere_Boundary; Num_refill_point[i_Nsphe] = num_refill_point;
    for (int i = 0; i < 3; i++){
        spheres[i_Nsphe].Coords[i]           = coord[i];
        spheres[i_Nsphe].Velos[i]       = velo[i];
        spheres[i_Nsphe].AngulVelos[i]  = angulV[i];
    }
    cudaErrorCheck(cudaMemcpy(spheres_gpu[i_Nsphe].Coords, spheres[i_Nsphe].Coords, mem_size_particle_3D_TP, cudaMemcpyHostToDevice));                
    cudaErrorCheck(cudaMemcpy(spheres_gpu[i_Nsphe].Velos, spheres[i_Nsphe].Velos, mem_size_particle_3D_TP, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(spheres_gpu[i_Nsphe].AngulVelos, spheres[i_Nsphe].AngulVelos, mem_size_particle_3D_TP, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(Particles_gpu[i_Nsphe].BC_I, Particles[i_Nsphe].BC_I, mem_particle_BC_max_long, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(Particles_gpu[i_Nsphe].BC_DV, Particles[i_Nsphe].BC_DV, mem_particle_BC_max_int, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(Particles_gpu[i_Nsphe].BC_q, Particles[i_Nsphe].BC_q, mem_particle_BC_max_TP, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(Particles_gpu[i_Nsphe].Refill_Point_I, Particles[i_Nsphe].Refill_Point_I, mem_refill_max_long, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(Particles_gpu[i_Nsphe].Refill_Point_DV, Particles[i_Nsphe].Refill_Point_DV, mem_refill_max_int, cudaMemcpyHostToDevice));

}
