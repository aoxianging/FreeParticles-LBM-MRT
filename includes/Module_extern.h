#ifndef MODULE_EXTERN_H
#define MODULE_EXTERN_H

#include "preprocessor.h"
#include "externLib.h"
#include "solver_precision.h"

extern T_P PI;
extern T_P eps;
extern T_P convergence_criteria;

// job status : new_simulation; continue_simulation; simulation_done; simulation_failed
extern string job_status;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~input commands ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  Initial fluid distribution option: #  1 - Lid-driven cavity flow  #  0 or else - error!
extern int initial_fluid_distribution_option;

// 1 - simulation ends; 0 - initial value(exceed max time step); 2 - simulation not finished, but save dataand exit program; 3 - simulation failed
extern int simulation_end_indicator;

// output field data precision(simulation is always double precision) : 0 - single precision; 1 - double precision
extern int output_fieldData_precision_cmd;

// # Whether to allow continue simulation: 0 - no continue simulation; 1 - allow continue simulation; 2-  Always continue simulation
extern int allow_continue_simulation;

// 1 - simulation ends; 0 - initial value(exceed max time step); 2 - simulation not finished, but save dataand exit program; 3 - simulation failed
extern int simulation_end_indicator;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~input commands ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~domain and geometry ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
extern I_INT nxGlobal, nyGlobal, nzGlobal;                                   // full grid : 1 to n ? Global
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Absorbing Layers for Nonreflecting Boundary Conditions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
extern I_INT nAbsorbingL;                                   // number of Absorbing Leyers
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~sphere particle boundary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
extern I_INT Num_Sphere_Boundary[2];
extern I_INT Num_refill_point[2];   //for moving boundary
extern I_INT num_Sphere_Boundary;
extern I_INT num_refill_point;   //for moving boundary

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~lattice ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// number of discrete velocity directions used in the D2Q9 model
#define NDIR 19
// streaming direction - D3Q19
extern T_P dx_LBM; //the size of grid step
extern T_P dt_LBM; //the size of time step
extern T_P c_LBM; //c_LBM = dx_LBM/dt_LBM
extern T_P overc_LBM; //overc_LBM = prc(1.)/c_LBM
extern T_P const_sound_LBM;   //the square of sound speed in lattice unit
extern T_P sound_speed_LBM;   //sound speed in LBM system

extern int ex[NDIR];
extern int ey[NDIR];
extern int ez[NDIR];

extern int Reverse[NDIR]; // reverse directions

// The index of velocity direction required to bounce on the boundary
extern int BB_xy_top[5]   ; // The axis direction need in the first one
extern int BB_yz_front[5] ; // The axis direction need in the first one
extern int BB_xz_Right[5] ; // The axis direction need in the first one

extern I_INT Length[NDIR];  // Single time step displacement in an array with different discrete velocity directions

// MRT
extern int MRT_Trans_M[NDIR*NDIR]; // For the D3Q19 model, the transformation matrix M
extern T_P MRT_Trans_M_inverse[NDIR*NDIR]; // the inverse of the transformation matrix M
extern int MRT_Trans_M_inverse_int[NDIR*NDIR];   //MRT_Trans_M_inverse*47880
extern T_P MRT_C_inverse_int;
extern T_P MRT_Collision_M[NDIR*NDIR]; // The collision matrix of MRT
extern T_P MRT_S[NDIR]; // The relaxation parameter of MRT
// extern T_P MRT_s1;
// extern T_P MRT_s2;
// extern T_P MRT_s4;
// extern T_P MRT_s10;
// extern T_P MRT_s16;

// D3Q19 MODEL
extern T_P w_equ[NDIR];
extern T_P w_equ_0;
extern T_P w_equ_1;
extern T_P w_equ_2;
extern T_P la_vel_norm_i; // account for the diagonal length

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~unsorted ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// timers
extern int ntime0, ntime, ntime_max, ntime_max_benchmark, ntime_macro, ntime_visual, ntime_pdf, ntime_relaxation, ntime_particles_info;
extern int ntime_check_save_checkpoint_data, ntime_animation;
extern T_P checkpoint_save_timer, checkpoint_2rd_save_timer, simulation_duration_timer;
extern int ntime_clock_sum, ntime_display_steps, ntime_monitor, ntime_monitor_profile, ntime_monitor_profile_ratio;
extern int num_slice;   // how many monitoring slices(evenly divided through y axis)
extern T_P d_sa_vol, sa_vol, d_vol_animation, d_vol_detail, d_vol_monitor, d_vol_monitor_prof;
extern int wallclock_timer_status_sum;   
extern double wallclock_pdfsave_timer;  // save PDF data for restart simulation based on wall clock timer

extern double memory_gpu; // total amount of memory needed on GPU

extern char empty_char[6];

/* domain and memory sizes */
extern I_INT num_cells_s1_TP;
extern I_INT num_size_pdf_TP;
extern I_INT mem_cells_s1_TP;
extern I_INT mem_size_pdf_TP;
/* particle boundary domain and memory sizes */
extern I_INT num_particle_BC_max;  // Represents the maximum number of grid points needed for particle boundaries
extern I_INT mem_particle_BC_max_long;
extern I_INT mem_particle_BC_max_int;
extern I_INT mem_particle_BC_max_TP;
extern I_INT num_refill_max;  // Represents the maximum number of new grid points due to the particle moving
extern I_INT mem_refill_max_long;
extern I_INT mem_refill_max_int;
extern I_INT mem_refill_max_TP;

extern I_INT mem_force_TP;
/* The space required to store the particle's T_P information  */
extern I_INT num_size_particle_3D_TP;
extern I_INT mem_size_particle_3D_TP;
/* The space required to Stores the particle's int information */
extern I_INT num_size_particle_3D_int;
extern I_INT mem_size_particle_3D_int;
/* The space required to Stores the particle's I_INT information */
extern I_INT num_size_particle_3D_I_INT;
extern I_INT mem_size_particle_3D_I_INT;

extern size_t pitch, pitch_old;

/* CUDA threads in a block */
extern int block_Threads_X, block_Threads_Y, block_Threads_Z;
#endif


