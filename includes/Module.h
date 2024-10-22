#ifndef MODULE_H
#define MODULE_H
#include "preprocessor.h"
#include "externLib.h"
#include "solver_precision.h"
#include "Module_extern.h"
#include "index_cpu.h"
#include <limits>

T_P PI = prc(3.14159265358979323846);
#if(PRECISION == SINGLE_PRECISION)
T_P eps = numeric_limits<float>::epsilon();
//T_P eps = prc(1.110223025e-8);
#elif (PRECISION == DOUBLE_PRECISION)
//T_P eps = prc(1.110223025e-16);
T_P eps = numeric_limits<float>::epsilon();
#endif

T_P convergence_criteria;

string job_status;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~input commands ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  Initial fluid distribution option: #  1 - Lid-driven cavity flow  #  0 or else - error! 
int initial_fluid_distribution_option;

// 1 - simulation ends; 0 - initial value(exceed max time step); 2 - simulation not finished, but save dataand exit program; 3 - simulation failed
int simulation_end_indicator;

// output field data precision(simulation is always double precision) : 0 - single precision; 1 - double precision
int output_fieldData_precision_cmd;

// # Whether to allow continue simulation: 0 - no continue simulation; 1 - allow continue simulation; 2-  Always continue simulation
int allow_continue_simulation;

// 1 - simulation ends; 0 - initial value(exceed max time step); 2 - simulation not finished, but save dataand exit program; 3 - simulation failed
extern int simulation_end_indicator;
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~input commands end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~domain and geometry ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
I_INT nxGlobal, nyGlobal, nzGlobal;                                   // full grid : 1 to n ? Global
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Absorbing Layers for Nonreflecting Boundary Conditions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
I_INT nAbsorbingL;                                   // number of Absorbing Leyers
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~sphere particle boundary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
I_INT Num_Sphere_Boundary[2] = {0};
I_INT Num_refill_point[2] = {0};   //for moving boundary
I_INT num_Sphere_Boundary = 0;
I_INT num_refill_point = 0;   //for moving boundary

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~lattice ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

T_P dx_LBM = prc(1.); //the size of grid step
T_P dt_LBM = prc(1.); //the size of time step
T_P c_LBM = dx_LBM/dt_LBM; //c_LBM = dx_LBM/dt_LBM
T_P overc_LBM = prc(1.)/c_LBM; //overc_LBM = prc(1.)/c_LBM
T_P const_sound_LBM = prc(1.) / sqrt(3.);   //the const parameter for sound speed in LBM 
T_P sound_speed_LBM = const_sound_LBM * c_LBM;   //sound speed in LBM system

// streaming direction - D3Q19
int ex[NDIR] =      { 0, 1, -1, 0,  0, 0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0 };
int ey[NDIR] =      { 0, 0,  0, 1, -1, 0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1 };
int ez[NDIR] =      { 0, 0,  0, 0,  0, 1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1 };
int Reverse[NDIR] = { 0, 2,  1, 4,  3, 6,  5, 10,  9,  8,  7, 14, 13, 12, 11, 18, 17, 16, 15 }; // reverse directions

// The index of velocity direction required to bounce on the boundary
int BB_xy_top[5]   = { 5, 11, 12, 15, 16}; 
int BB_yz_front[5] = { 1,  7,  9, 11, 13}; 
int BB_xz_Right[5] = { 3,  7,  8, 15, 17}; 

// Single time step displacement in an array with different discrete velocity directions
    // I_INT  Length[NDIR] = { 0, NXG1, -NXG1, -1, 1, NXG1*NYG1, -NXG1*NYG1,                                           //0-6
    //                 NXG1-1, NXG1+1, -NXG1-1, -NXG1+1, NXG1*NYG1+NXG1, -NXG1*NYG1+NXG1,                         //7-12
    //                 NXG1*NYG1-NXG1, -NXG1*NYG1-NXG1, NXG1*NYG1-1, -NXG1*NYG1-1, NXG1*NYG1+1, -NXG1*NYG1+1};    //13-18

I_INT  Length[NDIR] = { 0, 1, -1, NXG1, -NXG1, NXG1*NYG1, -NXG1*NYG1,        //0-6
                    NXG1+1, NXG1-1, -NXG1+1, -NXG1-1,                              //7-10
                    NXG1*NYG1+1, NXG1*NYG1-1, -NXG1*NYG1+1, -NXG1*NYG1-1,       //11-14
                    NXG1*NYG1+NXG1, NXG1*NYG1-NXG1, -NXG1*NYG1+NXG1, -NXG1*NYG1-NXG1};    //15-18

// MRT
// For the D3Q19 model, the transformation matrix M
int MRT_Trans_M[NDIR*NDIR]  = {  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
                               -30, -11, -11, -11, -11, -11, -11,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,
                                12,  -4,  -4,  -4,  -4,  -4,  -4,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
                                 0,   1,  -1,   0,   0,   0,   0,   1,  -1,   1,  -1,   1,  -1,   1,  -1,   0,   0,   0,   0,
                                 0,  -4,   4,   0,   0,   0,   0,   1,  -1,   1,  -1,   1,  -1,   1,  -1,   0,   0,   0,   0,
                                 0,   0,   0,   1,  -1,   0,   0,   1,   1,  -1,  -1,   0,   0,   0,   0,   1,  -1,   1,  -1,
                                 0,   0,   0,  -4,   4,   0,   0,   1,   1,  -1,  -1,   0,   0,   0,   0,   1,  -1,   1,  -1,
                                 0,   0,   0,   0,   0,   1,  -1,   0,   0,   0,   0,   1,   1,  -1,  -1,   1,   1,  -1,  -1,
                                 0,   0,   0,   0,   0,  -4,   4,   0,   0,   0,   0,   1,   1,  -1,  -1,   1,   1,  -1,  -1,
                                 0,   2,   2,  -1,  -1,  -1,  -1,   1,   1,   1,   1,   1,   1,   1,   1,  -2,  -2,  -2,  -2,
                                 0,  -4,  -4,   2,   2,   2,   2,   1,   1,   1,   1,   1,   1,   1,   1,  -2,  -2,  -2,  -2,
                                 0,   0,   0,   1,   1,  -1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,  -1,   0,   0,   0,   0,
                                 0,   0,   0,  -2,  -2,   2,   2,   1,   1,   1,   1,  -1,  -1,  -1,  -1,   0,   0,   0,   0,
                                 0,   0,   0,   0,   0,   0,   0,   1,  -1,  -1,   1,   0,   0,   0,   0,   0,   0,   0,   0,
                                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,  -1,  -1,   1,
                                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,  -1,  -1,   1,   0,   0,   0,   0,
                                 0,   0,   0,   0,   0,   0,   0,   1,  -1,   1,  -1,  -1,   1,  -1,   1,   0,   0,   0,   0,
                                 0,   0,   0,   0,   0,   0,   0,  -1,  -1,   1,   1,   0,   0,   0,   0,   1,  -1,   1,  -1,
                                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,  -1,  -1,  -1,  -1,   1,   1};    
// the inverse of the transformation matrix M //It is calculated by Mathematica
T_P MRT_Trans_M_inverse[NDIR*NDIR]  = {prc(1.)/prc(19.),-(prc(5.)/prc(399.)),prc(1.)/21,prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),
                    prc(1.)/prc(19.),-(prc(11.)/2394),-(prc(1.)/prc(63.)),prc(1.)/prc(10.),-(prc(1.)/prc(10.)),prc(0.),prc(0.),prc(0.),prc(0.),prc(1.)/prc(18.),-(prc(1.)/prc(18.)),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),
                    prc(1.)/prc(19.),-(prc(11.)/2394),-(prc(1.)/prc(63.)),-(prc(1.)/prc(10.)),prc(1.)/prc(10.),prc(0.),prc(0.),prc(0.),prc(0.),prc(1.)/prc(18.),-(prc(1.)/prc(18.)),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),
                    prc(1.)/prc(19.),-(prc(11.)/2394),-(prc(1.)/prc(63.)),prc(0.),prc(0.),prc(1.)/prc(10.),-(prc(1.)/prc(10.)),prc(0.),prc(0.),-(prc(1.)/prc(36.)),prc(1.)/prc(36.),prc(1.)/prc(12.),-(prc(1.)/prc(12.)),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),
                    prc(1.)/prc(19.),-(prc(11.)/2394),-(prc(1.)/prc(63.)),prc(0.),prc(0.),-(prc(1.)/prc(10.)),prc(1.)/prc(10.),prc(0.),prc(0.),-(prc(1.)/prc(36.)),prc(1.)/prc(36.),prc(1.)/prc(12.),-(prc(1.)/prc(12.)),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),
                    prc(1.)/prc(19.),-(prc(11.)/2394),-(prc(1.)/prc(63.)),prc(0.),prc(0.),prc(0.),prc(0.),prc(1.)/prc(10.),-(prc(1.)/prc(10.)),-(prc(1.)/prc(36.)),prc(1.)/prc(36.),-(prc(1.)/prc(12.)),prc(1.)/prc(12.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),
                    prc(1.)/prc(19.),-(prc(11.)/2394),-(prc(1.)/prc(63.)),prc(0.),prc(0.),prc(0.),prc(0.),-(prc(1.)/prc(10.)),prc(1.)/prc(10.),-(prc(1.)/prc(36.)),prc(1.)/prc(36.),-(prc(1.)/prc(12.)),prc(1.)/prc(12.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),prc(0.),
                    prc(1.)/prc(19.),prc(4.)/prc(1197.),prc(1.)/prc(252.),prc(1.)/prc(10.),prc(1.)/prc(40.),prc(1.)/prc(10.),prc(1.)/prc(40.),prc(0.),prc(0.),prc(1.)/prc(36.),prc(1.)/prc(72.),prc(1.)/prc(12.),prc(1.)/prc(24.),prc(1.)/prc(4.),prc(0.),prc(0.),prc(1.)/prc(8.),-(prc(1.)/prc(8.)),prc(0.),
                    prc(1.)/prc(19.),prc(4.)/prc(1197.),prc(1.)/prc(252.),-(prc(1.)/prc(10.)),-(prc(1.)/prc(40.)),prc(1.)/prc(10.),prc(1.)/prc(40.),prc(0.),prc(0.),prc(1.)/prc(36.),prc(1.)/prc(72.),prc(1.)/prc(12.),prc(1.)/prc(24.),-(prc(1.)/prc(4.)),prc(0.),prc(0.),-(prc(1.)/prc(8.)),-(prc(1.)/prc(8.)),prc(0.),
                    prc(1.)/prc(19.),prc(4.)/prc(1197.),prc(1.)/prc(252.),prc(1.)/prc(10.),prc(1.)/prc(40.),-(prc(1.)/prc(10.)),-(prc(1.)/prc(40.)),prc(0.),prc(0.),prc(1.)/prc(36.),prc(1.)/prc(72.),prc(1.)/prc(12.),prc(1.)/prc(24.),-(prc(1.)/prc(4.)),prc(0.),prc(0.),prc(1.)/prc(8.),prc(1.)/prc(8.),prc(0.),
                    prc(1.)/prc(19.),prc(4.)/prc(1197.),prc(1.)/prc(252.),-(prc(1.)/prc(10.)),-(prc(1.)/prc(40.)),-(prc(1.)/prc(10.)),-(prc(1.)/prc(40.)),prc(0.),prc(0.),prc(1.)/prc(36.),prc(1.)/prc(72.),prc(1.)/prc(12.),prc(1.)/prc(24.),prc(1.)/prc(4.),prc(0.),prc(0.),-(prc(1.)/prc(8.)),prc(1.)/prc(8.),prc(0.),
                    prc(1.)/prc(19.),prc(4.)/prc(1197.),prc(1.)/prc(252.),prc(1.)/prc(10.),prc(1.)/prc(40.),prc(0.),prc(0.),prc(1.)/prc(10.),prc(1.)/prc(40.),prc(1.)/prc(36.),prc(1.)/prc(72.),-(prc(1.)/prc(12.)),-(prc(1.)/prc(24.)),prc(0.),prc(0.),prc(1.)/prc(4.),-(prc(1.)/prc(8.)),prc(0.),prc(1.)/prc(8.),
                    prc(1.)/prc(19.),prc(4.)/prc(1197.),prc(1.)/prc(252.),-(prc(1.)/prc(10.)),-(prc(1.)/prc(40.)),prc(0.),prc(0.),prc(1.)/prc(10.),prc(1.)/prc(40.),prc(1.)/prc(36.),prc(1.)/prc(72.),-(prc(1.)/prc(12.)),-(prc(1.)/prc(24.)),prc(0.),prc(0.),-(prc(1.)/prc(4.)),prc(1.)/prc(8.),prc(0.),prc(1.)/prc(8.),
                    prc(1.)/prc(19.),prc(4.)/prc(1197.),prc(1.)/prc(252.),prc(1.)/prc(10.),prc(1.)/prc(40.),prc(0.),prc(0.),-(prc(1.)/prc(10.)),-(prc(1.)/prc(40.)),prc(1.)/prc(36.),prc(1.)/prc(72.),-(prc(1.)/prc(12.)),-(prc(1.)/prc(24.)),prc(0.),prc(0.),-(prc(1.)/prc(4.)),-(prc(1.)/prc(8.)),prc(0.),-(prc(1.)/prc(8.)),
                    prc(1.)/prc(19.),prc(4.)/prc(1197.),prc(1.)/prc(252.),-(prc(1.)/prc(10.)),-(prc(1.)/prc(40.)),prc(0.),prc(0.),-(prc(1.)/prc(10.)),-(prc(1.)/prc(40.)),prc(1.)/prc(36.),prc(1.)/prc(72.),-(prc(1.)/prc(12.)),-(prc(1.)/prc(24.)),prc(0.),prc(0.),prc(1.)/prc(4.),prc(1.)/prc(8.),prc(0.),-(prc(1.)/prc(8.)),
                    prc(1.)/prc(19.),prc(4.)/prc(1197.),prc(1.)/prc(252.),prc(0.),prc(0.),prc(1.)/prc(10.),prc(1.)/prc(40.),prc(1.)/prc(10.),prc(1.)/prc(40.),-(prc(1.)/prc(18.)),-(prc(1.)/prc(36.)),prc(0.),prc(0.),prc(0.),prc(1.)/prc(4.),prc(0.),prc(0.),prc(1.)/prc(8.),-(prc(1.)/prc(8.)),
                    prc(1.)/prc(19.),prc(4.)/prc(1197.),prc(1.)/prc(252.),prc(0.),prc(0.),-(prc(1.)/prc(10.)),-(prc(1.)/prc(40.)),prc(1.)/prc(10.),prc(1.)/prc(40.),-(prc(1.)/prc(18.)),-(prc(1.)/prc(36.)),prc(0.),prc(0.),prc(0.),-(prc(1.)/prc(4.)),prc(0.),prc(0.),-(prc(1.)/prc(8.)),-(prc(1.)/prc(8.)),
                    prc(1.)/prc(19.),prc(4.)/prc(1197.),prc(1.)/prc(252.),prc(0.),prc(0.),prc(1.)/prc(10.),prc(1.)/prc(40.),-(prc(1.)/prc(10.)),-(prc(1.)/prc(40.)),-(prc(1.)/prc(18.)),-(prc(1.)/prc(36.)),prc(0.),prc(0.),prc(0.),-(prc(1.)/prc(4.)),prc(0.),prc(0.),prc(1.)/prc(8.),prc(1.)/prc(8.),
                    prc(1.)/prc(19.),prc(4.)/prc(1197.),prc(1.)/prc(252.),prc(0.),prc(0.),-(prc(1.)/prc(10.)),-(prc(1.)/prc(40.)),-(prc(1.)/prc(10.)),-(prc(1.)/prc(40.)),-(prc(1.)/prc(18.)),-(prc(1.)/prc(36.)),prc(0.),prc(0.),prc(0.),prc(1.)/prc(4.),prc(0.),prc(0.),-(prc(1.)/prc(8.)),prc(1.)/prc(8.)};
 //MRT_Trans_M_inverse*47880
 int MRT_Trans_M_inverse_int[NDIR*NDIR] = {2520, -600, 2280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    2520, -220, -760, 4788, -4788, 0, 0, 0, 0, 2660, -2660, 0, 0, 0, 0, 0, 0, 0, 0,
                    2520, -220, -760, -4788, 4788, 0, 0, 0, 0, 2660, -2660, 0, 0, 0, 0, 0, 0, 0, 0,
                    2520, -220, -760, 0, 0, 4788, -4788, 0, 0, -1330, 1330, 3990, -3990, 0, 0, 0, 0, 0, 0,
                    2520, -220, -760, 0, 0, -4788, 4788, 0, 0, -1330, 1330, 3990, -3990, 0, 0, 0, 0, 0, 0,
                    2520, -220, -760, 0, 0, 0, 0, 4788, -4788, -1330, 1330, -3990, 3990, 0, 0, 0, 0, 0, 0,
                    2520, -220, -760, 0, 0, 0, 0, -4788, 4788, -1330, 1330, -3990, 3990, 0, 0, 0, 0, 0, 0,
                    2520, 160, 190, 4788, 1197, 4788, 1197, 0, 0, 1330, 665, 3990, 1995, 11970, 0, 0, 5985, -5985, 0,
                    2520, 160, 190, -4788, -1197, 4788, 1197, 0, 0, 1330, 665, 3990, 1995, -11970, 0, 0, -5985, -5985, 0,
                    2520, 160, 190, 4788, 1197, -4788, -1197, 0, 0, 1330, 665, 3990, 1995, -11970, 0, 0, 5985, 5985, 0,
                    2520, 160, 190, -4788, -1197, -4788, -1197, 0, 0, 1330, 665, 3990, 1995, 11970, 0, 0, -5985, 5985, 0,
                    2520, 160, 190, 4788, 1197, 0, 0, 4788, 1197, 1330, 665, -3990, -1995, 0, 0, 11970, -5985, 0, 5985,
                    2520, 160, 190, -4788, -1197, 0, 0, 4788, 1197, 1330, 665, -3990, -1995, 0, 0, -11970, 5985, 0, 5985,
                    2520, 160, 190, 4788, 1197, 0, 0, -4788, -1197, 1330, 665, -3990, -1995, 0, 0, -11970, -5985, 0, -5985,
                    2520, 160, 190, -4788, -1197, 0, 0, -4788, -1197, 1330, 665, -3990, -1995, 0, 0, 11970, 5985, 0, -5985,
                    2520, 160, 190, 0, 0, 4788, 1197, 4788, 1197, -2660, -1330, 0, 0, 0, 11970, 0, 0, 5985, -5985,
                    2520, 160, 190, 0, 0, -4788, -1197, 4788, 1197, -2660, -1330, 0, 0, 0, -11970, 0, 0, -5985, -5985,
                    2520, 160, 190, 0, 0, 4788, 1197, -4788, -1197, -2660, -1330, 0, 0, 0, -11970, 0, 0, 5985, 5985,
                    2520, 160, 190, 0, 0, -4788, -1197, -4788, -1197, -2660, -1330, 0, 0, 0, 11970, 0, 0, -5985, 5985};
T_P MRT_C_inverse_int = prc(1.)/47880;
// The collision matrix of MRT
T_P MRT_Collision_M[NDIR*NDIR] = {0};
// The relaxation parameter of MRT
T_P MRT_S[NDIR] = {0};
// T_P MRT_s1 = prc(1.19);
// T_P MRT_s2 = prc(1.4);
// T_P MRT_s4 = prc(1.2);
// T_P MRT_s10 = prc(1.4);
// T_P MRT_s16 = prc(1.98);

// D3Q19 MODEL
T_P w_equ[NDIR] = {
    prc(1.) / prc(3.),
    prc(1.) / prc(18.), prc(1.) / prc(18.), prc(1.) / prc(18.), prc(1.) / prc(18.), prc(1.) / prc(18.), prc(1.) / prc(18.),
    prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.),
    prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.)
};
T_P w_equ_0 = prc(1.) / prc(3.);
T_P w_equ_1 = prc(1.) / prc(18.);
T_P w_equ_2 = prc(1.) / prc(36.);
T_P la_vel_norm_i = prc(1.) / prc(sqrt)(prc(2.)); // account for the diagonal length

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~unsorted ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// timers
int ntime0, ntime, ntime_max, ntime_max_benchmark, ntime_macro, ntime_visual, ntime_pdf, ntime_relaxation, ntime_particles_info;
int ntime_check_save_checkpoint_data, ntime_animation;
T_P checkpoint_save_timer, checkpoint_2rd_save_timer, simulation_duration_timer;
int ntime_clock_sum, ntime_display_steps, ntime_monitor, ntime_monitor_profile, ntime_monitor_profile_ratio;
int num_slice;   // how many monitoring slices(evenly divided through y axis)
T_P d_sa_vol, sa_vol, d_vol_animation, d_vol_detail, d_vol_monitor, d_vol_monitor_prof;
int wallclock_timer_status_sum;   // wall clock time reach desired time for each MPI process
double wallclock_pdfsave_timer;  // save PDF data for restart simulation based on wall clock timer


double memory_gpu; // total amount of memory needed on GPU

char empty_char[6] = "empty";


/* domain and memory sizes */
I_INT num_cells_s1_TP;
I_INT num_size_pdf_TP;
I_INT mem_cells_s1_TP;
I_INT mem_size_pdf_TP;

/* particle boundary domain and memory sizes */
I_INT num_particle_BC_max;  // Represents the maximum number of grid points needed for particle boundaries
I_INT mem_particle_BC_max_long;
I_INT mem_particle_BC_max_int;
I_INT mem_particle_BC_max_TP;

I_INT num_refill_max;  // Represents the maximum number of new grid points due to the particle moving
I_INT mem_refill_max_long;
I_INT mem_refill_max_int;
I_INT mem_refill_max_TP;

I_INT mem_force_TP;

/* The space required to store the particle's coordinate position information  */
I_INT num_size_particle_3D_TP;
I_INT mem_size_particle_3D_TP;
/* The space required to Stores the particle's int information */
I_INT num_size_particle_3D_int;
I_INT mem_size_particle_3D_int;
/* The space required to Stores the particle's I_INT information */
I_INT num_size_particle_3D_I_INT;
I_INT mem_size_particle_3D_I_INT;

size_t pitch, pitch_old;

/* CUDA threads in a block */
int block_Threads_X, block_Threads_Y, block_Threads_Z;
#endif