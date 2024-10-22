#ifndef Fluid_field_H
#define Fluid_field_H

#include "Module_extern.h"
#include "Fluid_field_extern.h"

T_P* Velocity_ux, * Velocity_uy, * Velocity_uz, * Density_rho;
T_P* pdf;

T_P RelaxtionTime, Viscosity, KinematicViscosity_LBM;		 // relaxtion time and viscosity for fluid

T_P Lid_velocity;       //lid velocity
T_P Frequency_lbm, AnguFreq_lbm;    // frequency and angular frequency of sound wave in LBM system
T_P Reynolds_number;     //reynolds number
T_P Density;         //initial Density

T_P body_accelerate[3];       //the accelerate of body force in particle in LBM system
T_P accelerate_particle_force_LBM[3];  //the accelerate of particle force (Gravity - buoyancy)in LBM system

// Relaxtion Time tau in SRT
T_P SRT_OverTau;   //  1/tau

/* GPU pointers */
T_P* Velocity_ux_gpu, * Velocity_uy_gpu, * Velocity_uz_gpu, * Density_rho_gpu;
T_P* pdf_gpu, * pdf_old_gpu;

//Spherical basic info

int Num_Sphere;   // the number of sphere in fluid field
int Time_Step_Sphere_Join;  //the time step of the particles joining the flow field

struct Spheres spheres[2];  //two sphere
struct Spheres_gpu spheres_gpu[2];  //two sphere

struct Particles_MEM Particles[2];  //two particles
struct Particles_MEM Particles_gpu[2];  //two sphere

/* Grid points that represent the boundaries of the computational domain */
T_P * Boundary_xz0, * Boundary_xz1;
T_P * Boundary_xz0_gpu, * Boundary_xz1_gpu;
T_P * Boundary_xy0, * Boundary_xy1;
T_P * Boundary_xy0_gpu, * Boundary_xy1_gpu;
T_P * Boundary_yz0, * Boundary_yz1;
T_P * Boundary_yz0_gpu, * Boundary_yz1_gpu;

#endif