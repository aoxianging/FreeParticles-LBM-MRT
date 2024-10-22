#ifndef Fluid_field_extern_H
#define Fluid_field_extern_H

#include "Module_extern.h"

extern T_P* Velocity_ux, * Velocity_uy, * Velocity_uz, * Density_rho;
extern T_P* pdf;

extern T_P RelaxtionTime, Viscosity, KinematicViscosity_LBM;		 // relaxtion time and viscosity for fluid

extern T_P Lid_velocity;       //lid velocity
extern T_P Frequency_lbm, AnguFreq_lbm;     // frequency and angular frequency of sound wave in LBM system
extern T_P Reynolds_number;     //reynolds number
extern T_P Density;         //initial Density

extern T_P body_accelerate[3];       //the accelerate of body force in particle in LBM system
extern T_P accelerate_particle_force_LBM[3];  //the accelerate of particle force (Gravity - buoyancy)in LBM system

// Relaxtion Time tau in SRT
extern T_P SRT_OverTau;   //  1/tau

/* GPU pointers */
extern T_P* Velocity_ux_gpu, * Velocity_uy_gpu, * Velocity_uz_gpu, * Density_rho_gpu;
extern T_P* pdf_gpu, * pdf_old_gpu;

// struct used for particle
struct Spheres
{
    T_P Coords[3];             //Spherical coordinates   
    T_P GridNum_D;           // grid number of sphere diameter    
    T_P Density;             //Sphere density
    T_P Mass;                //Sphere mass 
    T_P MomentOfInertia;     //Sphere Moment of Inertia of partilce
    T_P Force[3];            //Sphere Moment of Inertia of partilce
    T_P MomentOfForce[3];    //Sphere Moment of Inertia of partilce
    T_P Velos[3];        //Sphere Velocitys
    T_P AngulVelos[3];   //Sphere Angul Velocitys with center
};
struct Spheres_gpu
{
    T_P * Coords;        //Stores the particle's coordinate position information
    T_P * Velos;         //Stores the particle's velocity information
    T_P * AngulVelos;   //Stores the particle's angular velocity information
};
struct Particles_MEM
{
    I_INT * BC_I;   // Index of the fluid point closest to the particle boundary
    int * BC_DV;     // Index of discrete velocity cross by particle boundary
    T_P * BC_q;      // The ratio of the distance between the boundary point and the nearest neighbor point inside and outside the particle 
    // T_P * Coords;        //Stores the particle's coordinate position information
    // T_P * Velos;         //Stores the particle's velocity information
    // T_P * AngulVelos;   //Stores the particle's angular velocity information
    T_P * BC_fOld;  // used for single-node boundary schemes proposed by Zhao and Yong, 2017 
    I_INT * Refill_Point_I;   // Index of the new fluid point due to the particle boundary moving
    int * Refill_Point_DV;   // Index of the direction of discrete velocity for Refill process
};
//Spherical basic info

extern int Num_Sphere;   // the number of sphere in fluid field
extern int Time_Step_Sphere_Join;  //the time step of the particles joining the flow field

extern struct Spheres spheres[2];  //two sphere
extern struct Spheres_gpu spheres_gpu[2];  //two sphere

extern struct Particles_MEM Particles[2];  //two particles
extern struct Particles_MEM Particles_gpu[2];  //two sphere

/* Grid points that represent the boundaries of the computational domain */
extern T_P * Boundary_xz0, * Boundary_xz1;
extern T_P * Boundary_xz0_gpu, * Boundary_xz1_gpu;
extern T_P * Boundary_xy0, * Boundary_xy1;
extern T_P * Boundary_xy0_gpu, * Boundary_xy1_gpu;
extern T_P * Boundary_yz0, * Boundary_yz1;
extern T_P * Boundary_yz0_gpu, * Boundary_yz1_gpu;
#endif