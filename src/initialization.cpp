#include "Fluid_field_extern.h"
#include "externLib.h"
#include "solver_precision.h"
#include "initialization.h"
#include "preprocessor.h"
#include "Module_extern.h"
#include "utils.h"
#include "index_cpu.h"
#include "IO_fluid.h"
#include "Misc.h"
//#include "Misc.h"


//=====================================================================================================================================
//----------------------initialization basic----------------------
//=====================================================================================================================================
void initialization_basic_fluid() {

    read_parameter_FluidField();
    calculate_basic_particles_info_after_read();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // create folders
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << "Creating directories if not exist" << endl;
    // create directory results
    string filepath = "results/"; // directory path
    bool file_check = fs::is_directory(filepath); // check if directory exists
    if (!file_check) { // If directory does not exist, create one!
        bool file_create_err = fs::create_directory(filepath); // create a directory
        if (!file_create_err) { ERROR("Could not Create results folder !"); } // check that the directory was created successfully
    }
    // create directory results/pdf_data/
    filepath = "results/pdf_data/"; // directory path
    file_check = fs::is_directory(filepath); // check if directory exists
    if (!file_check) { // If directory does not exist, create one!
        bool file_create_err = fs::create_directory(filepath); // create a directory
        if (!file_create_err) { ERROR("Could not Create results/pdf_data folder !"); } // check that the directory was created successfully
    }
    // create directory results/pdf_data/2rd_backup/
    filepath = "results/pdf_data/2rd_backup/"; // directory path
    file_check = fs::is_directory(filepath); // check if directory exists
    if (!file_check) { // If directory does not exist, create one!
        bool file_create_err = fs::create_directory(filepath); // create a directory
        if (!file_create_err) { ERROR("Could not Create results/pdf_data/2rd_backup folder !"); } // check that the directory was created successfully
    }
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
  
    cout << "**************************** Processing geometry *****************************" << endl;
    cout << "Single time step displacement in an array with different discrete velocity directions" << endl;
    
    I_INT Length_temp[NDIR] = { 0, 1, -1, NXG1, -NXG1, NXG1*NYG1, -NXG1*NYG1,        //0-6
                    NXG1+1, NXG1-1, -NXG1+1, -NXG1-1,                              //7-10
                    NXG1*NYG1+1, NXG1*NYG1-1, -NXG1*NYG1+1, -NXG1*NYG1-1,       //11-14
                    NXG1*NYG1+NXG1, NXG1*NYG1-NXG1, -NXG1*NYG1+NXG1, -NXG1*NYG1-NXG1};    //15-18
    for (int i_f=0; i_f<NDIR; i_f++){
        Length[i_f] = Length_temp[i_f];
    }

    cout << "........." << endl;

    cout << "************************** End Processing geometry ***************************" << endl;
    cout << endl;

    cout << "******************** Processing fluid and flow info **********************" << endl;
    
    /* initialize memory size */
    initMemSize();
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //allocate fluid memory
    MemAllocate_fluid(1);
    //allocate particle memory
    MemAllocate_particle(1);
    // get the parameters used for LBM model
    initLBMPara();

    cout << "******************** End processing fluid and flow info **********************" << endl;

}
//===================================================================
//------------- parameters initialize for simulation model -------------
//=================================================================== 
void initLBMPara() {
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // parameters 
    
    if (KinematicViscosity_LBM <= 0  && Reynolds_number>0){
        // calculate fluid viscosity based on the Reynolds number
        KinematicViscosity_LBM = Lid_velocity * (T_P)NXG0 / Reynolds_number;//Lid_velocity * (T_P)(spheres[0].GridNum_D) / Reynolds_number;
    }
    RelaxtionTime = prc(0.5) + KinematicViscosity_LBM / (sound_speed_LBM*sound_speed_LBM * dt_LBM);          // relaxation time in collision, related to Re number
    
    SRT_OverTau = prc(1.) / RelaxtionTime;
    cout << "Fluid Kinematic Viscosity = " << KinematicViscosity_LBM << endl;
    cout << "Fluid relaxation time = " << RelaxtionTime << endl;
    //set the relaxation parameter of MRT
    T_P temp_mrt = prc(1.);
    MRT_S[0] = temp_mrt; MRT_S[3] = temp_mrt; MRT_S[5] = temp_mrt; MRT_S[7] = temp_mrt;   //0, 3, 5, 7
    temp_mrt = SRT_OverTau;
    MRT_S[9] = temp_mrt; MRT_S[11] = temp_mrt; MRT_S[13] = temp_mrt; MRT_S[14] = temp_mrt; MRT_S[15] = temp_mrt; //9, 11, 13, 14, 15
    //MRT_S[9] = MRT_s9; MRT_S[11] = MRT_s9;                //9, 11
    //MRT_S[13] = MRT_s13; MRT_S[14] = MRT_s13; MRT_S[15] = MRT_s13;         //13, 14, 15
    // MRT_S[1] = MRT_s1;              //1
    // MRT_S[2] = MRT_s2;              //2    
    temp_mrt = MRT_S[4]; MRT_S[6] = temp_mrt; MRT_S[8] = temp_mrt;   //4, 6, 8    
    temp_mrt = MRT_S[10]; MRT_S[12] = temp_mrt;         //10, 12    
    temp_mrt = MRT_S[16]; MRT_S[17] = temp_mrt; MRT_S[18] = temp_mrt;         //16, 17, 18
    T_P temp_mrt_M[NDIR*NDIR] = {0};
    for(int i_m1 = 0; i_m1<NDIR; i_m1++)
    {
        for(int i_m2 = 0; i_m2<NDIR; i_m2++){
            temp_mrt_M[i_m1*NDIR + i_m2] = MRT_Trans_M_inverse[i_m1*NDIR + i_m2] * MRT_S[i_m2];
        }
    }
    for(int i_m1 = 0; i_m1<NDIR; i_m1++)
    {
        for(int i_m2 = 0; i_m2<NDIR; i_m2++){
            MRT_Collision_M[i_m1*NDIR + i_m2] = 0;
            for(int i_m3 = 0; i_m3<NDIR; i_m3++){
                MRT_Collision_M[i_m1*NDIR + i_m2] = MRT_Collision_M[i_m1*NDIR + i_m2] + temp_mrt_M[i_m1*NDIR + i_m3] * MRT_Trans_M[i_m3*NDIR + i_m2];
            }
        }
    }
    
}

//=====================================================================================================================================
//----------------------initialization for new simulation - field variables----------------------
//=====================================================================================================================================
void initialization_new_fluid() {
    ntime0 = 1;

    initialization_new_fluid_pdf();

}
// initial distribution functions
void initialization_new_fluid_pdf() {
    int i_f;
    I_INT p_i, p_f;
    I_INT i_z;//, i_y, i_x;
    T_P ux, uy, uz;
    ux = 0.,    uy = 0.;    uz = 0.;
    T_P udotu = ux*ux + uy*uy + uz*uz, edotu;
    T_P feq;
    for (p_i = 0; p_i < NGRID1; p_i++) {
        i_z = p_i/XYG1;
        // i_y = p_i%XYG1 / NXG1;
        // i_x = p_i%XYG1 % NXG1;
        uy = 0;//-Lid_velocity+ (prc(2.) * Lid_velocity*(i_z))/(NZG0+1);
        ux = ux/c_LBM; uy = uy/c_LBM; uz = uz/c_LBM;
        udotu = ux*ux + uy*uy + uz*uz;
        for (i_f=0; i_f < NDIR; i_f++ ){
            edotu = (T_P)ex[i_f] * ux + (T_P)ey[i_f] * uy + (T_P)ez[i_f] * uz;            
            feq = Density * w_equ[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);
            p_f = (I_INT)(i_f) * NGRID1 + p_i;
            pdf[ p_f ] = feq;
        }
    }
    ux = 0.,    uy = 0.;    uz = 0.;
    ux = ux/c_LBM; uy = uy/c_LBM; uz = uz/c_LBM;
    for (p_i = 0; p_i < XZG0; p_i++) {
        i_z = p_i / (NXG0) + 1;
        for (int ii=0; ii < 5; ii++ ){             
            p_f = p_i*5 + ii;
            i_f = BB_xz_Right[ii];            
            udotu = ux*ux + uy*uy + uz*uz;
            edotu = (T_P)ex[i_f] * ux + (T_P)ey[i_f] * uy + (T_P)ez[i_f] * uz;            
            feq = Density * w_equ[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);            
            Boundary_xz0[ p_f ] = feq;
            i_f = Reverse[i_f];
            edotu = (T_P)ex[i_f] * ux + (T_P)ey[i_f] * uy + (T_P)ez[i_f] * uz;            
            feq = Density * w_equ[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);
            Boundary_xz1[ p_f ] = feq;
        }
    }
    ux = 0.,    uy = 0.;    uz = 0.;
    ux = ux/c_LBM; uy = uy/c_LBM; uz = uz/c_LBM;
    for (p_i = 0; p_i < XYG0; p_i++) {
        for (int ii=0; ii < 5; ii++ ){
            p_f = p_i*5 + ii;
            i_f = BB_xy_top[ii];            
            udotu = ux*ux + uy*uy + uz*uz;
            edotu = (T_P)ex[i_f] * ux + (T_P)ey[i_f] * uy + (T_P)ez[i_f] * uz;            
            feq = Density * w_equ[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);            
            Boundary_xy0[ p_f ] = feq;
            i_f = Reverse[i_f];            
            edotu = (T_P)ex[i_f] * ux + (T_P)ey[i_f] * uy + (T_P)ez[i_f] * uz;            
            feq = Density * w_equ[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);
            Boundary_xy1[ p_f ] = feq;
        }
    }
    ux = 0.,    uy = 0.;    uz = 0.;
    ux = ux/c_LBM; uy = uy/c_LBM; uz = uz/c_LBM;
    for (p_i = 0; p_i < YZG0; p_i++) {
        i_z = p_i / (NYG0) + 1;
        for (int ii=0; ii < 5; ii++ ){            
            p_f = p_i*5 + ii;
            i_f = BB_yz_front[ii];
            udotu = ux*ux + uy*uy + uz*uz;
            edotu = (T_P)ex[i_f] * ux + (T_P)ey[i_f] * uy + (T_P)ez[i_f] * uz;            
            feq = Density * w_equ[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);            
            Boundary_yz0[ p_f ] = feq;
            i_f = Reverse[i_f];
            edotu = (T_P)ex[i_f] * ux + (T_P)ey[i_f] * uy + (T_P)ez[i_f] * uz;            
            feq = Density * w_equ[i_f] * (prc(1.) + prc(3.)*edotu + prc(4.5)*edotu*edotu - prc(1.5)* udotu);
            Boundary_yz1[ p_f ] = feq;
        }
    }

}

//calculate basic particles info after read
void calculate_basic_particles_info_after_read(){
    T_P size_Sphere_D;
    for (int ns = 0; ns < 2; ns++){
        size_Sphere_D = spheres[ns].GridNum_D * dx_LBM;
        spheres[ns].Mass = prc(1.)/(prc(6.)) * PI * pow(size_Sphere_D, 3) * spheres[ns].Density;
        spheres[ns].MomentOfInertia = prc(0.1) * spheres[ns].Mass * pow(size_Sphere_D, 2);
    }

    // The initial velocity and angular velocity and force and MomentOfForce of the particle are stored
    for (int ns = 0; ns < 2; ns++){
        for (int id = 0; id <3; id++){
            spheres[ns].Force[id] = prc(0.);
            spheres[ns].MomentOfForce[id] = prc(0.);
            spheres[ns].Velos[id] = prc(0.);
            spheres[ns].AngulVelos[id] = prc(0.);
        }
    }
    // According to the input information of the simulation_control.txt file, the number of particles in the flow field and the time step of joining the flow field are determined
    Num_Sphere = 0;
    for (int ns = 0; ns < 2; ns++){
        if (spheres[ns].GridNum_D > 0){
            Num_Sphere = Num_Sphere+1;
        }
    }
}
//=====================================================================================================================================
//----------------------initialization for old simulation - field variables----------------------
//=====================================================================================================================================
void initialization_old_fluid() {
    cout << "loading continue simulation data ... ";
    /* Open continue simulation file */
    /* File name */
    ostringstream flnm;
    flnm << "id" << setfill('0') << setw(4) << 0; // This line makes the width (4) and fills the values other than the required with (0) 
    
    // pdf
    ostringstream filepath_f;
    filepath_f << "results/pdf_data/pdf." << flnm.str();
    string fns_f = filepath_f.str();
    const char* fnc_f = fns_f.c_str();
    FILE* continue_simu_file_f = fopen(fnc_f, "rb"); // open the file (rb: read binary)
    if (continue_simu_file_f == NULL) {
        ERROR("Continue simulation data not found with exiting program!");
    }
    int nt_temp;
    if (!fread(&nt_temp, sizeof(int), 1, continue_simu_file_f)) { ERROR("Could not load from data continue_simu_file file!"); } ntime0 = nt_temp + 1;
    if (!fread(pdf, mem_size_pdf_TP, 1, continue_simu_file_f)) { ERROR("Could not load from data continue_simu_file file!"); }
    
    fclose(continue_simu_file_f);

    // boundary 
    ostringstream filepath_b;
    filepath_b << "results/pdf_data/boundary." << flnm.str();
    string fns_b = filepath_b.str();
    const char* fnc_b = fns_b.c_str();
    FILE* continue_simu_file_b = fopen(fnc_b, "rb"); // open the file (rb: read binary)
    if (continue_simu_file_b == NULL) {
        ERROR("Continue simulation data not found with exiting program!");
    }
    if (!fread(Boundary_xz0, (XZG0*5*sizeof(T_P)), 1, continue_simu_file_b)) { ERROR("Could not load from data continue_simu_file file!"); }
    if (!fread(Boundary_xz1, (XZG0*5*sizeof(T_P)), 1, continue_simu_file_b)) { ERROR("Could not load from data continue_simu_file file!"); }
    if (!fread(Boundary_xy0, (XYG0*5*sizeof(T_P)), 1, continue_simu_file_b)) { ERROR("Could not load from data continue_simu_file file!"); }
    if (!fread(Boundary_xy1, (XYG0*5*sizeof(T_P)), 1, continue_simu_file_b)) { ERROR("Could not load from data continue_simu_file file!"); }
    if (!fread(Boundary_yz0, (YZG0*5*sizeof(T_P)), 1, continue_simu_file_b)) { ERROR("Could not load from data continue_simu_file file!"); }
    if (!fread(Boundary_yz1, (YZG0*5*sizeof(T_P)), 1, continue_simu_file_b)) { ERROR("Could not load from data continue_simu_file file!"); }    
    fclose(continue_simu_file_b);

    // particle info 
    ostringstream filepath_p;
    filepath_p << "results/pdf_data/particle." << flnm.str();
    string fns_p = filepath_p.str();
    const char* fnc_p = fns_p.c_str();
    FILE* continue_simu_file_p = fopen(fnc_p, "rb"); // open the file (rb: read binary)
    if (continue_simu_file_p == NULL) {
        ERROR("Continue simulation data not found with exiting program!");
    }
    // particles info
    for (int i_p=0; i_p<Num_Sphere; i_p++){
        if (!fread(spheres[i_p].Coords, mem_size_particle_3D_TP, 1, continue_simu_file_p)) { ERROR("Could not load from data continue_simu_file file!"); }
        if (!fread(spheres[i_p].Velos, mem_size_particle_3D_TP, 1, continue_simu_file_p)) { ERROR("Could not load from data continue_simu_file file!"); }
        if (!fread(spheres[i_p].AngulVelos, mem_size_particle_3D_TP, 1, continue_simu_file_p)) { ERROR("Could not load from data continue_simu_file file!"); }
        
        if (!fread(Particles[i_p].BC_I, mem_particle_BC_max_long, 1, continue_simu_file_p)) { ERROR("Could not load from data continue_simu_file file!"); }
        if (!fread(Particles[i_p].BC_DV, mem_particle_BC_max_int, 1, continue_simu_file_p)) { ERROR("Could not load from data continue_simu_file file!"); }
        if (!fread(Particles[i_p].BC_q, mem_particle_BC_max_TP, 1, continue_simu_file_p)) { ERROR("Could not load from data continue_simu_file file!"); }
        if (!fread(Particles[i_p].BC_fOld, mem_particle_BC_max_TP, 1, continue_simu_file_p)) { ERROR("Could not load from data continue_simu_file file!"); }

        if (!fread(&Num_refill_point[i_p], sizeof(I_INT), 1, continue_simu_file_p)) { ERROR("Could not load from data continue_simu_file file!"); }
        if (Num_refill_point[i_p]>0){
            // Index of the new fluid point due to the particle boundary moving
            if (!fread(Particles[i_p].Refill_Point_I, mem_refill_max_long, 1, continue_simu_file_p)) { ERROR("Could not load from data continue_simu_file file!"); }
            // Index of the direction of discrete velocity for Refill process
            if (!fread(Particles[i_p].Refill_Point_DV, mem_refill_max_int, 1, continue_simu_file_p)) { ERROR("Could not load from data continue_simu_file file!"); }            
        }
        
    }

    fclose(continue_simu_file_p);

    cout << "Complete" << endl;

}

//=====================================================================================================================================
//----------------------memory allocate / deallocate----------------------
//=====================================================================================================================================
//************* fluid flow related memory allocate/deallocate ******************************
void MemAllocate_fluid(int flag) {
    int FLAG = flag;
    if (FLAG == 1) {
        Velocity_ux = (T_P*)calloc(num_cells_s1_TP, sizeof(T_P));
        Velocity_uy = (T_P*)calloc(num_cells_s1_TP, sizeof(T_P));
        Velocity_uz = (T_P*)calloc(num_cells_s1_TP, sizeof(T_P));
        Density_rho = (T_P*)calloc(num_cells_s1_TP, sizeof(T_P));

        pdf = (T_P *)calloc(num_size_pdf_TP, sizeof(T_P));//(T_P *)malloc(sizeof(T_P)*NGRID1*NDIR);
        
        Boundary_xz0 = (T_P*)calloc(XZG0*5, sizeof(T_P));
        Boundary_xz1 = (T_P*)calloc(XZG0*5, sizeof(T_P)); 
        Boundary_xy0 = (T_P*)calloc(XYG0*5, sizeof(T_P));
        Boundary_xy1 = (T_P*)calloc(XYG0*5, sizeof(T_P));
        Boundary_yz0 = (T_P*)calloc(YZG0*5, sizeof(T_P));
        Boundary_yz1 = (T_P*)calloc(YZG0*5, sizeof(T_P)); 
    }
    else {
        free(Velocity_ux);
        free(Velocity_uy);
        free(Velocity_uz);
        free(Density_rho);
        free(pdf);

        free(Boundary_xz0);
        free(Boundary_xz1);
        free(Boundary_xy0);
        free(Boundary_xy1);
        free(Boundary_yz0);
        free(Boundary_yz1);
    }
}
//************* particle boundary related memory allocate/deallocate ******************************
void MemAllocate_particle(int flag) {
    int FLAG = flag;
    if (FLAG == 1) {
        
        for (int i_p=0; i_p < 2; i_p++){
            Particles[i_p].BC_I = (I_INT*)calloc(num_particle_BC_max, sizeof(I_INT));
            Particles[i_p].BC_DV = (int*)calloc(num_particle_BC_max, sizeof(int));
            Particles[i_p].BC_q = (T_P*)calloc(num_particle_BC_max, sizeof(T_P));
            Particles[i_p].BC_fOld = (T_P*)calloc(num_particle_BC_max, sizeof(T_P));
            Particles[i_p].Refill_Point_I = (I_INT*)calloc(num_refill_max, sizeof(I_INT)); 
            Particles[i_p].Refill_Point_DV = (int*)calloc(num_refill_max, sizeof(int));
        }
    }
    else {
        for (int i_p=0; i_p < 2; i_p++){
            free(Particles[i_p].BC_I );
            free(Particles[i_p].BC_DV );
            free(Particles[i_p].BC_q );
            free(Particles[i_p].BC_fOld );
            free(Particles[i_p].Refill_Point_I );
            free(Particles[i_p].Refill_Point_DV );
        }
    }
}

//------------- initialize memory size -------------
//=================================================================== 
void initMemSize() {

    /* Memory size of arrays */
    num_cells_s1_TP = NGRID1;
    num_size_pdf_TP = NDIR*NGRID1;
    mem_cells_s1_TP = num_cells_s1_TP* sizeof(T_P);
    mem_size_pdf_TP = num_size_pdf_TP* sizeof(T_P);

    num_particle_BC_max = pow( max((T_P)(spheres[0].GridNum_D),(T_P)(spheres[1].GridNum_D)), 2) * 30;
    mem_particle_BC_max_long = num_particle_BC_max * sizeof(I_INT);
    mem_particle_BC_max_int = num_particle_BC_max * sizeof(int);
    mem_particle_BC_max_TP = num_particle_BC_max * sizeof(T_P);

    num_refill_max = num_particle_BC_max / 10;
    mem_refill_max_long = num_refill_max * sizeof(I_INT);
    mem_refill_max_int = num_refill_max * sizeof(int);
    mem_refill_max_TP = num_refill_max * sizeof(T_P);

    mem_force_TP = 3 * sizeof(T_P);
    
    num_size_particle_3D_TP = 3;
    mem_size_particle_3D_TP = num_size_particle_3D_TP * sizeof(T_P);
    num_size_particle_3D_int = 3;
    mem_size_particle_3D_int = num_size_particle_3D_TP * sizeof(int);

    //mem_force_block_TP = ((num_particle_BC_max + block_Threads_X - 1) / block_Threads_X) * mem_force_TP;
}

//=====================================================================================================================================
//----------------------save some basic info that the simulation used----------------------
//=====================================================================================================================================
void save_basic_info_Simulation() {
    // ~~~~~~~~~~~~~~~~~~~some info of computional~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    string File_info = "results/info.txt";
    ofstream file_info(File_info.c_str(), ios_base::out);
    if (file_info.good()) {
        file_info << " Grid information:" << endl;
        file_info << "nxGlobal = " << nxGlobal << " , nyGlobal = " << nyGlobal << " , nzGlobal = " << nzGlobal << endl;
        file_info << "first particle info:"<< "\tThe coordinates of particle center (x,y,z) = (" << spheres[0].Coords[0] << "," << spheres[0].Coords[1] << "," << spheres[0].Coords[2]<<")"
                    << ";\n\t\tDiameter of particle: " << spheres[0].GridNum_D
                    << ";\n\t\tDensity of particle: " << spheres[0].Density 
                    << ";\n\t\tthe mass of particle: " << spheres[0].Mass<< ";\tthe Inertia moment of particle: " << spheres[0].MomentOfInertia << endl;
        file_info << "second particle info:"<< "\tThe coordinates of particle center (x,y,z) = (" << spheres[1].Coords[0] << "," << spheres[1].Coords[1] << "," << spheres[1].Coords[2]<<")"
                    << ";\n\t\tDiameter of particle: " << spheres[1].GridNum_D
                    << ";\n\t\tDensity of particle: " << spheres[1].Density 
                    << ";\n\t\tthe mass of particle: " << spheres[1].Mass<< ";\tthe Inertia moment of particle: " << spheres[1].MomentOfInertia << endl;
                   
        file_info << "Lid_velocity = " << Lid_velocity << " , Reynolds_number = " << Reynolds_number << endl;
        file_info << "KinematicViscosity_LBM = " << KinematicViscosity_LBM  << endl;
        file_info << "nAbsorbingL = " << nAbsorbingL  << endl;
        file_info << "block_Threads_X = " << block_Threads_X << " , block_Threads_Y = " << block_Threads_Y << " , block_Threads_Z = " << block_Threads_Z << endl;
        file_info << "For initial, num_Sphere_Boundary is :" <<num_Sphere_Boundary<< "\t"<<"num_particle_BC_max is :"<<num_particle_BC_max<<endl;
        file_info << "For initial, num_refill_point is :" <<num_refill_point<< "\t"<<"num_refill_max is :"<<num_refill_max<<endl;

        file_info.close();
    }
    else {
        ERROR("Couldn't open the file results/info.txt !");
    }
}
