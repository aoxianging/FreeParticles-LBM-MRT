#include "externLib.h"
#include "solver_precision.h"
#include "preprocessor.h"
#include "Module_extern.h"
#include "Fluid_field_extern.h"
#include "utils.h"
#include "index_cpu.h"
#include "Misc.h"
#include "IO_fluid.h"
#include "MemAllocate_gpu.cuh"

//=========================================================================================================================== =
//----------------------Read input parameters----------------------
//=========================================================================================================================== =
void read_parameter_FluidField() {
    cout << "Checking simulation status ... " << endl;
    string FILE_job_status = "input/job_status.txt";
    ifstream file_job_status(FILE_job_status.c_str());
    bool ALIVE = file_job_status.good();
    if (!ALIVE) {
        ERROR("Missing job status file! Exiting program!");
    }
    /* open the simulation status file */
    string job_status_s((istreambuf_iterator<char>(file_job_status)), istreambuf_iterator<char>());
    job_status = job_status_s;
    file_job_status.close();
    
    /* open the simulation control file */
    string FILE_simulation_control = "input/simulation_control.txt";
    ifstream file_simulation_control(FILE_simulation_control.c_str());
    ALIVE = file_simulation_control.good();
    if (!ALIVE) {
        ERROR("Missing simulation control file! Exiting program!");
    }
    cout << " " << endl;
    cout << "****************** Reading in parameters from control file *******************" << endl;
    cout << "........." << endl;
    READ_INT(FILE_simulation_control.c_str(), initial_fluid_distribution_option);
    cout << "initial_fluid_distribution_option: " << initial_fluid_distribution_option << endl;
    cout << "---------------------------" << endl;
    READ_T_P(FILE_simulation_control.c_str(), convergence_criteria); // 
    cout << "convergence_criteria:  " << convergence_criteria << endl;
    cout << "---------------------------" << endl;
    
    READ_INT(FILE_simulation_control.c_str(), output_fieldData_precision_cmd); // 
    cout << "output_fieldData_precision_cmd:    " << output_fieldData_precision_cmd << endl;
    READ_INT(FILE_simulation_control.c_str(), allow_continue_simulation); // Whether to allow continue simulation
    cout << "allow_continue_simulation:    " << allow_continue_simulation << endl;

    READ_T_P(FILE_simulation_control.c_str(), dx_LBM);  // grid size in LBM
    cout << "dx_LBM:   " << dx_LBM << endl;
    READ_T_P(FILE_simulation_control.c_str(), c_LBM);  // the ratio of grid size and time step in LBM
    cout << "c_LBM:   " << c_LBM << endl;
    overc_LBM = prc(1.)/c_LBM;
    sound_speed_LBM = const_sound_LBM * c_LBM;   //sound speed in LBM system   
    dt_LBM = dx_LBM * overc_LBM;  // READ_T_P(FILE_simulation_control.c_str(), dt_LBM);  // time step in LBM
    cout << "dt_LBM:   " << dt_LBM << endl;
    

    T_P MRT_s1;  READ_T_P(FILE_simulation_control.c_str(), MRT_s1); MRT_S[1] = MRT_s1;  // MRT_s1
    cout << "MRT_s1:   " << MRT_S[1] << endl;
    T_P MRT_s2;  READ_T_P(FILE_simulation_control.c_str(), MRT_s2); MRT_S[2] = MRT_s2;  // MRT_s1
    cout << "MRT_s2:   " << MRT_S[2] << endl;
    T_P MRT_s4;  READ_T_P(FILE_simulation_control.c_str(), MRT_s4); MRT_S[4] = MRT_s4;  // MRT_s1
    cout << "MRT_s4:   " << MRT_S[4] << endl;
    T_P MRT_s10; READ_T_P(FILE_simulation_control.c_str(), MRT_s10); MRT_S[10] = MRT_s10;  // MRT_s1
    cout << "MRT_s10:   " << MRT_S[10] << endl;
    T_P MRT_s16; READ_T_P(FILE_simulation_control.c_str(), MRT_s16); MRT_S[16] = MRT_s16;  // MRT_s1
    cout << "MRT_s16:   " << MRT_S[16] << endl;
    
    READ_INTEGER(FILE_simulation_control.c_str(), nxGlobal); //  total nodes for the whole domain
    cout << "nxGlobal:   " << nxGlobal << endl;
    READ_INTEGER(FILE_simulation_control.c_str(), nyGlobal);
    cout << "nyGlobal:   " << nyGlobal << endl;
    READ_INTEGER(FILE_simulation_control.c_str(), nzGlobal);
    cout << "nzGlobal:   " << nzGlobal << endl;
    READ_INTEGER(FILE_simulation_control.c_str(), nAbsorbingL);
    cout << "nAbsorbingL:   " << nAbsorbingL << endl;

    T_P Coor_Sphere_1_x; READ_T_P(FILE_simulation_control.c_str(), Coor_Sphere_1_x); spheres[0].Coords[0] = Coor_Sphere_1_x; // the x position of particle center
    cout << "Coor_Sphere_1_x:   " << spheres[0].Coords[0] << endl;
    T_P Coor_Sphere_1_y; READ_T_P(FILE_simulation_control.c_str(), Coor_Sphere_1_y); spheres[0].Coords[1] = Coor_Sphere_1_y; // the y position of particle center
    cout << "Coor_Sphere_1_y:   " << spheres[0].Coords[1] << endl;
    T_P Coor_Sphere_1_z; READ_T_P(FILE_simulation_control.c_str(), Coor_Sphere_1_z); spheres[0].Coords[2] = Coor_Sphere_1_z; // the z position of particle center
    cout << "Coor_Sphere_1_z:   " << spheres[0].Coords[2] << endl;
    T_P GridNum_Sphere_1_D; READ_T_P(FILE_simulation_control.c_str(), GridNum_Sphere_1_D); spheres[0].GridNum_D = GridNum_Sphere_1_D; // the diameter of particle
    cout << "GridNum_Sphere_1_D:   " << spheres[0].GridNum_D << endl;
    T_P sphere_1_Density; READ_T_P(FILE_simulation_control.c_str(), sphere_1_Density); spheres[0].Density = sphere_1_Density;// Density of particle
    cout << "Density of particle 1: " << spheres[0].Density << endl;
    
    T_P Coor_Sphere_2_x; READ_T_P(FILE_simulation_control.c_str(), Coor_Sphere_2_x); spheres[1].Coords[0] = Coor_Sphere_2_x; // the x position of particle center
    cout << "Coor_Sphere_2_x:   " << spheres[1].Coords[0] << endl;
    T_P Coor_Sphere_2_y; READ_T_P(FILE_simulation_control.c_str(), Coor_Sphere_2_y); spheres[1].Coords[1] = Coor_Sphere_2_y; // the y position of particle center
    cout << "Coor_Sphere_2_y:   " << spheres[1].Coords[1] << endl;
    T_P Coor_Sphere_2_z; READ_T_P(FILE_simulation_control.c_str(), Coor_Sphere_2_z); spheres[1].Coords[2] = Coor_Sphere_2_z; // the z position of particle center
    cout << "Coor_Sphere_2_z:   " << spheres[1].Coords[2] << endl;
    T_P GridNum_Sphere_2_D; READ_T_P(FILE_simulation_control.c_str(), GridNum_Sphere_2_D); spheres[1].GridNum_D = GridNum_Sphere_2_D; // the diameter of particle
    cout << "GridNum_Sphere_2_D:   " << spheres[1].GridNum_D << endl;
    T_P sphere_2_Density; READ_T_P(FILE_simulation_control.c_str(), sphere_2_Density); spheres[1].Density = sphere_2_Density;// Density of particle
    cout << "Density of particle 2: " << spheres[1].Density << endl;

    READ_INT(FILE_simulation_control.c_str(), Time_Step_Sphere_Join); // the time step of the particles joining the flow field
    cout << "Time_Step_Sphere_Join =   " << Time_Step_Sphere_Join << endl;
    

    cout << "---------------------------" << endl;
    READ_T_P(FILE_simulation_control.c_str(), Lid_velocity); // lid_velocity
    cout << "Lid_velocity:   " << Lid_velocity << endl;
    READ_T_P(FILE_simulation_control.c_str(), Frequency_lbm); // frequency of sound wave in LBM system
    cout << "Frequency_lbm:   " << Frequency_lbm << endl;
    AnguFreq_lbm = prc(2.)*PI*Frequency_lbm;  //// angular frequency of sound wave in LBM system
    READ_T_P(FILE_simulation_control.c_str(), Reynolds_number); // Reynolds_number
    cout << "Reynolds_number:   " << Reynolds_number << endl;
    READ_T_P(FILE_simulation_control.c_str(), KinematicViscosity_LBM); // Kinematic Viscosity
    cout << "KinematicViscosity_LBM:   " << KinematicViscosity_LBM << endl;
    READ_T_P(FILE_simulation_control.c_str(), Density); // Density of background flow field
    cout << "Density of background flow field: " << Density << endl;
    
    T_P body_accelerate_x; READ_T_P(FILE_simulation_control.c_str(), body_accelerate_x);  body_accelerate[0] = body_accelerate_x; // body_accelerate_x
    cout << "body_accelerate_x:   " << body_accelerate[0] << endl;
    T_P body_accelerate_y; READ_T_P(FILE_simulation_control.c_str(), body_accelerate_y);  body_accelerate[1] = body_accelerate_y; // body_accelerate_y
    cout << "body_accelerate_y:   " << body_accelerate[1] << endl;
    T_P body_accelerate_z; READ_T_P(FILE_simulation_control.c_str(), body_accelerate_z);  body_accelerate[2] = body_accelerate_z; // body_accelerate_z
    cout << "body_accelerate_z:   " << body_accelerate[2] << endl;

    T_P accelerate_particle_force_LBM_x; READ_T_P(FILE_simulation_control.c_str(), accelerate_particle_force_LBM_x);  accelerate_particle_force_LBM[0] = accelerate_particle_force_LBM_x; // body_accelerate_x
    cout << "accelerate_particle_force_LBM_x:   " << accelerate_particle_force_LBM[0] << endl;
    T_P accelerate_particle_force_LBM_y; READ_T_P(FILE_simulation_control.c_str(), accelerate_particle_force_LBM_y);  accelerate_particle_force_LBM[1] = accelerate_particle_force_LBM_y; // body_accelerate_y
    cout << "accelerate_particle_force_LBM_y:   " << accelerate_particle_force_LBM[1] << endl;
    T_P accelerate_particle_force_LBM_z; READ_T_P(FILE_simulation_control.c_str(), accelerate_particle_force_LBM_z);  accelerate_particle_force_LBM[2] = accelerate_particle_force_LBM_z; // body_accelerate_z
    cout << "accelerate_particle_force_LBM_z:   " << accelerate_particle_force_LBM[2] << endl;

    int max_time_step; READ_INT(FILE_simulation_control.c_str(), max_time_step); ntime_max = max_time_step; // timer: max iterations
    cout << "Max_iterations =   " << ntime_max << endl;    
    READ_INT(FILE_simulation_control.c_str(), ntime_visual); // timer: when to output detailed visualization data
    cout << "ntime_visual =   " << ntime_visual << endl;    
    READ_INT(FILE_simulation_control.c_str(), ntime_animation); // timer: when to output animation data, only particle
    cout << "ntime_animation =   " << ntime_animation << endl;
    READ_INT(FILE_simulation_control.c_str(), ntime_particles_info); // timer: when to output particle info data (i.e. particles coords, velocity, force, and so on)
    cout << "ntime_particles_info =   " << ntime_particles_info << endl;
    
    READ_INT(FILE_simulation_control.c_str(), ntime_check_save_checkpoint_data); // timer: which time step to check output distribution function info data for continue simulation 
    cout << "ntime_check_save_checkpoint_data =   " << ntime_check_save_checkpoint_data << endl;
    READ_T_P(FILE_simulation_control.c_str(), checkpoint_save_timer); // save PDF data for restart simulation based on wall clock timer  (unit hours)
    cout << "Checkpoint_save_timer (wall clock time, hours) =   " << checkpoint_save_timer << endl;
    READ_T_P(FILE_simulation_control.c_str(), checkpoint_2rd_save_timer); //  save secondary PDF data for restart simulation based on wall clock timer  (unit hours)
    cout << "Checkpoint_2rd_save_timer (wall clock time, hours) =   " << checkpoint_2rd_save_timer << endl;
    READ_T_P(FILE_simulation_control.c_str(), simulation_duration_timer); // simulation duration in hours, exit and save simulation afterwards
    cout << "Simulation_duration (wall clock time, hours) =   " << simulation_duration_timer << endl;
    
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), block_Threads_X);
    cout << "CUDA block_Threads_X: " << block_Threads_X << endl;
    READ_INT(FILE_simulation_control.c_str(), block_Threads_Y);
    cout << "CUDA block_Threads_Y: " << block_Threads_Y << endl;
    READ_INT(FILE_simulation_control.c_str(), block_Threads_Z);
    cout << "CUDA block_Threads_Z: " << block_Threads_Z << endl;
    file_simulation_control.close();
    cout << "---------------------------" << endl;
    cout << "************** End reading in parameters from control file *******************" << endl;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ check correctness of input parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cout << "************ Start checking correctness of input parameters ******************" << endl;
    int error_signal = 0;
    if ((job_status == "new_simulation" && allow_continue_simulation == 1) || allow_continue_simulation == 0) {
        cout << "Simulation status is: New Simulation!" << endl;
    }else if ((job_status == "continue_simulation" && allow_continue_simulation == 1) || allow_continue_simulation == 2) {
        cout << "Simulation status is: Continue existing simulation!" << endl;
    }
    else {
        ERROR("Wrong simlation status! Exiting program!");
    }
    
    if (error_signal == 1) {
        ERROR("Exit Program!");
    }
    else {
        cout << "Everything looks good!" << endl;
        cout << "************** End checking correctness of input parameters ******************" << endl;
        cout << endl;
    }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ check correctness of input parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


}

//=========================================================================================================================== =
//----------------------save data ----------------------
//=========================================================================================================================== = 

/* swap byte order (Little endian > Big endian) */
template <typename T>
void SwapEnd(T& var)
{
    char* varArray = reinterpret_cast<char*>(&var);
    for (long i = 0; i < static_cast<long>(sizeof(var) / 2); i++)
        std::swap(varArray[sizeof(var) - 1 - i], varArray[i]);
}
//=========================================================================================================================== =
//----------------------save data ----------------------
//=========================================================================================================================== = 
// ******************************* save distribution function data for continue_simulation*************************************
void save_old_fluid(int save_option) {
    cout << "Saving continue simulation data ... "<<endl;
    // Transfer information from the GPU to the CPU (DeviceToHost)
    copy_old_fluid_DeviceToHost();
    /* Open continue simulation file */
    /* File name */
    ostringstream flnm;
    flnm << "id" << setfill('0') << setw(4) << 0; // This line makes the width (4) and fills the values other than the required with (0)
    ostringstream filepath;
    if (save_option == 0) {
        filepath << "results/pdf_data/";
    }
    else {
        filepath << "results/pdf_data/2rd_backup/";
    }
    
    // pdf
    ostringstream filepath_f;
    filepath_f << filepath.str() << "pdf." << flnm.str();
    string fns_f = filepath_f.str();
    const char* fnc_f = fns_f.c_str();
    FILE* continue_simu_file_f = fopen(fnc_f, "wb+"); // open the file (rb: read binary)
    if (continue_simu_file_f == NULL) {
        ERROR("Could not create continue_simu_file_f file!");
    }
    
    fwrite(&ntime, sizeof(int), 1, continue_simu_file_f);
    fwrite(pdf, mem_size_pdf_TP, 1, continue_simu_file_f);
    
    fclose(continue_simu_file_f);

    // boundary 
    ostringstream filepath_b;
    filepath_b << filepath.str() << "boundary." << flnm.str();
    string fns_b = filepath_b.str();
    const char* fnc_b = fns_b.c_str();
    FILE* continue_simu_file_b = fopen(fnc_b, "wb+"); // open the file (rb: read binary)
    if (continue_simu_file_b == NULL) {
        ERROR("Could not create continue_simu_file_b file!");
    }
    fwrite(Boundary_xz0, (XZG0*5*sizeof(T_P)), 1, continue_simu_file_b);
    fwrite(Boundary_xz1, (XZG0*5*sizeof(T_P)), 1, continue_simu_file_b);
    fwrite(Boundary_xy0, (XYG0*5*sizeof(T_P)), 1, continue_simu_file_b);
    fwrite(Boundary_xy1, (XYG0*5*sizeof(T_P)), 1, continue_simu_file_b);
    fwrite(Boundary_yz0, (YZG0*5*sizeof(T_P)), 1, continue_simu_file_b);
    fwrite(Boundary_yz1, (YZG0*5*sizeof(T_P)), 1, continue_simu_file_b);
    
    fclose(continue_simu_file_b);

    // particle info 
    ostringstream filepath_p;
    filepath_p << filepath.str() << "particle." << flnm.str();
    string fns_p = filepath_p.str();
    const char* fnc_p = fns_p.c_str();
    FILE* continue_simu_file_p = fopen(fnc_p, "wb+"); // open the file (rb: read binary)
    if (continue_simu_file_p == NULL) {
        ERROR("Could not create continue_simu_file_p file!");
    }
    // particles info
    for (int i_p=0; i_p<Num_Sphere; i_p++){
        fwrite(spheres[i_p].Coords, mem_size_particle_3D_TP, 1, continue_simu_file_p);
        fwrite(spheres[i_p].Velos, mem_size_particle_3D_TP, 1, continue_simu_file_p);
        fwrite(spheres[i_p].AngulVelos, mem_size_particle_3D_TP, 1, continue_simu_file_p);

        fwrite(Particles[i_p].BC_I, mem_particle_BC_max_long, 1, continue_simu_file_p);
        fwrite(Particles[i_p].BC_DV, mem_particle_BC_max_int, 1, continue_simu_file_p);
        fwrite(Particles[i_p].BC_q, mem_particle_BC_max_TP, 1, continue_simu_file_p);        
        fwrite(Particles[i_p].BC_fOld, mem_particle_BC_max_TP, 1, continue_simu_file_p);

        fwrite(&Num_refill_point[i_p], sizeof(I_INT), 1, continue_simu_file_p);
        if (Num_refill_point[i_p]>0){
            // Index of the new fluid point due to the particle boundary moving
            fwrite(Particles[i_p].Refill_Point_I, mem_refill_max_long, 1, continue_simu_file_p);
            // Index of the direction of discrete velocity for Refill process
            fwrite(Particles[i_p].Refill_Point_DV, mem_refill_max_int, 1, continue_simu_file_p);
        }
    }
    
    
    fclose(continue_simu_file_p);

    if (save_option == 0) {
        cout << "Saving checkpoint data completed!" << endl;
        string filepath = "input/job_status.txt";
        ofstream job_stat(filepath.c_str(), ios_base::out);
        if (job_stat.good()) {
            job_stat << "continue_simulation";
            job_stat.close();
        }
        else {
            ERROR("Could not open ./job_status.txt");
        }
    }
    if (save_option == 1) {
        cout << "Saving secondary checkpoint data completed!" << endl;
    }
}
// ******************************* save data - macro variables *************************************
// vtk
void save_full_field_vtk(int nt) {  // Full flow field data with tecplot
    cout << "Start to save a flow field data .... ";
    I_INT i, j, k;    
    T_P rho=0, ux=0, uy=0, uz=0;    
    //T_P lx = LXG, ly = LYG, lz = LZG;
    I_INT p_i;
    string fmt;

    // // The flow field inside the particle is set to 0
    // I_INT par_x = NXG0/2 + 1, par_y = NYG0/4 + 1, par_z = NZG0/2 + 1;
    // I_INT edge2 = NXG0/8/2;
    // for (k=par_z-edge2+1; k<=par_z+edge2; k++){
    //     for (j=par_y-edge2+1; j<=par_y+edge2; j++){
    //         for (i=par_x-edge2+1; i<=par_x+edge2; i++){
    //             p_i = p_index(i, j, k);
    //             Density_rho[p_i] = 0;
    //             Velocity_ux[p_i] = 0;  Velocity_uy[p_i] = 0;   Velocity_uz[p_i] = 0;
    //         }
    //     }
    // }

    ostringstream flnm;
    flnm << "full_macro_" << setfill('0') << setw(9) << nt <<".vtk";
    ostringstream filepath;
    filepath << "results/" << flnm.str();
    string fns = filepath.str();
    const char* fnc = fns.c_str();
    ofstream field_vtk;
    field_vtk.open(fnc, ios_base::out | std::ios::binary);

    if (field_vtk.good()) {
        field_vtk << "# vtk DataFile Version 3.0" << endl;
        field_vtk << "vtk output" << endl;
        field_vtk << "BINARY" << endl;
        field_vtk << "DATASET STRUCTURED_POINTS" << endl;
        field_vtk << "DIMENSIONS " << NXG0 << " " << NYG0 << " " << NZG0 << endl;
        field_vtk << "ORIGIN " << 1 << " " << 1 << " " << 1 << endl;
        field_vtk << "SPACING " << 1 << " " << 1 << " " << 1 << endl;
        field_vtk << "POINT_DATA " << NGRID0 << endl;

        if (PRECISION == SINGLE_PRECISION) {
            fmt = "float";
        }
        else if (PRECISION == DOUBLE_PRECISION){
            fmt = "double";
        }else{
            ERROR("error with solver precision in save_full_field_vtk()!");
        }

        field_vtk << "SCALARS " << "rho " << fmt << endl;
        field_vtk << "LOOKUP_TABLE" << " default" << endl;
        for (k = 1; k <= NZG0; k++) {
            for (j = 1; j <= NYG0; j++) {
                for (i = 1; i <= NXG0; i++) {
                    p_i = p_index(i, j, k);
                    rho = Density_rho[p_i];
                    SwapEnd(rho);
                    field_vtk.write(reinterpret_cast<char*>(&rho), sizeof(T_P));
                }
            }
        }
        field_vtk << "SCALARS " << "ux " << fmt << endl;
        field_vtk << "LOOKUP_TABLE" << " default" << endl;
        for (k = 1; k <= NZG0; k++) {
            for (j = 1; j <= NYG0; j++) {
                for (i = 1; i <= NXG0; i++) {
                    p_i = p_index(i, j, k);
                    ux = Velocity_ux[p_i];
                    SwapEnd(ux);
                    field_vtk.write(reinterpret_cast<char*>(&ux), sizeof(T_P));
                }
            }
        }
        field_vtk << "SCALARS " << "uy " << fmt << endl;
        field_vtk << "LOOKUP_TABLE" << " default" << endl;
        for (k = 1; k <= NZG0; k++) {
            for (j = 1; j <= NYG0; j++) {
                for (i = 1; i <= NXG0; i++) {
                    p_i = p_index(i, j, k);
                    uy = Velocity_uy[p_i];
                    SwapEnd(uy);
                    field_vtk.write(reinterpret_cast<char*>(&uy), sizeof(T_P));
                }
            }
        }
        field_vtk << "SCALARS " << "uz " << fmt << endl;
        field_vtk << "LOOKUP_TABLE" << " default" << endl;
        for (k = 1; k <= NZG0; k++) {
            for (j = 1; j <= NYG0; j++) {
                for (i = 1; i <= NXG0; i++) {
                    p_i = p_index(i, j, k);
                    uz = Velocity_uz[p_i];
                    SwapEnd(uz);
                    field_vtk.write(reinterpret_cast<char*>(&uz), sizeof(T_P));
                }
            }
        }
    }
    field_vtk.close();
    
    cout << "complete" << endl;
    
}

// ******************************* save the force, velocity, coords of particles and so on *************************************
void save_sphere_info(chrono::steady_clock::duration td_particles){
    string filepath = "results/time.dat";
    ofstream time_dat;
    if (ntime0 == 1 && ntime == 1) { // open for the first time, new file
        time_dat.open(filepath.c_str(), ios_base::out);
    }
    else { // open during simulation after timesteps, append file
        time_dat.open(filepath.c_str(), ios_base::app);
    }
    if (time_dat.good()) { 
        // chrono::steady_clock::time_point tend1 = chrono::steady_clock::now();
        // chrono::steady_clock::duration td_particles = tend1 - ts_console; // result in nanoseconds
        double duration_console = td_particles.count() * 1e-9; // result in seconds
        time_dat << ntime << "\t" << duration_console << "\t" << endl;  //sec

        time_dat.close();
        // ts_console = chrono::steady_clock::now();
    }
    else {
        ERROR("Could not open results/time.dat");
    }

    // save the info of first sphere change with time
    filepath = "results/time_sphere_1.dat";
    if (ntime0 == 1 && ntime == 1) { // open for the first time, new file
        time_dat.open(filepath.c_str(), ios_base::out);
    }
    else { // open during simulation after timesteps, append file
        time_dat.open(filepath.c_str(), ios_base::app);
    }
    if (time_dat.good()) { 
        // chrono::steady_clock::time_point tend1 = chrono::steady_clock::now();
        // chrono::steady_clock::duration td_particles = tend1 - ts_console; // result in nanoseconds
        time_dat << ntime << "\t" 
        << spheres[0].Force[0]           << "\t"<< spheres[0].Force[1]            << "\t"<< spheres[0].Force[2] << "\t" 
        << spheres[0].MomentOfForce[0]   << "\t"<< spheres[0].MomentOfForce[1]    << "\t"<< spheres[0].MomentOfForce[2] << "\t"
        << spheres[0].Velos[0]      << "\t"<< spheres[0].Velos[1]       << "\t"<< spheres[0].Velos[2] << "\t"
        << spheres[0].AngulVelos[0] << "\t"<< spheres[0].AngulVelos[1]  << "\t"<< spheres[0].AngulVelos[2] << "\t"
        << spheres[0].Coords[0]          << "\t"<< spheres[0].Coords[1]           << "\t"<< spheres[0].Coords[2] << "\t" 
        << Num_refill_point[0] << endl;

        time_dat.close();
        // ts_console = chrono::steady_clock::now();
    }
    else {
        ERROR("Could not open results/time_sphere_1.dat");
    }

    // save the info of second sphere change with time
    filepath = "results/time_sphere_2.dat";
    if (ntime0 == 1 && ntime == 1) { // open for the first time, new file
        time_dat.open(filepath.c_str(), ios_base::out);
    }
    else { // open during simulation after timesteps, append file
        time_dat.open(filepath.c_str(), ios_base::app);
    }
    if (time_dat.good()) { 
        // chrono::steady_clock::time_point tend1 = chrono::steady_clock::now();
        // chrono::steady_clock::duration td_particles = tend1 - ts_console; // result in nanoseconds
        time_dat << ntime << "\t" 
        << spheres[1].Force[0] << "\t" << spheres[1].Force[1] << "\t" << spheres[1].Force[2] << "\t" 
        << spheres[1].MomentOfForce[0] << "\t"<< spheres[1].MomentOfForce[1] << "\t"<< spheres[1].MomentOfForce[2] << "\t"
        << spheres[1].Velos[0] << "\t"<< spheres[1].Velos[1] << "\t"<< spheres[1].Velos[2] << "\t"
        << spheres[1].AngulVelos[0] << "\t"<< spheres[1].AngulVelos[1] << "\t"<< spheres[1].AngulVelos[2] << "\t"
        << spheres[1].Coords[0] << "\t"<< spheres[1].Coords[1] << "\t"<< spheres[1].Coords[2] << "\t" 
        << Num_refill_point[1] << endl;
        
        time_dat.close();
        // ts_console = chrono::steady_clock::now();
    }
    else {
        ERROR("Could not open results/time_sphere_2.dat");
    }
}

// save .plt binary data  (unfinished)
void save_full_field_plt(int nt) {  // Full flow field data with tecplot
    // cout << "Start to save a flow field data .... ";
    // I_INT i, j, k;    
    // T_P rho=0, ux=0, uy=0, uz=0;    
    // //T_P lx = LXG, ly = LYG, lz = LZG;
    // I_INT p_i;
    
    // // cout << "compute macro variables from distribution function .... ";
    // //compute_macro_vars_OSI();
    // // cout << " Complete " << endl;
    // ostringstream flnm;
    // flnm << "full_macro_" << setfill('0') << setw(9) << nt <<".plt";
    // ostringstream filepath;
    // filepath << "results/" << flnm.str();
    // string fns = filepath.str();
    // const char* fnc = fns.c_str();
    // FILE* field_plt = fopen(fnc, "wb+"); // open the file for the first time (create the file)
    // if (field_plt == NULL) { ERROR("Could not create field_plt file!"); }

    // ostringstream zone1, zone2;
    // string strZ1,strZ2;
    // zone1 << "Variables = x,y,z,rho,u,v,w"<< endl;
    // strZ1 = zone1.str();
    // fwrite(strZ1.c_str(), sizeof(strZ1) , strZ1.length(),  field_plt);
    // zone2 << "ZONE I="<< NXG0 << ", J="<<NYG0 <<", K="<< NZG0<< ",F=POINT" << endl;
    // strZ2 = zone2.str();
    // fwrite(strZ2.c_str(), sizeof(strZ2) , strZ2.length(),  field_plt);
    // fwrite(Density_rho, mem_cells_s1_TP, 1, field_plt);
    // fwrite(Velocity_ux, mem_cells_s1_TP, 1, field_plt);
    // fwrite(Velocity_uy, mem_cells_s1_TP, 1, field_plt);
    // fwrite(Velocity_uz, mem_cells_s1_TP, 1, field_plt);

    // // for (k = 1; k <= NZG0; k++) {
    // //     for (j = 1; j <= NYG0; j++) {
    // //         for (i = 1; i <= NXG0; i++) {
    // //             p_i = p_index(i, j, k);
    // //             rho = Density_rho[p_i];
    // //             ux = Velocity_ux[p_i];
    // //             uy = Velocity_uy[p_i];
    // //             uz = Velocity_uz[p_i];                    
    // //             // field_plt << (lx*(i-0.5)) << "\t"<< (ly*(j-0.5))<< "\t"<< (lz*(k-0.5)) << "\t"
    // //             // << rho<<"\t"<< ux<<"\t"<< uy<<"\t"<< uz<<endl;
                
    // //         }
    // //     }
    // // }
    // fclose(field_plt);
    
    // cout << "complete" << endl;
    
}