//============================================================================= =
//PROGRAM: AcousticField-Particle-LBM - C++/CUDA version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2023, 12, 08  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
// This is a C++/CUDA version of the LBM solver for acoustic field.
// Author: TianZhuangZhuang (田壮壮)
// reference: Mahmoud Sedahmed's code, from Github repository: https://github.com/lanl/MF-LBM

#include "externLib.h"						/* C++/CUDA standard libraries */
#include "solver_precision.h"				/* Solver precision */
#include "preprocessor.h"					/* MRT parameters set */
#include "utils.h"							/* Helper functions (ERROR, read)*/
#include "Module_extern.h"					
#include "Module.h"							
#include "Fluid_field_extern.h"
#include "Fluid_field.h"
#include "initialization.h"				
#include "Misc.h"
#include "IO_fluid.h"
#include "MemAllocate_gpu.cuh"
//#include "Global_Variables_extern_gpu.cuh"
#include "main_iteration_gpu.cuh"
#include "index_cpu.h"
#include "utils_gpu.cuh"

int main(int argc , char *argv[])
{
    cout << "==============================================================================" << endl;
    cout << "This is a C++/CUDA version of the LBM solver for acoustic field." << endl;
    cout << "Author: TianZhuangZhuang" << endl;  
    cout << "==============================================================================" << endl;

    // indicator used to save extra backup checkpoint(pdf) data
	int save_checkpoint_data_indicator, save_2rd_checkpoint_data_indicator;
	int counter_checkpoint_save, counter_2rd_checkpoint_save;

    //################################################################################################################################
	//													    	 Preparation
	//################################################################################################################################
	
    simulation_end_indicator = 0;   // default 0 :continue, others: break
	save_checkpoint_data_indicator = 0;          // default 0, saving data 1, after saving data 0
	save_2rd_checkpoint_data_indicator = 0;       // default 0, saving data 1, after saving data 0

	// initial value 1; after each checkpoint data saving, counter_checkpoint_save = counter_checkpoint_save + 1
	counter_checkpoint_save = 1;
	// initial value 1; after each checkpoint data saving, counter_2rd_checkpoint_save = counter_2rd_checkpoint_save + 1
	counter_2rd_checkpoint_save = 1;

    cout << " " << endl;
#if (PRECISION == SINGLE_PRECISION)
	cout << "Solver precision: Single precision" << endl;
#elif (PRECISION == DOUBLE_PRECISION)
	cout << "Solver precision: Double precision" << endl;
#endif

	cout << " " << endl;
	cout << "============================ Initialization ==============================>>" << endl;
    
    initialization_basic_fluid();
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    
    if ((job_status == "new_simulation" && allow_continue_simulation == 1) || allow_continue_simulation == 0) {
        initialization_new_fluid();
    }else if ((job_status == "continue_simulation" && allow_continue_simulation == 1) || allow_continue_simulation == 2) {
        initialization_old_fluid();
    }
    else {
        ERROR("Wrong simlation status! Exiting program!");
    }
	
    save_basic_info_Simulation(); //save some basic infomation
    cout << "=============================== Initialization ends =============================<<" << endl;

    ntime = ntime0;

	cout << "***************************** Initialization - GPU **********************************" << endl;
	initialization_GPU();
	/* copy constant data to GPU */
	copyConstantData();
	cout << "************************** Initialization ends - GPU ********************************" << endl;

    //################################################################################################################################
	//													Main loop Starts
	//################################################################################################################################
    //save_full_field_vtk(ntime);
    // for (int i=0; i<288; i++){
    //     cout << Particl_BC_DV[i]<<endl;
    // }
    cout << "************************** Entering main loop *********************************" << endl;
    chrono::steady_clock::time_point ts_console = chrono::steady_clock::now(); // will be reset every specific interval
    chrono::steady_clock::time_point ts2_console = chrono::steady_clock::now(); // mark the start of the loop
    chrono::steady_clock::time_point ts3_console = chrono::steady_clock::now(); // will be reset every specific interval
    chrono::steady_clock::time_point tend1 = chrono::steady_clock::now();  // used for particles info
    chrono::steady_clock::duration td_particles = tend1 - ts_console; // result in nanoseconds

    for (ntime = ntime0; ntime < ntime0 + ntime_max; ntime++) {
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main kernel in CUDA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        main_iteration_kernel_GPU();
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main kernel in CUDA end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~output ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		// ---------- - computation time---------- -
        if (ntime % ntime_particles_info == 0){
            tend1 = chrono::steady_clock::now();
            td_particles = tend1 - ts_console; // result in nanoseconds
            save_sphere_info(td_particles);
            ts_console = chrono::steady_clock::now();
        }
        
        if (ntime % ntime_visual == 0) {
            // Output fluid macro variables
            get_macro_vars_cpu();
            save_full_field_vtk(ntime);

            // output run time now
            chrono::steady_clock::time_point tend3 = chrono::steady_clock::now();
            chrono::steady_clock::duration td3_console = tend3 - ts3_console; // result in nanoseconds
            double code_speed = static_cast<double>(NGRID0) / static_cast<double>(td3_console.count() * 1e-3); // 1e-3 >>> since the count in nano (1e-9) and result in (million) (1e6)
            code_speed *= static_cast<double>(ntime_visual);
            cout << "ntime =	" << ntime << endl;
            cout << "simulation speed: " << code_speed << " MLUPS" << endl;
            ts3_console = chrono::steady_clock::now();

            chrono::steady_clock::time_point tend2 = chrono::steady_clock::now();
			chrono::steady_clock::duration td2_console = tend2 - ts2_console; // result in nanoseconds
            double duration_console = td2_console.count() * 1e-9; // result in seconds
            duration_console *= prc(2.77777777777e-4);   // second to hour  1 / 3600
            cout << "simulation has run	" << duration_console << "	hours" << endl ;

        }

        if (ntime % ntime_check_save_checkpoint_data == 0) { // frequency to check continue simulation data saving
			chrono::steady_clock::time_point tend2 = chrono::steady_clock::now();
			chrono::steady_clock::duration td2_console = tend2 - ts2_console; // result in nanoseconds
			double duration_console = td2_console.count() * 1e-9; // result in seconds
			duration_console *= prc(2.77777777777e-4);   // second to hour  1 / 3600

			cout << "simulation has run	" << duration_console << "	hours" << endl;
			if (duration_console >= simulation_duration_timer) {
				cout << "Time to save continue simulation data and exit the program!" << endl;
				simulation_end_indicator = 2;
			}
			if (duration_console >= counter_checkpoint_save * checkpoint_save_timer) {
				cout << "Time to save continue simulation data!" << endl;
				save_checkpoint_data_indicator = 1;
				counter_checkpoint_save = counter_checkpoint_save + 1;
			}
			if (duration_console >= counter_2rd_checkpoint_save * checkpoint_2rd_save_timer) {
				cout << "Time to save secondary continue simulation data!" << endl;
				save_2rd_checkpoint_data_indicator = 1;
				counter_2rd_checkpoint_save = counter_2rd_checkpoint_save + 1;
			}

			if (save_checkpoint_data_indicator == 1) {
				save_old_fluid(0);    // save pdf to the default location when option is 0
				save_checkpoint_data_indicator = 0;   //reset status
			}
			if (save_2rd_checkpoint_data_indicator == 1) {
				save_old_fluid(1);    // save pdf to the secondary backup location when option is 1
				save_2rd_checkpoint_data_indicator = 0;   //reset status
			}
		}

        // ~~~~~~~~~~~~~~~~~~SAVE PDF DATA for restarting simulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		if (simulation_end_indicator > 0) { break; }

    }
    cout << "************************** Exiting main iteration *********************************" << endl;


    /* print total simulation speed */
    chrono::steady_clock::time_point te2_console = chrono::steady_clock::now(); // mark the end of the loop
    chrono::steady_clock::duration td2_e_console = te2_console - ts2_console; // result in nanoseconds
    double duration_console_total = td2_e_console.count() * 1e-9; // result in seconds
    double code_speed = static_cast<double>(nxGlobal * nyGlobal * nzGlobal) * static_cast<double>((ntime - ntime0) - 1) / (1e6 * duration_console_total);
    cout << "Code speed:\t" << code_speed << " MLUPS" << endl;

    ntime = ntime - 1;  //dial back ntime
    save_old_fluid(0);
    switch (simulation_end_indicator){
        case 0:{
            cout << "Simulation ended after	" << ntime << " iterations which reached the maximum time step!" << endl;
				
            string filepath = "input/job_status.txt";
            ofstream job_stat(filepath.c_str(), ios_base::out);
            if (job_stat.good()) {
                job_stat << "simulation_reached_max_step";
                job_stat.close();
            }
            else {
                ERROR("Could not open input/job_status.txt");
            }
        }
        break;
        case 1:{
            cout << "Simulation finished after	" << ntime << " iterations which particles collision!" << endl;

            string filepath = "input/job_status.txt";
            ofstream job_stat(filepath.c_str(), ios_base::out);
            if (job_stat.good()) {
                job_stat << "simulation_end_particcles_collision";
                job_stat.close();
            }
            else {
                ERROR("Could not open input/job_status.txt");
            }
        }
        break;
        case 2:{
            cout << "Simulation finished after	" << ntime << " iterations which particles collision!" << endl;

            string filepath = "input/job_status.txt";
            ofstream job_stat(filepath.c_str(), ios_base::out);
            if (job_stat.good()) {
                job_stat << "simulation_end_walltime_limit";
                job_stat.close();
            }
            else {
                ERROR("Could not open input/job_status.txt");
            }
        }
        break;
        default :{
            string filepath = "input/job_status.txt";
            ofstream job_stat(filepath.c_str(), ios_base::out);
            if (job_stat.good()) {
                job_stat << "simulation_end_with_error";
                job_stat.close();
            }
            else {
                ERROR("Could not open input/job_status.txt");
            }
        }
        break;
    }

    if (ntime % ntime_visual != 0){ // Output fluid macro variables final
        get_macro_vars_cpu();
        save_full_field_vtk(ntime);
    }    

    /* free the memory  */
    MemAllocate_fluid(2);
    MemAllocate_particle(2);

    MemAllocate_fluid_GPU(2);
    MemAllocate_particle_GPU(2);

    cout << endl << "Code Finished " << endl;
    return 0;
}
