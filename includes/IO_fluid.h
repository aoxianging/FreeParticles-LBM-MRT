#ifndef IO_fluid_H
#define IO_fluid_H

#include "Module_extern.h"
#include "utils.h"

//=========================================================================================================================== =
//----------------------Read input parameters----------------------
//=========================================================================================================================== =
void read_parameter_FluidField();

//=========================================================================================================================== =
//----------------------save data ----------------------
//=========================================================================================================================== =
// ******************************* save distribution function data for continue_simulation*************************************
void save_old_fluid(int save_option);
// ******************************* save data - macro variables *************************************
void save_full_field_plt(int nt); 
void save_full_field_vtk(int nt); 
// ******************************* save the force, velocity, coords of particles and so on *************************************
void save_sphere_info(chrono::steady_clock::duration td_particles);

//===================================================================
//---------------------- Print a scalar array to a bindary file ----------------------
//=================================================================== 
 
template <typename T>
void printScalarBinary_gh(T* scalar_array, const string name, I_INT mem_size, const string path = "results/out3.field_data/") {
	// Get the name of the passed array
	ostringstream filepath;
	filepath << path << name << ".bin";
	string fns = filepath.str();
	const char* fnc = fns.c_str();
	FILE* binary_file = fopen(fnc, "wb+"); // open the file for the first time (create the file)
	if (binary_file == NULL) { ERROR("Could not create scalar binary file!"); }
	I_INT file_size = mem_size;
	fwrite(scalar_array, file_size, 1, binary_file);
	fclose(binary_file);
}



#endif