#ifndef INDEX_CPU_H
#define INDEX_CPU_H
#include "Module_extern.h"

/* Total number of array nodes after adding the ghost nodes */
// zero ghost layers
#define NXG0	(nxGlobal)
#define NYG0	(nyGlobal)
#define NZG0	(nzGlobal)
// one ghost layer
#define NXG1	(nxGlobal + 2)
#define NYG1	(nyGlobal + 2)
#define NZG1	(nzGlobal + 2)

// the length of grid
#define LXG	((double)(1.)/(double)nxGlobal)
#define LYG	((double)(1.)/(double)nyGlobal)
#define LZG	((double)(1.)/(double)nzGlobal)

/* Total number of grid nodes */
#define NGRID1	(NXG1 * NYG1 * NZG1)
/* Total number of domain grid nodes */
#define NGRID0	(NXG0 * NYG0 * NZG0)

/* the number of grids on different faces of the domain (interior point) */
#define XYG0	(NXG0*NYG0)			
#define XZG0	(NXG0*NZG0)
#define YZG0	(NYG0*NZG0)
/* the number of grids on different faces of the domain (one ghost layer) */
#define XYG1	(NXG1*NYG1)			
#define XZG1	(NXG1*NZG1)
#define YZG1	(NYG1*NZG1)


/* indexing functions for arrays */
// scalar array with one ghost layer
#define p_index(x,y,z) ((x) + NXG1 * ((y) + NYG1 * (z)))

// Calculates the position of the index in the array after the corresponding time step
//#define p_step_index(p, i_f) (((p -Length[i_f] * ntime)<0) ? ((p -Length[i_f] * ntime)%NGRID1 + NGRID1) : ((p -Length[i_f] * ntime)%NGRID1))

#define p_step_index(p_step) (((p_step)%NGRID1 + NGRID1) % NGRID1)

#endif