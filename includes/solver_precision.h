#ifndef SOLVER_PRECISION_H
#define SOLVER_PRECISION_H

/* Solver precision */
#define SINGLE_PRECISION (1)
#define DOUBLE_PRECISION (2)
/* Select solver precision: SINGLE_PRECISION / DOUBLE_PRECISION */
#define PRECISION (DOUBLE_PRECISION)		

#if (PRECISION == SINGLE_PRECISION)
#define T_P float
#elif (PRECISION == DOUBLE_PRECISION)
#define T_P double
#endif

/* integer precision */
#define INT_PRECISION (1)
#define LONG_PRECISION (2)
/* Select integer precision: INT_PRECISION / LONG_PRECISION */
#define PRECISION_INTEGER (LONG_PRECISION)	

#if (PRECISION_INTEGER == INT_PRECISION)
#define I_INT int
#elif (PRECISION_INTEGER == LONG_PRECISION)
#define I_INT long long
#endif

#if (PRECISION == SINGLE_PRECISION)
#define prc(x)	x##f
#define pprc(x)	f##x
#elif (PRECISION == DOUBLE_PRECISION)
#define prc(x)	x
#define pprc(x)	x
#endif



#endif