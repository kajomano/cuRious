#include <cuda.h>
#include <cuda_runtime_api.h>

#define R_NO_REMAP 1
//#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

#include "debug.h"

// TODO: Maybe just write this whole thing in a macro
#define cudaTry(ans) { cudaAssert((ans), __FILE__, __LINE__); }
void cudaAssert( cudaError_t code, char* file, int line );
