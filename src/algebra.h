#include "common.h"

#ifndef CUDA_EXCLUDE

#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C"
void cuR_alg_saxpy_cu( float* tens_x, float* tens_y, int l, float al, cudaStream_t* stream );

#endif
