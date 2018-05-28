#include "common.h"

#ifndef CUDA_EXCLUDE

#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C"
#ifdef _WIN32
__declspec( dllimport )
#endif
void cuR_transfer_device_device_cu(
    void* src_ptr,
    void* dst_ptr,
    const char type,
    int* src_dims,
    int* dst_dims,
    int* dims,
    int* src_perm_ptr,
    int* dst_perm_ptr,
    int src_span_off,
    int dst_span_off,
    cudaStream_t* stream_ptr
);



// extern "C"
// #ifdef _WIN32
// __declspec( dllimport )
// #endif
// void cuR_transfer_device_device_i_cu( int* src,   int* dst,   int* dims, int osrc, int odst, int* csrc, int* cdst, cudaStream_t* stream );
//
// extern "C"
// #ifdef _WIN32
// __declspec( dllimport )
// #endif
// void cuR_transfer_device_device_l_cu( bool* src,  bool* dst,  int* dims, int osrc, int odst, int* csrc, int* cdst, cudaStream_t* stream );

#endif
