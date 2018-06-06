#include "common.h"

#ifndef CUDA_EXCLUDE

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

#endif
