#include "common_cuda.h"

#ifndef CUDA_EXCLUDE

extern "C"
#ifdef _WIN32
__declspec( dllexport )
#endif
  void* cuR_thrust_allocator_create_cu();

extern "C"
#ifdef _WIN32
__declspec( dllexport )
#endif
  void cuR_thrust_allocator_destroy_cu( void* allocator_ptr );

extern "C"
#ifdef _WIN32
__declspec( dllimport )
#endif
  void cuR_thrust_pow_cu( float* A_ptr, float* B_ptr, int* dims, float pow, void* allocator_ptr, cudaStream_t* stream_ptr = NULL );

extern "C"
#ifdef _WIN32
__declspec( dllimport )
#endif
  void cuR_thrust_cmin_pos_cu( float* A_ptr, int* x_ptr, int* dims, void* allocator_ptr, cudaStream_t* stream_ptr = NULL );

extern "C"
#ifdef _WIN32
__declspec( dllimport )
#endif
  void cuR_thrust_table_cu( int* x_ptr, int* p_ptr, int* w_ptr, int* s_ptr, int* x_dims, int* w_dims, int x_span_off, void* allocator_ptr, cudaStream_t* stream_ptr = NULL );

#endif
