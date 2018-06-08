#include "common.h"

extern "C"
#ifdef _WIN32
__declspec( dllimport )
#endif
void cuR_thrust_pow_cu( float* A_ptr, float* B_ptr, int* dims, float pow, cudaStream_t* stream_ptr );

extern "C"
#ifdef _WIN32
__declspec( dllimport )
#endif
void cuR_thrust_cmin_pos_cu( float* A_ptr, int* x_ptr, int* dims, cudaStream_t* stream_ptr );

// void cuB_thrust_table_cu( int* quant, int* perm, int* temp_quant, int* dims, int* weights, int* dims_weights );

