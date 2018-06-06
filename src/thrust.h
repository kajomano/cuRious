#include "common.h"

extern "C"
#ifdef _WIN32
__declspec( dllimport )
#endif
void cuR_thrust_pow2_cu( float* A_ptr, float* B_ptr, int* dims );

// void cuB_thrust_cmins_cu( float* prod, int* dims, int* quant );
// void cuB_thrust_table_cu( int* quant, int* perm, int* temp_quant, int* dims, int* weights, int* dims_weights );

