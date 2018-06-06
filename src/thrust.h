#include <cublas_v2.h>

extern "C"{
  void cuB_thrust_pow2_csum_cublas( float* cent, int* dims, float* norm, cublasHandle_t* handle );
  void cuB_thrust_cmins_cu( float* prod, int* dims, int* quant );
  void cuB_thrust_table_cu( int* quant, int* perm, int* temp_quant, int* dims, int* weights, int* dims_weights );
}
