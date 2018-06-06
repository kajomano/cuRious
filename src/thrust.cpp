// #include "thrust.h"
//
// #define R_NO_REMAP 1
//
// #include <R.h>
// #include <Rinternals.h>
//
// #include <cuda.h>
// #include <cuda_runtime_api.h>
//
// #define cudaTry(ans){ if( cudaAssert( (ans), __FILE__, __LINE__ ) ) return R_NilValue; }
// inline bool cudaAssert( cudaError_t code, const char *file, int line){
//   if (code != cudaSuccess){
//     Rprintf("cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
//     return true;
//   }
//   return false;
// }
//
// extern "C"
// SEXP cuB_thrust_pow2_csum( SEXP cent_r, SEXP dims_r, SEXP norm_r, SEXP handle_r ) {
//   float* cent = (float*)R_ExternalPtrAddr( cent_r );
//   int*   dims = INTEGER( dims_r );
//   float* norm = (float*)R_ExternalPtrAddr( norm_r );
//   cublasHandle_t* handle = (cublasHandle_t*)R_ExternalPtrAddr( handle_r );
//
//   cuB_thrust_pow2_csum_cublas( cent, dims, norm, handle );
//
//   // Return something that is not null
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuB_thrust_cmins( SEXP prod_r, SEXP dims_r, SEXP quant_r ) {
//   float* prod  = (float*)R_ExternalPtrAddr( prod_r );
//   int*   dims  = INTEGER( dims_r );
//   int*   quant = (int*)R_ExternalPtrAddr( quant_r );
//
//   cuB_thrust_cmins_cu( prod, dims, quant );
//
//   // Return something that is not null
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
//
// extern "C"
// SEXP cuB_thrust_table( SEXP quant_r, SEXP perm_r, SEXP temp_quant_r, SEXP dims_r, SEXP weights_r, SEXP dims_weights_r ) {
//   int* quant        = (int*)R_ExternalPtrAddr( quant_r );
//   int* perm         = (int*)R_ExternalPtrAddr( perm_r );
//   int* temp_quant   = (int*)R_ExternalPtrAddr( temp_quant_r );
//   int* dims         = INTEGER( dims_r );
//
//   int* weights      = (int*)R_ExternalPtrAddr( weights_r );
//   int* dims_weights = INTEGER( dims_weights_r );
//
//   cuB_thrust_table_cu( quant, perm, temp_quant, dims, weights, dims_weights );
//
//   // Return something that is not null
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
