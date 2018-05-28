// Options
// TODO ====
// Reinstate CUDA_EXCLUDES!
//#define CUDA_EXCLUDE 1
#define DEBUG_PRINTS 1

#define R_NO_REMAP 1

#include <R.h>
#include <Rinternals.h>

// Debug macros for cuda functions
#ifndef CUDA_EXCLUDE

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define cublasTry(ans){ cublasAssert( (ans), __FILE__, __LINE__ ); }
inline void cublasAssert( cublasStatus_t stat, const char *file, int line ){
  if( stat == CUBLAS_STATUS_NOT_INITIALIZED ){
    Rf_error( "cuBLAS assert: NOT_INITIALIZED %s %d\n", file, line );
  }else if( stat == CUBLAS_STATUS_ALLOC_FAILED ){
    Rf_error( "cuBLAS assert: ALLOC_FAILED %s %d\n", file, line );
  }else if( stat == CUBLAS_STATUS_INVALID_VALUE ){
    Rf_error( "cuBLAS assert: INVALID_VALUE %s %d\n", file, line );
  }else if( stat == CUBLAS_STATUS_EXECUTION_FAILED ){
    Rf_error( "cuBLAS assert: EXECUTION_FAILED %s %d\n", file, line );
  }else if( stat == CUBLAS_STATUS_INTERNAL_ERROR ){
    Rf_error( "cuBLAS assert: INTERNAL_ERROR %s %d\n", file, line );
  }else{
    Rf_error( "cuBLAS assert: Unmapped error %s %d\n", file, line );
  }
}

#define cudaTry(ans){ cudaAssert( (ans), __FILE__, __LINE__ ); }
inline void cudaAssert( cudaError_t code, const char *file, int line){
  if ( code != cudaSuccess ){
    Rf_error("CUDA assert: %s %s %d\n", cudaGetErrorString( code ), file, line );
  }
}

#endif

// Debug print macros
#ifdef DEBUG_PRINTS
#define debugPrint(ans){ (ans); }
#else
#define debugPrint(ans){ }
#endif
