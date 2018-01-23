// Options
#define CUDA_EXCLUDE 1
//#define DEBUG_PRINTS 1

#define R_NO_REMAP 1

#include <R.h>
#include <Rinternals.h>

// Debug macros for cuda functions
#ifndef CUDA_EXCLUDE

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define cublasTry(ans){ if( cublasAssert( (ans), __FILE__, __LINE__ ) ) return R_NilValue; }
bool cublasAssert( cublasStatus_t stat, const char *file, int line){
  if( stat == CUBLAS_STATUS_SUCCESS ){
    return false;
  }else if( stat == CUBLAS_STATUS_NOT_INITIALIZED ){
    Rprintf( "cublas assert: NOT_INITIALIZED %s %d\n", file, line);
  }else if( stat == CUBLAS_STATUS_ALLOC_FAILED ){
    Rprintf( "cublas assert: ALLOC_FAILED %s %d\n", file, line);
  }else if( stat == CUBLAS_STATUS_INVALID_VALUE ){
    Rprintf( "cublas assert: INVALID_VALUE %s %d\n", file, line);
  }else if( stat == CUBLAS_STATUS_EXECUTION_FAILED ){
    Rprintf( "cublas assert: EXECUTION_FAILED %s %d\n", file, line);
  }else if( stat == CUBLAS_STATUS_INTERNAL_ERROR ){
    Rprintf( "cublas assert: INTERNAL_ERROR %s %d\n", file, line);
  }else{
    Rprintf( "cublas assert: Unmapped error %s %d\n", file, line);
  }
  return true;
}

#define cudaTry(ans){ if( cudaAssert( (ans), __FILE__, __LINE__ ) ) return R_NilValue; }
bool cudaAssert( cudaError_t code, const char *file, int line){
  if (code != cudaSuccess){
    Rprintf("cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    return true;
  }
  return false;
}

#define cudaDo(ans){ (ans); }

#else
#define cublasTry(ans){ }
#define cudaTry(ans){ }
#define cudaDo(ans){ }
#endif

// Debug print macros
#ifdef DEBUG_PRINTS
#define debugPrint(ans){ (ans); }
#else
#define debugPrint(ans){ }
#endif
