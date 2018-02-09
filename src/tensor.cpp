#include "common.h"
#include <cstring>

// Level 0 clearing
extern "C"
SEXP cuR_clear_tensor_0_n( SEXP tens_r, SEXP dims_r ){
  int* dims    = INTEGER( dims_r );
  double* tens = REAL( tens_r );

  memset( tens, 0, sizeof(double) * dims[0]*dims[1] );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_clear_tensor_0_i( SEXP tens_r, SEXP dims_r ){
  int* dims = INTEGER( dims_r );
  int* tens = INTEGER( tens_r );

  memset( tens, 0, sizeof(int) * dims[0]*dims[1] );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_clear_tensor_0_l( SEXP tens_r, SEXP dims_r ){
  int* dims = INTEGER( dims_r );
  int* tens = LOGICAL( tens_r );

  memset( tens, 0, sizeof(int) * dims[0]*dims[1] );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// Various memory allocations/deallocations and clearing
// Level 1 ---------------------------------------------------------------------

template <typename t>
void cuR_fin_tensor_1( SEXP tens_r ){
  t* tens = (t*)R_ExternalPtrAddr( tens_r );

  if( tens ){
    debugPrint( Rprintf( "<%p> Finalizing L1 tensor\n", (void*)tens ) );

    delete[] tens;
    R_ClearExternalPtr( tens_r );
  }
}

template <typename t>
SEXP cuR_create_tensor_1( SEXP dims_r ){
  int* dims = INTEGER(dims_r);
  int l = dims[0]*dims[1];

  t* tens = new t[l];

  debugPrint( Rprintf( "<%p> Creating L1 tensor\n", (void*)tens ) );

  SEXP tens_r = Rf_protect( R_MakeExternalPtr( tens, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( tens_r, cuR_fin_tensor_1<t>, TRUE );

  Rf_setAttrib( tens_r, R_ClassSymbol,        Rf_mkString("tensor.ptr") );
  Rf_setAttrib( tens_r, Rf_mkString("level"), Rf_ScalarInteger(1) );
  Rf_setAttrib( tens_r, Rf_mkString("dim0"),  Rf_ScalarInteger(dims[0]) );
  Rf_setAttrib( tens_r, Rf_mkString("dim1"),  Rf_ScalarInteger(dims[1]) );

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_create_tensor_1_n( SEXP dims_r ){
  SEXP tens_r = Rf_protect( cuR_create_tensor_1<float>( dims_r ) );
  Rf_setAttrib(tens_r, Rf_mkString("type"), Rf_mkString("n") );

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_create_tensor_1_i( SEXP dims_r ){
  SEXP tens_r = Rf_protect( cuR_create_tensor_1<int>( dims_r ) );
  Rf_setAttrib(tens_r, Rf_mkString("type"), Rf_mkString("i") );

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_create_tensor_1_l( SEXP dims_r ){
  SEXP tens_r = Rf_protect( cuR_create_tensor_1<bool>( dims_r ) );
  Rf_setAttrib(tens_r, Rf_mkString("type"), Rf_mkString("l") );

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_clear_tensor_1_n( SEXP tens_r, SEXP dims_r ){
  int* dims   = INTEGER( dims_r );
  float* tens = (float*)R_ExternalPtrAddr( tens_r );

  memset( tens, 0, sizeof(float) * dims[0]*dims[1] );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_clear_tensor_1_i( SEXP tens_r, SEXP dims_r ){
  int* dims = INTEGER( dims_r );
  int* tens = (int*)R_ExternalPtrAddr( tens_r );

  memset( tens, 0, sizeof(int) * dims[0]*dims[1] );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_clear_tensor_1_l( SEXP tens_r, SEXP dims_r ){
  int* dims   = INTEGER( dims_r );
  bool* tens = (bool*)R_ExternalPtrAddr( tens_r );

  memset( tens, 0, sizeof(bool) * dims[0]*dims[1] );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_destroy_tensor_1_n( SEXP tens_r ){
  cuR_fin_tensor_1<float>( tens_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_destroy_tensor_1_i( SEXP tens_r ){
  cuR_fin_tensor_1<int>( tens_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_destroy_tensor_1_l( SEXP tens_r ){
  cuR_fin_tensor_1<bool>( tens_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// Level 2 ---------------------------------------------------------------------
#ifndef CUDA_EXCLUDE

template <typename t>
void cuR_fin_tensor_2( SEXP tens_r ){
  t* tens = (t*)R_ExternalPtrAddr( tens_r );

  // Destroy context and free memory!
  // Clear R object too
  if( tens ){
    debugPrint( Rprintf( "<%p> Finalizing L2 tensor\n", (void*)tens ) );

    cudaFreeHost( tens );
    R_ClearExternalPtr( tens_r );
  }
}

template <typename t>
SEXP cuR_create_tensor_2( SEXP dims_r ){
  t* tens;
  int* dims = INTEGER(dims_r);
  int l = dims[0]*dims[1];

  cudaTry( cudaHostAlloc( (void**)&tens, l*sizeof(t), cudaHostAllocDefault) );

  debugPrint( Rprintf( "<%p> Creating L2 tensor\n", (void*)tens ) );

  SEXP tens_r = Rf_protect( R_MakeExternalPtr( tens, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( tens_r, cuR_fin_tensor_2<t>, TRUE );

  Rf_setAttrib( tens_r, R_ClassSymbol,        Rf_mkString("tensor.ptr") );
  Rf_setAttrib( tens_r, Rf_mkString("level"), Rf_ScalarInteger(2) );
  Rf_setAttrib( tens_r, Rf_mkString("dim0"),  Rf_ScalarInteger(dims[0]) );
  Rf_setAttrib( tens_r, Rf_mkString("dim1"),  Rf_ScalarInteger(dims[1]) );

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_create_tensor_2_n( SEXP dims_r ){
  SEXP tens_r = Rf_protect( cuR_create_tensor_2<float>( dims_r ) );
  Rf_setAttrib(tens_r, Rf_mkString("type"), Rf_mkString("n") );

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_create_tensor_2_i( SEXP dims_r ){
  SEXP tens_r = Rf_protect( cuR_create_tensor_2<int>( dims_r ) );
  Rf_setAttrib(tens_r, Rf_mkString("type"), Rf_mkString("i") );

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_create_tensor_2_l( SEXP dims_r ){
  SEXP tens_r = Rf_protect( cuR_create_tensor_2<bool>( dims_r ) );
  Rf_setAttrib(tens_r, Rf_mkString("type"), Rf_mkString("l") );

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_clear_tensor_2_n( SEXP tens_r, SEXP dims_r ){
  int* dims   = INTEGER( dims_r );
  float* tens = (float*)R_ExternalPtrAddr( tens_r );

  memset( tens, 0, sizeof(float) * dims[0]*dims[1] );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_clear_tensor_2_i( SEXP tens_r, SEXP dims_r ){
  int* dims = INTEGER( dims_r );
  int* tens = (int*)R_ExternalPtrAddr( tens_r );

  memset( tens, 0, sizeof(int) * dims[0]*dims[1] );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_clear_tensor_2_l( SEXP tens_r, SEXP dims_r ){
  int* dims   = INTEGER( dims_r );
  bool* tens = (bool*)R_ExternalPtrAddr( tens_r );

  memset( tens, 0, sizeof(bool) * dims[0]*dims[1] );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_destroy_tensor_2_n( SEXP tens_r ){
  cuR_fin_tensor_2<float>( tens_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_destroy_tensor_2_i( SEXP tens_r ){
  cuR_fin_tensor_2<int>( tens_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_destroy_tensor_2_l( SEXP tens_r ){
  cuR_fin_tensor_2<bool>( tens_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// Level 3 ---------------------------------------------------------------------

template <typename t>
void cuR_fin_tensor_3( SEXP tens_r ){
  t* tens = (t*)R_ExternalPtrAddr( tens_r );

  if( tens ){
    debugPrint( Rprintf( "<%p> Finalizing L3 tensor\n", (void*)tens ) );

    // Free memory
    cudaFree( tens );
    R_ClearExternalPtr( tens_r );
  }
}

template <typename t>
SEXP cuR_create_tensor_3( SEXP dims_r ){
  t* tens;
  int* dims = INTEGER(dims_r);
  int l = dims[0]*dims[1];

  cudaTry( cudaMalloc( (void**)&tens, l*sizeof(t) ) );

  debugPrint( Rprintf( "<%p> Creating L3 tensor\n", (void*)tens ) );

  // Wrap pointer in a SEXP and register a finalizer
  SEXP tens_r = Rf_protect( R_MakeExternalPtr( tens, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( tens_r, cuR_fin_tensor_3<t>, TRUE );

  Rf_setAttrib( tens_r, R_ClassSymbol,        Rf_mkString("tensor.ptr"));
  Rf_setAttrib( tens_r, Rf_mkString("level"), Rf_ScalarInteger(3));
  Rf_setAttrib( tens_r, Rf_mkString("dim0"),  Rf_ScalarInteger(dims[0]));
  Rf_setAttrib( tens_r, Rf_mkString("dim1"),  Rf_ScalarInteger(dims[1]));

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_create_tensor_3_n( SEXP dims_r ){
  SEXP tens_r = Rf_protect( cuR_create_tensor_3<float>( dims_r ) );
  Rf_setAttrib(tens_r, Rf_mkString("type"), Rf_mkString("n") );

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_create_tensor_3_i( SEXP dims_r ){
  SEXP tens_r = Rf_protect( cuR_create_tensor_3<int>( dims_r ) );
  Rf_setAttrib(tens_r, Rf_mkString("type"), Rf_mkString("i") );

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_create_tensor_3_l( SEXP dims_r ){
  SEXP tens_r = Rf_protect( cuR_create_tensor_3<bool>( dims_r ) );
  Rf_setAttrib(tens_r, Rf_mkString("type"), Rf_mkString("l") );

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_clear_tensor_3_n( SEXP tens_r, SEXP dims_r ){
  int* dims   = INTEGER( dims_r );
  float* tens = (float*)R_ExternalPtrAddr( tens_r );

  cudaTry( cudaMemset( tens, 0, sizeof(float) * dims[0]*dims[1] ) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_clear_tensor_3_i( SEXP tens_r, SEXP dims_r ){
  int* dims = INTEGER( dims_r );
  int* tens = (int*)R_ExternalPtrAddr( tens_r );

  cudaTry( cudaMemset( tens, 0, sizeof(int) * dims[0]*dims[1] ) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_clear_tensor_3_l( SEXP tens_r, SEXP dims_r ){
  int* dims   = INTEGER( dims_r );
  bool* tens = (bool*)R_ExternalPtrAddr( tens_r );

  cudaTry( cudaMemset( tens, 0, sizeof(bool) * dims[0]*dims[1] ) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_destroy_tensor_3_n( SEXP tens_r ){
  cuR_fin_tensor_3<float>( tens_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_destroy_tensor_3_i( SEXP tens_r ){
  cuR_fin_tensor_3<int>( tens_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_destroy_tensor_3_l( SEXP tens_r ){
  cuR_fin_tensor_3<bool>( tens_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

#endif
