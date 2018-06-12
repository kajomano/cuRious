#include "common_R.h"
#include "common_cuda.h"
#include "common_debug.h"

#include <cstring>

template <typename t>
void cuR_tensor_fin_1( SEXP ptr_r ){
  t* ptr = (t*)R_ExternalPtrAddr( ptr_r );

  if( ptr ){
    debugPrint( Rprintf( "<%p> Finalizing L1 tensor\n", (void*)ptr ) );

    delete[] ptr;
    R_ClearExternalPtr( ptr_r );
  }
}

#ifndef CUDA_EXCLUDE

template <typename t>
void cuR_tensor_fin_2( SEXP ptr_r ){
  t* ptr = (t*)R_ExternalPtrAddr( ptr_r );

  if( ptr ){
    debugPrint( Rprintf( "<%p> Finalizing L2 tensor\n", (void*)ptr ) );

    cudaFreeHost( ptr );
    R_ClearExternalPtr( ptr_r );
  }
}

template <typename t>
void cuR_tensor_fin_3( SEXP ptr_r ){
  t* ptr = (t*)R_ExternalPtrAddr( ptr_r );

  if( ptr ){
    debugPrint( Rprintf( "<%p> Finalizing L3 tensor\n", (void*)ptr ) );

    cudaFree( ptr );
    R_ClearExternalPtr( ptr_r );
  }
}

#endif

extern "C"
SEXP cuR_tensor_create( SEXP level_r, SEXP dims_r, SEXP type_r ){
  int level       = Rf_asInteger( level_r );
  const char type = CHAR( STRING_ELT( type_r, 0 ) )[0];
  int* dims       = INTEGER( dims_r );
  int l           = dims[0] * dims[1];

  void* ptr;
  SEXP ptr_r;

  switch( level ){
  case 1:
    switch( type ){
    case 'n':
      ptr = new float[l];
      ptr_r = Rf_protect( R_MakeExternalPtr( ptr, R_NilValue, R_NilValue ) );
      R_RegisterCFinalizerEx( ptr_r, cuR_tensor_fin_1<float>, TRUE );
      break;
    case 'i':
      ptr = new int[l];
      ptr_r = Rf_protect( R_MakeExternalPtr( ptr, R_NilValue, R_NilValue ) );
      R_RegisterCFinalizerEx( ptr_r, cuR_tensor_fin_1<int>, TRUE );
      break;
    case 'l':
      ptr = new bool[l];
      ptr_r = Rf_protect( R_MakeExternalPtr( ptr, R_NilValue, R_NilValue ) );
      R_RegisterCFinalizerEx( ptr_r, cuR_tensor_fin_1<bool>, TRUE );
      break;
    default:
      Rf_error( "Invalid type in tensor clear call" );
    }
    break;

#ifndef CUDA_EXCLUDE

  case 2:
    switch( type ){
    case 'n':
      cudaTry( cudaHostAlloc( (void**)&ptr, l*sizeof(float), cudaHostAllocPortable) );
      ptr_r = Rf_protect( R_MakeExternalPtr( ptr, R_NilValue, R_NilValue ) );
      R_RegisterCFinalizerEx( ptr_r, cuR_tensor_fin_2<float>, TRUE );
      break;
    case 'i':
      cudaTry( cudaHostAlloc( (void**)&ptr, l*sizeof(int), cudaHostAllocPortable) );
      ptr_r = Rf_protect( R_MakeExternalPtr( ptr, R_NilValue, R_NilValue ) );
      R_RegisterCFinalizerEx( ptr_r, cuR_tensor_fin_2<int>, TRUE );
      break;
    case 'l':
      cudaTry( cudaHostAlloc( (void**)&ptr, l*sizeof(bool), cudaHostAllocPortable) );
      ptr_r = Rf_protect( R_MakeExternalPtr( ptr, R_NilValue, R_NilValue ) );
      R_RegisterCFinalizerEx( ptr_r, cuR_tensor_fin_2<bool>, TRUE );
      break;
    default:
      Rf_error( "Invalid type in tensor clear call" );
    }
    break;

  case 3:
    switch( type ){
    case 'n':
      cudaTry( cudaMalloc( (void**)&ptr, l*sizeof(float) ) );
      ptr_r = Rf_protect( R_MakeExternalPtr( ptr, R_NilValue, R_NilValue ) );
      R_RegisterCFinalizerEx( ptr_r, cuR_tensor_fin_3<float>, TRUE );
      break;
    case 'i':
      cudaTry( cudaMalloc( (void**)&ptr, l*sizeof(int) ) );
      ptr_r = Rf_protect( R_MakeExternalPtr( ptr, R_NilValue, R_NilValue ) );
      R_RegisterCFinalizerEx( ptr_r, cuR_tensor_fin_3<int>, TRUE );
      break;
    case 'l':
      cudaTry( cudaMalloc( (void**)&ptr, l*sizeof(bool) ) );
      ptr_r = Rf_protect( R_MakeExternalPtr( ptr, R_NilValue, R_NilValue ) );
      R_RegisterCFinalizerEx( ptr_r, cuR_tensor_fin_3<bool>, TRUE );
      break;
    default:
      Rf_error( "Invalid type in tensor clear call" );
    }
    break;

#endif

  default:
    Rf_error( "Invalid level in tensor clear call" );
  }

  debugPrint( Rprintf( "<%p> Creating L%d tensor\n", ptr, level ) );

  Rf_unprotect(1);
  return ptr_r;
}

extern "C"
SEXP cuR_tensor_destroy( SEXP ptr_r, SEXP level_r, SEXP type_r ){
  int level       = Rf_asInteger( level_r );
  const char type = CHAR( STRING_ELT( type_r, 0 ) )[0];

  switch( level ){
  case 1:
    switch( type ){
    case 'n':
      cuR_tensor_fin_1<float>( ptr_r );
      break;
    case 'i':
      cuR_tensor_fin_1<int>( ptr_r );
      break;
    case 'l':
      cuR_tensor_fin_1<bool>( ptr_r );
      break;
    default:
      Rf_error( "Invalid type in tensor clear call" );
    }
    break;

#ifndef CUDA_EXCLUDE

  case 2:
    switch( type ){
    case 'n':
      cuR_tensor_fin_2<float>( ptr_r );
      break;
    case 'i':
      cuR_tensor_fin_2<int>( ptr_r );
      break;
    case 'l':
      cuR_tensor_fin_2<bool>( ptr_r );
      break;
    default:
      Rf_error( "Invalid type in tensor clear call" );
    }
    break;

  case 3:
    switch( type ){
    case 'n':
      cuR_tensor_fin_3<float>( ptr_r );
      break;
    case 'i':
      cuR_tensor_fin_3<int>( ptr_r );
      break;
    case 'l':
      cuR_tensor_fin_3<bool>( ptr_r );
      break;
    default:
      Rf_error( "Invalid type in tensor clear call" );
    }
    break;

#endif

  default:
    Rf_error( "Invalid level in tensor clear call" );
  }

  return R_NilValue;
}


extern "C"
SEXP cuR_tensor_clear( SEXP ptr_r, SEXP level_r, SEXP dims_r, SEXP type_r ){
  int level       = Rf_asInteger( level_r );
  const char type = CHAR( STRING_ELT( type_r, 0 ) )[0];
  int* dims       = INTEGER( dims_r );
  int l           = dims[0] * dims[1];

  switch( level ){
  case 0:
    switch( type ){
    case 'n':
      memset( REAL( ptr_r ), 0, sizeof(double) * l );
      break;
    case 'i':
      memset( INTEGER( ptr_r ), 0, sizeof(int) * l );
      break;
    case 'l':
      memset( LOGICAL( ptr_r ), 0, sizeof(int) * l );
      break;
    default:
      Rf_error( "Invalid type in tensor clear call" );
    }
    break;

  case 1:
    switch( type ){
    case 'n':
      memset( (float*) R_ExternalPtrAddr( ptr_r ), 0, sizeof(float) * l );
      break;
    case 'i':
      memset( (int*) R_ExternalPtrAddr( ptr_r ), 0, sizeof(int) * l );
      break;
    case 'l':
      memset( (bool*) R_ExternalPtrAddr( ptr_r ), 0, sizeof(bool) * l );
      break;
    default:
      Rf_error( "Invalid type in tensor clear call" );
    }
    break;

#ifndef CUDA_EXCLUDE

  case 2:
    switch( type ){
    case 'n':
      memset( (float*) R_ExternalPtrAddr( ptr_r ), 0, sizeof(float) * l );
      break;
    case 'i':
      memset( (int*) R_ExternalPtrAddr( ptr_r ), 0, sizeof(int) * l );
      break;
    case 'l':
      memset( (bool*) R_ExternalPtrAddr( ptr_r ), 0, sizeof(bool) * l );
      break;
    default:
      Rf_error( "Invalid type in tensor clear call" );
    }
    break;

  case 3:
    switch( type ){
    case 'n':
      cudaTry( cudaMemset( (float*) R_ExternalPtrAddr( ptr_r ), 0, sizeof(float) * l ) );
      break;
    case 'i':
      cudaTry( cudaMemset( (int*) R_ExternalPtrAddr( ptr_r ), 0, sizeof(int) * l ) );
      break;
    case 'l':
      cudaTry( cudaMemset( (bool*) R_ExternalPtrAddr( ptr_r ), 0, sizeof(bool) * l ) );
      break;
    default:
      Rf_error( "Invalid type in tensor clear call" );
    }
    break;

#endif

  default:
    Rf_error( "Invalid level in tensor clear call" );
  }

  return R_NilValue;
}

extern "C"
SEXP cuR_object_duplicate( SEXP obj_r ){
  SEXP duplicate_r = Rf_protect( Rf_duplicate( obj_r ) );

  Rf_unprotect(1);
  return duplicate_r;
}
