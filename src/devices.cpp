#include "common_R.h"
#include "common_cuda.h"
#include "common_debug.h"

#ifdef CUDA_EXCLUDE

// Device selection and query ==================================================
extern "C"
SEXP cuR_device_count(){
  SEXP count_r = Rf_protect( Rf_ScalarInteger( -1 ) );
  Rf_unprotect(1);
  return count_r;
}

extern "C"
SEXP cuR_device_get(){
  SEXP dev_r = Rf_protect( Rf_ScalarInteger( -1 ) );
  Rf_unprotect(1);
  return dev_r;
}

#else

extern "C"
SEXP cuR_device_count(){
  int count;
  cudaTry( cudaGetDeviceCount	(	&count ) );

  SEXP count_r = Rf_protect( Rf_ScalarInteger( count ) );
  Rf_unprotect(1);
  return count_r;
}

extern "C"
SEXP cuR_device_get(){
  int dev;
  cudaTry( cudaGetDevice ( &dev ) );

  SEXP dev_r = Rf_protect( Rf_ScalarInteger( dev ) );
  Rf_unprotect(1);
  return dev_r;
}

extern "C"
SEXP cuR_device_set( SEXP dev_r ){
  int dev = Rf_asInteger( dev_r );
  cudaTry( cudaSetDevice ( dev ) );

  return R_NilValue;
}

#endif
