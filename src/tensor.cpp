#include <cuda.h>
#include <cuda_runtime_api.h>

#define R_NO_REMAP 1
//#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

#include "debug.h"

int cuR_get_tensor_length( int n_dims, int* dims ){
  int l = dims[0];
  for( int i = 1; i < n_dims; i++ ){
    l *= dims[i];
  }

  return l;
}

void cuR_finalize_tensor( SEXP ptr ){
  float* tens_dev = (float*)R_ExternalPtrAddr(ptr);

  if( tens_dev ){
#ifdef DEBUG_PRINTS
    Rprintf( "Finalizing tensor at <%p>\n", (void*) tens_dev );
#endif

    // Free memory
    cudaFree( tens_dev );
    R_ClearExternalPtr(ptr);
  }
}

extern "C"
SEXP cuR_dive_tensor( SEXP tens_r, SEXP n_dims_r, SEXP dims_r ) {
  // Calculate tensor length
  // Dimensions and tensor length
  int l = cuR_get_tensor_length( Rf_asInteger(n_dims_r), INTEGER( dims_r ) );

  // Create pointer to the actual data in the SEXP
  double* tens_c = REAL( tens_r );

  // Allocate memory on the host
  float* tens_host = new float[l];

  // TODO: it would be nice to remove this step everywhere
  // Convert to float in host memory
  for (int i = 0; i < l; i++){
    tens_host[i] = (float)tens_c[i];
  }

  // Allocate device memory and copy host vector
  // As cuBLAS does not seem to support pitched memory, we are not bothering
  // with that, everything uses cudaMemcpy (even matrices)
  float* tens_dev;
  cudaError_t cuda_stat;
  cudaTry( cudaMalloc( (void**)&tens_dev, l*sizeof(float) ) )

#ifdef DEBUG_PRINTS
  Rprintf( "Creating tensor at <%p>\n", (void*)tens_dev );
#endif

  cudaTry( cudaMemcpy( tens_dev, tens_host, l*sizeof(float), cudaMemcpyHostToDevice ) )

  // Free the host memory
  delete[] tens_host;

  // Return to R with an external pointer SEXP
  SEXP ptr = Rf_protect( R_MakeExternalPtr( tens_dev, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ptr, cuR_finalize_tensor, TRUE );

  Rf_unprotect(1);
  return ptr;
}

extern "C"
SEXP cuR_surface_tensor( SEXP ptr, SEXP n_dims_r, SEXP dims_r ) {
  // Dimensions and tensor length
  int n_dims = Rf_asInteger(n_dims_r);
  int* dims  = INTEGER( dims_r );
  int l      = cuR_get_tensor_length( n_dims, dims );

  // Create pointer to the device memory object
  float* tens_dev = (float*)R_ExternalPtrAddr(ptr);

  // Allocate host memory and copy back content from device
  float* tens_host = new float[l];
  cudaError_t cuda_stat;
  cudaTry( cudaMemcpy(tens_host, tens_dev, l*sizeof(float), cudaMemcpyDeviceToHost) );

  // Create the correct R object
  SEXP tens_r;
  if( n_dims == 1 ){
    tens_r = Rf_protect( Rf_allocVector( REALSXP, dims[0] ) );
  }else if( n_dims == 2 ){
    tens_r = Rf_protect( Rf_allocMatrix( REALSXP, dims[0], dims[1] ) );
  }else{
    return R_NilValue;
  }

  // Create a pointer to the actual data in the SEXP
  double* tens_c = REAL( tens_r );

  // Fill the SEXP with data
  for (int i = 0; i < l; i++){
    tens_c[i] = (double)tens_host[i];
  }

  // Free host memory, device memory will be taken care of with finalize
  delete[] tens_host;

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_push_tensor( SEXP ptr, SEXP tens_r, SEXP n_dims_r, SEXP dims_r ) {
  // Create pointer to the actual data in the tens_r
  // and to the device memory object
  double* tens_c = REAL( tens_r );
  float* tens_dev = (float*)R_ExternalPtrAddr(ptr);

  // Calculate vector length
  int l = cuR_get_tensor_length( Rf_asInteger(n_dims_r), INTEGER(dims_r) );

  // Allocate memory on the host
  float* tens_host = new float[l];

  // Convert to float in host memory
  for (int i = 0; i < l; i++){
    tens_host[i] = (float)tens_c[i];
  }

  // Copy host vector to device
  cudaError_t cuda_stat;
  cudaTry( cudaMemcpy(tens_dev, tens_host, l*sizeof(float), cudaMemcpyHostToDevice) );

#ifdef DEBUG_PRINTS
  Rprintf( "Copied tensor to <%p>\n", (void*) tens_dev );
#endif

  // Free the host memory
  delete[] tens_host;

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}
