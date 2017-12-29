#include "common_cpp.h"


void finalize_tensor( SEXP ptr ){
  float* tens_dev = (float*)R_ExternalPtrAddr(ptr);

#ifdef DEBUG_PRINTS
  Rprintf( "Finalizing object at <%p>\n", (void*) tens_dev );
#endif

  // Free memory
  cudaTry( cudaFree( tens_dev ) );
  R_ClearExternalPtr(ptr);
}

extern "C"
SEXP dive_tensor( SEXP tens_r, SEXP n_dims_r, SEXP dims_r ) {
  // Calculate tensor length
  int l = get_tensor_length( Rf_asInteger(n_dims_r), INTEGER(dims_r) );

  // Create pointer to the actual data in the SEXP
  double* tens_c = REAL( tens_r );

  // Allocate memory on the host
  float* tens_host = new float[l];

  // Convert to float in host memory
  for (int i = 0; i < l; i++){
    tens_host[i] = (float)tens_c[i];
  }

  // Allocate device memory and copy host vector
  float* tens_dev;
  cudaTry( cudaMalloc( (void**)&tens_dev, l*sizeof(float) ) );
  cudaTry( cudaMemcpy( tens_dev, tens_host, l*sizeof(float), cudaMemcpyHostToDevice ) );

#ifdef DEBUG_PRINTS
  Rprintf( "Created object at <%p>\n", (void*)tens_dev );
#endif

  // Free the host memory
  delete[] tens_host;

  // Return to R with an external pointer SEXP
  SEXP ptr = Rf_protect( R_MakeExternalPtr( tens_dev, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ptr, finalize_tensor, TRUE );

  Rf_unprotect(1);
  return ptr;
}

extern "C"
SEXP surface_tensor( SEXP ptr, SEXP n_dims_r, SEXP dims_r ) {
  // Dimensions and tensor length
  int n_dims = Rf_asInteger(n_dims_r);
  int* dims  = INTEGER( dims_r );
  int l      = get_tensor_length( n_dims, dims );

  // Create pointer to the device memory object
  float* tens_dev = (float*)R_ExternalPtrAddr(ptr);

  // Allocate host memory and copy back content from device
  float* tens_host = new float[l];
  cudaTry( cudaMemcpy(tens_host, tens_dev, l*sizeof(float), cudaMemcpyDeviceToHost) );

  // Create the correct R object
  SEXP tens_r;
  if( n_dims == 1 ){
    tens_r = Rf_protect( Rf_allocVector( REALSXP, dims[0] ) );
  }else if( n_dims == 2 ){
    tens_r = Rf_protect( Rf_allocMatrix( REALSXP, dims[0], dims[1] ) );
  }else{
    Rprintf( "Error: Invalid number of dimensions for tensor!" );
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
SEXP push_tensor( SEXP ptr, SEXP tens_r, SEXP n_dims_r, SEXP dims_r ) {
  // Create pointer to the actual data in the tens_r
  // and to the device memory object
  double* tens_c = REAL( tens_r );
  float* tens_dev = (float*)R_ExternalPtrAddr(ptr);

  // Calculate vector length
  int l = get_tensor_length( Rf_asInteger(n_dims_r), INTEGER(dims_r) );

  // Allocate memory on the host
  float* tens_host = new float[l];

  // Convert to float in host memory
  for (int i = 0; i < l; i++){
    tens_host[i] = (float)tens_c[i];
  }

  // Copy host vector to device
  cudaTry( cudaMemcpy(tens_dev, tens_host, l*sizeof(float), cudaMemcpyHostToDevice) );

#ifdef DEBUG_PRINTS
  Rprintf( "Copied data to <%p>\n", (void*) tens_dev );
#endif

  // Free the host memory
  delete[] tens_host;

  return R_NilValue;
}
