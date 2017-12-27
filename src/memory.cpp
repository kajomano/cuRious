#include <cuda.h>
#include <cuda_runtime_api.h>

#define R_NO_REMAP 1
#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

void finalize_num_vect(SEXP ptr){
  float* vect_dev = (float*)R_ExternalPtrAddr(ptr);

  Rprintf( "Finalizing object at <%p>\n", (void*) vect_dev );

  // Free memory
  cudaFree(vect_dev);
  R_ClearExternalPtr(ptr);
}

extern "C"
SEXP dive_num_vect( SEXP vect_r ) {
  // Create pointer to the actual data in the SEXP
  double* vect_c = REAL( vect_r );

  // Store length of array
  int L = Rf_length(vect_r);

  // Allocate memory on the host
  float* vect_host;
  vect_host = new float[L];

  // Convert to float in host memory
  for (int i = 0; i < L; i++){
    vect_host[i] = (float)vect_c[i];
  }

  // Allocate device memory and copy host vector
  float* vect_dev;
  cudaMalloc((void**)&vect_dev, L*sizeof(float));
  cudaMemcpy(vect_dev, vect_host, L*sizeof(float), cudaMemcpyHostToDevice);

  Rprintf( "Created object at <%p>\n", (void*) vect_dev );

  // Free the host memory
  delete[] vect_host;

  // Return to R with an external pointer SEXP
  SEXP ptr = Rf_protect( R_MakeExternalPtr( vect_dev, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ptr, finalize_num_vect, TRUE );

  Rf_unprotect(1);
  return ptr;
}

extern "C"
SEXP surface_num_vect( SEXP ptr, SEXP l ) {
  // Save vector length
  int L = Rf_asInteger(l);

  // Create pointer to the device memory object
  float* vect_dev = (float*)R_ExternalPtrAddr(ptr);

  // Allocate host memory and copy back content from device
  float* vect_host;
  vect_host = new float[L];
  cudaMemcpy(vect_host, vect_dev, L*sizeof(float), cudaMemcpyDeviceToHost);

  // Create vector R object
  SEXP vect_r = Rf_protect( Rf_allocVector( REALSXP, L ) );

  // Create a pointer to the actual data in the SEXP
  double* vect_c = REAL( vect_r );

  // Fill the SEXP with data
  for (int i = 0; i < L; i++){
    vect_c[i] = (double)vect_host[i];
  }

  // Free host memory, device memory will be taken care of with finalize
  delete[] vect_host;

  Rf_unprotect(1);
  return vect_r;
}


