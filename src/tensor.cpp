#include <cuda.h>
#include <cuda_runtime_api.h>

#define R_NO_REMAP 1
//#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

#include "debug.h"

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
SEXP cuR_push_tensor( SEXP ptr, SEXP tens_r, SEXP dims_r ) {
  // Create pointer to the actual data in the tens_r
  double* tens_c = REAL( tens_r );

  // Calculate tensor length
  int* dims  = INTEGER( dims_r );
  int l = dims[0]*dims[1];

  // Allocate memory on the host
  float* tens_host = new float[l];

  // Convert to float in host memory
  for (int i = 0; i < l; i++){
    tens_host[i] = (float)tens_c[i];
  }

  // Copy host vector to device
  cudaError_t cuda_stat;

  float* tens_dev;
  if( ptr == R_NilValue ){
    cudaTry( cudaMalloc( (void**)&tens_dev, l*sizeof(float) ) )
  }else{
    tens_dev = (float*)R_ExternalPtrAddr(ptr);
  }

#ifdef DEBUG_PRINTS
    Rprintf( "Copying tensor to <%p>\n", (void*) tens_dev );
#endif

  cudaTry( cudaMemcpy(tens_dev, tens_host, l*sizeof(float), cudaMemcpyHostToDevice) );

  // Free the host memory
  delete[] tens_host;

  // Return to R with an external pointer SEXP if we called this function
  // in a dive
  SEXP ret_r;
  if( ptr == R_NilValue ){
    ret_r = Rf_protect( R_MakeExternalPtr( tens_dev, R_NilValue, R_NilValue ) );
    R_RegisterCFinalizerEx( ret_r, cuR_finalize_tensor, TRUE );
  }else{
    // Or just return something that is not null
    ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  }

  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_pull_tensor( SEXP ptr, SEXP dims_r, SEXP fin_r ) {
  // Dimensions and tensor length
  int* dims  = INTEGER( dims_r );
  int l = dims[0]*dims[1];

  // Create pointer to the device memory object
  float* tens_dev = (float*)R_ExternalPtrAddr(ptr);

  // Allocate host memory and copy back content from device
  float* tens_host = new float[l];
  cudaError_t cuda_stat;
  cudaTry( cudaMemcpy(tens_host, tens_dev, l*sizeof(float), cudaMemcpyDeviceToHost) );

  // Create the correct R object
  SEXP tens_r;
  if( dims[1] == 1 ){
    tens_r = Rf_protect( Rf_allocVector( REALSXP, dims[0] ) );
  }else{
    tens_r = Rf_protect( Rf_allocMatrix( REALSXP, dims[0], dims[1] ) );
  }

  // Create a pointer to the actual data in the SEXP
  double* tens_c = REAL( tens_r );

  // Fill the SEXP with data
  for (int i = 0; i < l; i++){
    tens_c[i] = (double)tens_host[i];
  }

  // Free host and device memory
  delete[] tens_host;
  if( Rf_asLogical( fin_r ) == 1 ){
    cuR_finalize_tensor( ptr );
  }

  Rf_unprotect(1);
  return tens_r;
}

// extern "C"
// SEXP cuR_dive_tensor( SEXP tens_r, SEXP dims_r ) {
//   // Calculate tensor length
//   int* dims = INTEGER( dims_r );
//   int l = dims[0]*dims[1];
//
//   // Create pointer to the actual data in the SEXP
//   double* tens_c = REAL( tens_r );
//
//   // Allocate memory on the host
//   float* tens_host = new float[l];
//
//   // TODO: it would be nice to remove this step everywhere
//   // Convert to float in host memory
//   for (int i = 0; i < l; i++){
//     tens_host[i] = (float)tens_c[i];
//   }
//
//   // Allocate device memory and copy host vector
//   // As cuBLAS does not seem to support pitched memory, we are not bothering
//   // with that, everything uses cudaMemcpy (even matrices)
//   float* tens_dev;
//   cudaError_t cuda_stat;
//   cudaTry( cudaMalloc( (void**)&tens_dev, l*sizeof(float) ) )
//
// #ifdef DEBUG_PRINTS
//     Rprintf( "Creating tensor at <%p>\n", (void*)tens_dev );
// #endif
//
//     cudaTry( cudaMemcpy( tens_dev, tens_host, l*sizeof(float), cudaMemcpyHostToDevice ) )
//
//       // Free the host memory
//       delete[] tens_host;
//
//     // Return to R with an external pointer SEXP
//     SEXP ptr = Rf_protect( R_MakeExternalPtr( tens_dev, R_NilValue, R_NilValue ) );
//     R_RegisterCFinalizerEx( ptr, cuR_finalize_tensor, TRUE );
//
//     Rf_unprotect(1);
//     return ptr;
// }
//
// extern "C"
// SEXP cuR_surface_tensor( SEXP ptr, SEXP dims_r ) {
//   // Dimensions and tensor length
//   int* dims  = INTEGER( dims_r );
//   int l = dims[0]*dims[1];
//
//   // Create pointer to the device memory object
//   float* tens_dev = (float*)R_ExternalPtrAddr(ptr);
//
//   // Allocate host memory and copy back content from device
//   float* tens_host = new float[l];
//   cudaError_t cuda_stat;
//   cudaTry( cudaMemcpy(tens_host, tens_dev, l*sizeof(float), cudaMemcpyDeviceToHost) );
//
//   // Create the correct R object
//   SEXP tens_r;
//   if( dims[1] == 1 ){
//     tens_r = Rf_protect( Rf_allocVector( REALSXP, dims[0] ) );
//   }else{
//     tens_r = Rf_protect( Rf_allocMatrix( REALSXP, dims[0], dims[1] ) );
//   }
//
//   // Create a pointer to the actual data in the SEXP
//   double* tens_c = REAL( tens_r );
//
//   // Fill the SEXP with data
//   for (int i = 0; i < l; i++){
//     tens_c[i] = (double)tens_host[i];
//   }
//
//   // Free host and device memory
//   delete[] tens_host;
//
//   // TODO ====
//   cuR_finalize_tensor( ptr );
//   // VEGIGGONDOLNI mivan ha többen hívnak surface-t?
//   // solution: ezeket mar az r oldalon lehet csekkolni, ide meg nem kell szefti
//
//   Rf_unprotect(1);
//   return tens_r;
// }
