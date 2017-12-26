#include <cuda.h>
#include <cuda_runtime_api.h>

#define R_NO_REMAP 1
#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

void finalize_num_vect(SEXP ptr){
  float* vect_cu = (float*)R_ExternalPtrAddr(ptr);

  // Free memory
  // cudaFree(vect_cu);
  delete[] vect_cu;
  R_ClearExternalPtr(ptr);
}

extern "C"
SEXP dive_num_vect( SEXP vect_r ) {
  // Create pointer to the actual data in the SEXP
  double* vect_c = REAL( vect_r );

  // Store length of array
  int L = Rf_length(vect_r);

  // Allocate unified memory
  float* vect_cu;
  //cudaMallocManaged((void**)&vect_cu, L*sizeof(float));
  vect_cu = new float[L];

  // Copy to unified memory
  for (int i = 0; i < L; i++){
    vect_cu[i] = (float)vect_c[i];
  }

  // // Prefetch the data to the GPU
  // int device = -1;
  // cudaGetDevice(&device);
  // cudaMemPrefetchAsync(vect_cu, L*sizeof(float), device, NULL);

  // Return to R with an external pointer SEXP
  SEXP ptr = Rf_protect( R_MakeExternalPtr( vect_cu, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ptr, finalize_num_vect, TRUE );

  Rf_unprotect(1);
  return ptr;
}

extern "C"
SEXP surface_num_vect( SEXP ptr, SEXP l ) {
  // Save vector length
  int L = Rf_asInteger(l);

  // Create pointer to the unified memory object
  float* vect_cu = (float*)R_ExternalPtrAddr(ptr);

  // Create vector R object
  SEXP vect_r = Rf_protect( Rf_allocVector( REALSXP, L ) );

  // Create a pointer to the actual data in the SEXP
  double* vect_c = REAL( vect_r );

  // Fill the SEXP with data
  for (int i = 0; i < L; i++){
    vect_c[i] = (double)vect_cu[i];
  }

  Rf_unprotect(1);
  return vect_r;
}


