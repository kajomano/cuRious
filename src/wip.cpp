#include "common_R.h"
#include "common_cuda.h"
#include "common_debug.h"

extern "C"
SEXP cuR_wip_overlap_bug(){
  const int n = 1000;

  float* data_in_host;
  float* data_in_device;

  cudaHostAlloc( (void**)&data_in_host, n*n*sizeof(float), cudaHostAllocPortable);
  cudaMalloc( (void**)&data_in_device, n*n*sizeof(float) );

  float* data_out_host;
  float* data_out_device;

  cudaHostAlloc( (void**)&data_out_host, n*n*sizeof(float), cudaHostAllocPortable);
  cudaMalloc( (void**)&data_out_device, n*n*sizeof(float) );


  float* data_proc_A_device;
  float* data_proc_B_device;
  float* data_proc_C_device;

  cudaMalloc( (void**)&data_proc_A_device, n*n*sizeof(float) );
  cudaMalloc( (void**)&data_proc_B_device, n*n*sizeof(float) );
  cudaMalloc( (void**)&data_proc_C_device, n*n*sizeof(float) );

  cudaStream_t stream_in;
  cudaStream_t stream_out;
  cudaStream_t stream_proc;

  cudaStreamCreate( &stream_in );
  cudaStreamCreate( &stream_out );
  cudaStreamCreate( &stream_proc );

  cublasHandle_t handle;

  cublasCreate( &handle );
  cublasSetStream( handle, stream_proc );

  float al = 1;
  float be = 1;

  // ====================================================

  cudaMemcpyAsync( data_in_device, data_in_host, n*n*sizeof(float), cudaMemcpyHostToDevice, stream_in );
  // cudaMemcpyAsync( data_out_host, data_out_device, n*n*sizeof(float), cudaMemcpyDeviceToHost, stream_out );
  cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, data_proc_A_device, n, data_proc_B_device, n, &be, data_proc_C_device, n );

  // ====================================================

  cublasDestroy( handle );

  cudaStreamDestroy( stream_in );
  cudaStreamDestroy( stream_out );
  cudaStreamDestroy( stream_proc );

  cudaFreeHost( data_in_host );
  cudaFree( data_in_device );

  cudaFreeHost( data_out_host );
  cudaFree( data_out_device );

  cudaFree( data_proc_A_device );
  cudaFree( data_proc_B_device );
  cudaFree( data_proc_C_device );

  return R_NilValue;
}
