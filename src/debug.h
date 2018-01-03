// Comment this out to have debug prints
// #define DEBUG_PRINTS 1

#define cublasTry(ans){ stat = (ans); if( stat != CUBLAS_STATUS_SUCCESS ) return R_NilValue; }
#define cudaTry(ans){ cudaError_t cuda_stat = (ans); if( cuda_stat != cudaSuccess ) return R_NilValue; }
