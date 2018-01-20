void cuR_threaded_dd( double* src, double* dst, int* dims, int* csrc, int* cdst, int threads );

void cuR_conv_2_float( double* data, float* buff, int l, int n_threads );
void cuR_conv_2_double( float* buff, double* data, int l, int n_threads );
