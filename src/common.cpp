extern "C"
int get_tensor_length( int n_dims, int* dims ){
  int l = dims[1];
  for( int i = 0; i < n_dims; i++ ){
    l *= dims[i];
  }

  return l;
}
