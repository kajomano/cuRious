#include <windows.h>
#include <process.h>

#include "debug.h"

struct cuR_arg_list{
  double* data;
  float* buff;
  int l;
};

unsigned int __stdcall cuR_conv_2_float_thread( void* args ){
  cuR_arg_list* arg_list = ( cuR_arg_list* )args;

  for( int i = 0; i < arg_list->l; i++ ){
    arg_list->buff[i] = (float)arg_list->data[i];
  }
  return 0;
}

void cuR_conv_2_float( double* data, float* buff, int l, int n_threads ){
  // Fallback to no extra threading
  if( !(n_threads > 1) || l < n_threads ){
    for( int i = 0; i < l; i++ ){
      buff[i] = (float)data[i];
    }
  }else{
    int threads = n_threads - 1;
    int span    = l / n_threads;

    HANDLE* handles = new HANDLE[threads];

    // Launch threads
    for( int i = 0; i < threads; i++ ){
      cuR_arg_list args;
      args.data = data + i*span;
      args.buff = buff + i*span;
      args.l    = span;

      handles[i] = (HANDLE)_beginthreadex(0, 0, cuR_conv_2_float_thread, &args, 0, 0);
    }

    // Do the remainder of the processing
    for( int j = 0; j < (l - threads*span); j++ ){
      buff[threads*span + j] = (float)data[threads*span + j];
    }

    // Wait for threads to finish
    WaitForMultipleObjects( threads, handles, true, INFINITE );

    // Close threads
    for( int i = 0; i < threads; i++ ){
      CloseHandle( handles[i] );
    }

    delete[] handles;
  }
}

unsigned int __stdcall cuR_conv_2_double_thread( void* args ){
  cuR_arg_list* arg_list = ( cuR_arg_list* )args;

  for( int i = 0; i < arg_list->l; i++ ){
    arg_list->data[i] = (double)arg_list->buff[i];
  }
  return 0;
}

void cuR_conv_2_double( float* buff, double* data, int l, int n_threads ){
  if( !(n_threads > 1) || l < n_threads ){
    for (int i = 0; i < l; i++){
      data[i] = (double)buff[i];
    }
  }else{
    int threads = n_threads - 1;
    int span    = l / n_threads;

    HANDLE* handles = new HANDLE[threads];

    // Launch threads
    for( int i = 0; i < threads; i++ ){
      cuR_arg_list args;
      args.data = data + i*span;
      args.buff = buff + i*span;
      args.l    = span;

      handles[i] = (HANDLE)_beginthreadex(0, 0, cuR_conv_2_double_thread, &args, 0, 0);
    }

    // Do the remainder of the processing
    for( int j = 0; j < (l - threads*span); j++ ){
      data[threads*span + j] = (double)buff[threads*span + j];
    }

    // Wait for threads to finish
    WaitForMultipleObjects( threads, handles, true, INFINITE );

    // Close threads
    for( int i = 0; i < threads; i++ ){
      CloseHandle( handles[i] );
    }

    delete[] handles;
  }
}

