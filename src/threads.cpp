#include <windows.h>
#include <process.h>
#include <algorithm>
#include <stdio.h>

struct cuR_thread_args_dd{
  double* src;
  double* dst;
  int* dims;
  int* rsrc;
  int* rdst;
  int offset; // in rows
  int span;
};

unsigned int __stdcall cuR_thread_dd( void* args ){
  cuR_thread_args_dd* arg_list = ( cuR_thread_args_dd* )args;

  int i      = arg_list->offset;
  int i_stop = arg_list->offset + arg_list->span;
  int j, pos_src, pos_dst;

  // Both subsetted
  if( arg_list->rsrc && arg_list->rdst ){


  }else if( !arg_list->rsrc && !arg_list->rdst ){

    // No subsetting
    for(; i < i_stop ; i++ ){
      for( j = 0; j < arg_list->dims[1]; j++ ){
        pos_dst = pos_src = i + j*arg_list->dims[0];
        arg_list->dst[pos_dst] = arg_list->src[pos_src];
      }
    }

  // Destination subsetted
  }else if( !arg_list->rsrc ){

  // Source subsetted
  }else{

  }

  printf( "Thread finished\n" );

  return 0;
}

void cuR_threaded_dd( double* src, double* dst, int* dims, int* rsrc, int* rdst, int threads ){
  threads = std::max( std::min( dims[0]*dims[1], threads ), 1 );
  int span = dims[0] / threads;
  threads -= 1;

  HANDLE* handles = new HANDLE[threads];
  cuR_thread_args_dd* args = new cuR_thread_args_dd[threads];

  // Launch threads
  for( int i = 0; i < threads; i++ ){
    args[i].src    = src;
    args[i].dst    = dst;
    args[i].dims   = dims;
    args[i].rsrc   = rsrc;
    args[i].rdst   = rdst;
    args[i].offset = i*span;
    args[i].span   = span;

    handles[i] = (HANDLE)_beginthreadex(0, 0, cuR_thread_dd, args+i, 0, 0);
  }

  // Do the remainder of the processing on the main thread
  int i      = threads*span;
  int i_stop = dims[0];
  int j, pos_src, pos_dst;

  // Both subsetted
  if( rsrc && rdst ){


  }else if( !rsrc && !rdst ){

    // No subsetting
    for(; i < i_stop ; i++ ){
      for( j = 0; j < dims[1]; j++ ){
        pos_dst = pos_src = i + j*dims[0];
        dst[pos_dst] = src[pos_src];
      }
    }

    // Destination subsetted
  }else if( !rsrc ){

    // Source subsetted
  }else{

  }

  // Wait for threads to finish
  WaitForMultipleObjects( threads, handles, true, INFINITE );

  // Close threads
  for( int i = 0; i < threads; i++ ){
    CloseHandle( handles[i] );
  }

  delete[] handles;
  delete[] args;
}





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

    HANDLE* handles    = new HANDLE[threads];
    cuR_arg_list* args = new cuR_arg_list[threads];

    // Launch threads
    for( int i = 0; i < threads; i++ ){
      args[i].data = data + i*span;
      args[i].buff = buff + i*span;
      args[i].l    = span;

      handles[i] = (HANDLE)_beginthreadex(0, 0, cuR_conv_2_float_thread, args+i, 0, 0);
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
    delete[] args;
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

    HANDLE* handles    = new HANDLE[threads];
    cuR_arg_list* args = new cuR_arg_list[threads];

    // Launch threads
    for( int i = 0; i < threads; i++ ){
      args[i].data = data + i*span;
      args[i].buff = buff + i*span;
      args[i].l    = span;

      handles[i] = (HANDLE)_beginthreadex(0, 0, cuR_conv_2_double_thread, args+i, 0, 0);
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
    delete[] args;
  }
}

