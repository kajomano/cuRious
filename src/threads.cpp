#include <windows.h>
#include <process.h>
#include <algorithm>
#include <cstring>
#include <stdio.h>

inline int round_int_div( int nom, int denom ){
  int whole_div = nom / denom;
  int resid_div = nom % denom;
  if( resid_div > 0 ){
    if( (denom / resid_div) < 3 ){
      whole_div += 1;
    }
  }

  return whole_div;
}

// -----------------------------------------------------------------------------

struct cuR_thread_args_dd{
  double* src;
  double* dst;
  int* dims;
  int* csrc;
  int* cdst;
  int offset;
  int span;
};

unsigned int __stdcall cuR_thread_dd( void* arg_struct ){
  cuR_thread_args_dd* args = ( cuR_thread_args_dd* )arg_struct;

  int curc     = args->offset / args->dims[0];
  int curc_off = args->offset % args->dims[0];
  int src_off, dst_off, block, i;

  if( !args->csrc && !args->cdst ){
    // No subsetting
    while( args->span > 0 ){
      // Copy till the end of this column, or the end of my span
      block   = std::min( args->dims[0] - curc_off, args->span );
      src_off = args->offset;
      dst_off = args->offset;

      // for( i = 0; i < block; i++ ){
      //   args->dst[dst_off+i] = args->src[src_off+i];
      // }
      memcpy( args->dst+dst_off,
              args->src+src_off,
              block*sizeof(double) );

      args->span   -= block;
      args->offset += block;
    }

  }else if( args->csrc && args->cdst ){
    // Both subsetted
    while( args->span > 0 ){
      // Copy till the end of this column, or the end of my span
      block   = std::min( args->dims[0] - curc_off, args->span );
      src_off = (args->csrc[curc]-1)*args->dims[0]+curc_off;
      dst_off = (args->cdst[curc]-1)*args->dims[0]+curc_off;

      // for( i = 0; i < block; i++ ){
      //   args->dst[dst_off+i] = args->src[src_off+i];
      // }
      memcpy( args->dst+dst_off,
              args->src+src_off,
              block*sizeof(double) );

      args->span   -= block;
      args->offset += block;
      curc     += 1;
      curc_off = 0;
    }

  }else if( !args->csrc ){
    // Destination subsetted
    while( args->span > 0 ){
      // Copy till the end of this column, or the end of my span
      block = std::min( args->dims[0] - curc_off, args->span );
      src_off = args->offset;
      dst_off = (args->cdst[curc]-1)*args->dims[0]+curc_off;

      // for( i = 0; i < block; i++ ){
      //   args->dst[dst_off+i] = args->src[src_off+i];
      // }
      memcpy( args->dst+dst_off,
              args->src+src_off,
              block*sizeof(double) );

      args->span   -= block;
      args->offset += block;
      curc     += 1;
      curc_off = 0;
    }

  }else{
    // Source subsetted
    while( args->span > 0 ){
      // Copy till the end of this column, or the end of my span
      block = std::min( args->dims[0] - curc_off, args->span );
      src_off = (args->csrc[curc]-1)*args->dims[0]+curc_off;
      dst_off = args->offset;

      // for( i = 0; i < block; i++ ){
      //   args->dst[dst_off+i] = args->src[src_off+i];
      // }
      memcpy( args->dst+dst_off,
              args->src+src_off,
              block*sizeof(double) );

      args->span   -= block;
      args->offset += block;
      curc     += 1;
      curc_off = 0;
    }
  }

  return 0;
}

void cuR_threaded_dd( double* src, double* dst, int* dims_ptr, int* csrc, int* cdst, int threads ){
  int dims[2] = { dims_ptr[0], dims_ptr[1] };
  int l       = dims[0]*dims[1];
  int offset  = 0;
  int span;

  // Subsetless processing copies need not be chunked by columns,
  // make rows = l, cols = 1
  if( !csrc && !cdst ){
    dims[0] = l;
    dims[1] = 1;
  }

  // At most as many threads as points, but at least 1
  threads = std::max( std::min( l, threads ), 1 );

  HANDLE* handles = new HANDLE[threads - 1];
  cuR_thread_args_dd* args = new cuR_thread_args_dd[threads - 1];

  // Launch threads
  for( int i = 0; i < (threads - 1); i++ ){
    span = round_int_div( l - offset, threads - i );

    args[i].src    = src;
    args[i].dst    = dst;
    args[i].dims   = dims;
    args[i].csrc   = csrc;
    args[i].cdst   = cdst;
    args[i].offset = offset;
    args[i].span   = span;

    handles[i] = (HANDLE)_beginthreadex(0, 0, cuR_thread_dd, args+i, 0, 0);

    offset += span;
  }

  // Do the remainder of the processing on the main thread
  span = l - offset;

  int curc     = offset / dims[0];
  int curc_off = offset % dims[0];
  int src_off, dst_off, block, i;

  if( !csrc && !cdst ){
    // No subsetting
    while( span > 0 ){
      // Copy till the end of this column, or the end of my span
      block   = std::min( dims[0] - curc_off, span );
      src_off = offset;
      dst_off = offset;

      // for( i = 0; i < block; i++ ){
      //   dst[dst_off+i] = src[src_off+i];
      // }
      memcpy( dst+dst_off,
              src+src_off,
              block*sizeof(double) );

      span     -= block;
      offset   += block;
    }
  }else if( csrc && cdst ){
    // Both subsetted
    while( span > 0 ){
      // Copy till the end of this column, or the end of my span
      block   = std::min( dims[0] - curc_off, span );
      src_off = (csrc[curc]-1)*dims[0]+curc_off;
      dst_off = (cdst[curc]-1)*dims[0]+curc_off;

      // for( i = 0; i < block; i++ ){
      //   dst[dst_off+i] = src[src_off+i];
      // }
      memcpy( dst+dst_off,
              src+src_off,
              block*sizeof(double) );

      span     -= block;
      offset   += block;
      curc     += 1;
      curc_off = 0;
    }

  }else if( !csrc ){
    // Destination subsetted
    while( span > 0 ){
      // Copy till the end of this column, or the end of my span
      block   = std::min( dims[0] - curc_off, span );
      src_off = offset;
      dst_off = (cdst[curc]-1)*dims[0]+curc_off;

      // for( i = 0; i < block; i++ ){
      //   dst[dst_off+i] = src[src_off+i];
      // }
      memcpy( dst+dst_off,
              src+src_off,
              block*sizeof(double) );

      span     -= block;
      offset   += block;
      curc     += 1;
      curc_off = 0;
    }

  }else{
    // Source subsetted
    while( span > 0 ){
      // Copy till the end of this column, or the end of my span
      block   = std::min( dims[0] - curc_off, span );
      src_off = (csrc[curc]-1)*dims[0]+curc_off;
      dst_off = offset;

      // for( i = 0; i < block; i++ ){
      //   dst[dst_off+i] = src[src_off+i];
      // }
      memcpy( dst+dst_off,
              src+src_off,
              block*sizeof(double) );

      span     -= block;
      offset   += block;
      curc     += 1;
      curc_off = 0;
    }
  }

  // Wait for threads to finish
  WaitForMultipleObjects( threads - 1, handles, true, INFINITE );

  // Close threads
  for( int i = 0; i < (threads - 1); i++ ){
    CloseHandle( handles[i] );
  }

  delete[] handles;
  delete[] args;
}



//==============================================================================

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

