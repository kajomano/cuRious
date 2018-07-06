#include "common_debug.h"

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <map>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>

// TODO ====
// Error returns somehow

class cached_allocator{
public:
  // just allocate bytes
  typedef char value_type;

  cached_allocator(){}

  ~cached_allocator(){
    deallocate_all();
  }

  void deallocate_all(){
    // deallocate all outstanding blocks in both lists
    for ( free_blocks_type::iterator i = free_blocks.begin();
          i != free_blocks.end();
          i++ ){
      // transform the pointer to cuda::pointer before calling cuda::free
      thrust::cuda::free( thrust::cuda::pointer<char>( i->second ) );
    }

    for ( allocated_blocks_type::iterator i = allocated_blocks.begin();
          i != allocated_blocks.end();
          i++ ){
      // transform the pointer to cuda::pointer before calling cuda::free
      thrust::cuda::free( thrust::cuda::pointer<char>( i->first ) );
    }
  };

  char* allocate( std::ptrdiff_t num_bytes ){
    char* result = 0;

    // search the cache for a free block
    // Or exact match with find()
    free_blocks_type::iterator free_block = free_blocks.lower_bound( num_bytes );

    if( free_block != free_blocks.end() ){
      // get the pointer
      result = free_block->second;

      // erase from the free_blocks map
      free_blocks.erase( free_block );
    }
    else{
      // no allocation of the right size exists
      // create a new one with cuda::malloc

      // allocate memory and convert cuda::pointer to raw pointer
      result = thrust::cuda::malloc<char>(num_bytes).get();
    }

    // insert the allocated pointer into the allocated_blocks map
    allocated_blocks.insert( std::make_pair( result, num_bytes ) );

    return result;
  }

  void deallocate( char* ptr, size_t n ){
    // erase the allocated block from the allocated blocks map
    allocated_blocks_type::iterator iter = allocated_blocks.find( ptr );
    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase( iter );

    // insert the block into the free blocks map
    free_blocks.insert( std::make_pair( num_bytes, ptr ) );
  }

private:
  typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
  typedef std::map<char*, std::ptrdiff_t> allocated_blocks_type;

  free_blocks_type free_blocks;
  allocated_blocks_type allocated_blocks;
};

extern "C"
#ifdef _WIN32
__declspec( dllexport )
#endif
  void* cuR_thrust_allocator_create_cu(){
    return (void*)new cached_allocator;
  }

extern "C"
#ifdef _WIN32
__declspec( dllexport )
#endif
  void cuR_thrust_allocator_destroy_cu( void* allocator_ptr ){
    cached_allocator* allocator = ( cached_allocator* ) allocator_ptr;
    delete allocator;
  }

// -----------------------------------------------------------------------------

// Functor for pow
class power_functor{
  const float p; // Power
public:
  power_functor(float _p) : p(_p) {}

  __host__ __device__
  float operator()(const float& x) const {
    return pow( x, p );
  }
};

// Convert a linear index to col index
struct linear_index_to_col_index : public thrust::unary_function<int, int>
{
  const int r; // Number of rows

  __host__ __device__
  linear_index_to_col_index(int _r) : r(_r) {}

  __host__ __device__
  int operator()(int i){
    return i / r + 1;
  }
};

// Convert a linear index to row index
struct linear_index_to_row_index : public thrust::unary_function<int, int>
{
  const int r; // Number of rows

  __host__ __device__
  linear_index_to_row_index(int _r) : r(_r) {}

  __host__ __device__
  int operator()(int i){
    return i % r + 1;
  }
};

extern "C"
#ifdef _WIN32
__declspec( dllexport )
#endif
  void cuR_thrust_pow_cu( float* A_ptr, float* B_ptr, int* dims, float pow, void* allocator_ptr, cudaStream_t* stream_ptr ){
    cached_allocator* allocator = ( cached_allocator* ) allocator_ptr;

    // Thrust pointers
    thrust::device_ptr<float> t_A_ptr( A_ptr );
    thrust::device_ptr<float> t_B_ptr( B_ptr );

    // cent ^ 2
    if( stream_ptr ){
      thrust::transform(
        thrust::cuda::par( *allocator ).on( *stream_ptr ),
        t_A_ptr,
        t_A_ptr + ( dims[0] * dims[1] ),
        t_B_ptr,
        power_functor( pow )
      );
    }else{
      thrust::transform(
        thrust::cuda::par( *allocator ),
        t_A_ptr,
        t_A_ptr + ( dims[0] * dims[1] ),
        t_B_ptr,
        power_functor( pow )
      );
    }
  }

extern "C"
#ifdef _WIN32
__declspec( dllexport )
#endif
  void cuR_thrust_cmin_pos_cu( float* A_ptr, int* x_ptr, int* dims, void* allocator_ptr, cudaStream_t* stream_ptr ){
    cached_allocator* allocator = ( cached_allocator* ) allocator_ptr;

    // Thrust pointers
    thrust::device_ptr<float> t_A_ptr( A_ptr );
    thrust::device_ptr<int>   t_x_ptr( x_ptr );

    if( stream_ptr ){
      thrust::reduce_by_key(
        thrust::cuda::par( *allocator ).on( *stream_ptr ),

        thrust::make_transform_iterator(
          thrust::counting_iterator<int>( 0 ),
          linear_index_to_col_index( dims[0] )
        ),

        thrust::make_transform_iterator(
          thrust::counting_iterator<int>( 0 ),
          linear_index_to_col_index( dims[0] )
        ) + ( dims[0] * dims[1] ),

        thrust::make_zip_iterator(
          thrust::make_tuple(
            t_A_ptr,
            thrust::make_transform_iterator(
              thrust::counting_iterator<int>( 0 ),
              linear_index_to_row_index( dims[0] )
            )
          )
        ),

        thrust::make_discard_iterator(),

        thrust::make_zip_iterator(
          thrust::make_tuple(
            thrust::make_discard_iterator(),
            t_x_ptr
          )
        ),

        thrust::equal_to<int>(),

        thrust::minimum< thrust::tuple<float, int> >()
      );
    }else{
      thrust::reduce_by_key(
        thrust::cuda::par( *allocator ),

        thrust::make_transform_iterator(
          thrust::counting_iterator<int>( 0 ),
          linear_index_to_col_index( dims[0] )
        ),

        thrust::make_transform_iterator(
          thrust::counting_iterator<int>( 0 ),
          linear_index_to_col_index( dims[0] )
        ) + ( dims[0] * dims[1] ),

        thrust::make_zip_iterator(
          thrust::make_tuple(
            t_A_ptr,
            thrust::make_transform_iterator(
              thrust::counting_iterator<int>( 0 ),
              linear_index_to_row_index( dims[0] )
            )
          )
        ),

        thrust::make_discard_iterator(),

        thrust::make_zip_iterator(
          thrust::make_tuple(
            thrust::make_discard_iterator(),
            t_x_ptr
          )
        ),

        thrust::equal_to<int>(),

        thrust::minimum< thrust::tuple<float, int> >()
      );
    }
  }

extern "C"
#ifdef _WIN32
__declspec( dllexport )
#endif
void cuR_thrust_table_cu( int* x_ptr, int* p_ptr, int* w_ptr, int* s_ptr,
                          int* x_dims, int* w_dims, int x_span_off,
                          void* allocator_ptr, cudaStream_t* stream_ptr ){

  cached_allocator* allocator = ( cached_allocator* ) allocator_ptr;

  // Thrust pointers
  thrust::device_ptr<int> t_x_ptr( x_ptr );
  thrust::device_ptr<int> t_p_ptr( p_ptr );
  thrust::device_ptr<int> t_w_ptr( w_ptr );
  thrust::device_ptr<int> t_s_ptr( s_ptr );

  // Ask the allocator for a temporary allocation
  int* tmp_ptr = (int*) allocator -> allocate( sizeof( int ) * w_dims[1] );

  if( stream_ptr ){
    // // Initialize perm
    // thrust::sequence(
    //   thrust::cuda::par( *allocator ).on( *stream_ptr ),
    //   t_p_ptr,
    //   ( t_p_ptr + x_dims[1] ),
    //   1 + x_span_off
    // );
    //
    // // Copy x to s
    // thrust::copy(
    //   thrust::cuda::par( *allocator ).on( *stream_ptr ),
    //   t_x_ptr,
    //   ( t_x_ptr + x_dims[1] ),
    //   t_s_ptr
    // );
    //
    // // Stable sort s together with perm
    // thrust::stable_sort_by_key(
    //   thrust::cuda::par( *allocator ).on( *stream_ptr ),
    //   t_s_ptr,
    //   ( t_s_ptr + x_dims[1] ),
    //   t_p_ptr
    // );
    //
    // // TODO ====
    // // Check if there is enough space in weights
    //
    // // Count occurences
    // thrust::reduce_by_key(
    //   thrust::cuda::par( *allocator ).on( *stream_ptr ),
    //   t_s_ptr,
    //   ( t_s_ptr + x_dims[1] ),
    //   thrust::make_constant_iterator( (int) 1 ),
    //   thrust::make_discard_iterator(),
    //   t_w_ptr,
    //   thrust::equal_to<int>(),
    //   thrust::plus<int>()
    // );
  }else{
    // Initialize perm
    thrust::sequence(
      thrust::cuda::par( *allocator ),
      t_p_ptr,
      ( t_p_ptr + x_dims[1] ),
      1 + x_span_off
    );

    // Copy x to s
    thrust::copy(
      thrust::cuda::par( *allocator ),
      t_x_ptr,
      ( t_x_ptr + x_dims[1] ),
      t_s_ptr
    );

    // Stable sort s together with perm
    thrust::stable_sort_by_key(
      thrust::cuda::par( *allocator ),
      t_s_ptr,
      ( t_s_ptr + x_dims[1] ),
      t_p_ptr
    );

    // TODO ====
    // Check if there is enough space in weights

    // Count occurences
    thrust::reduce_by_key(
      thrust::cuda::par( *allocator ),
      t_s_ptr,
      ( t_s_ptr + x_dims[1] ),
      thrust::make_constant_iterator( (int) 1 ),
      thrust::make_discard_iterator(),
      t_w_ptr,
      thrust::equal_to<int>(),
      thrust::plus<int>()
    );

    // Count unique keys with count_if

    // Permutate-copy to correct places

    allocator -> deallocate( (char*)tmp_ptr, 0 );

  }
}
