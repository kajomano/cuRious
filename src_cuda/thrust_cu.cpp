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


#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/pair.h>
#include <map>


#include "stdio.h"

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

// Cached allocator ============================================================
// cached_allocator: a simple allocator for caching allocation requests
class cached_allocator
{
public:
  // just allocate bytes
  typedef char value_type;

  cached_allocator() { }

  ~cached_allocator()
  {
    // free all allocations when cached_allocator goes out of scope
    free_all();
  }

  char* allocate(std::ptrdiff_t num_bytes)
  {
    char* result = 0;

    // search the cache for a free block
    free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

    if (free_block != free_blocks.end())
    {
      printf("hit\n");

      // get the pointer
      result = free_block->second;

      // erase from the free_blocks map
      free_blocks.erase(free_block);
    }
    else
    {
      // no allocation of the right size exists
      // create a new one with cuda::malloc
      // throw if cuda::malloc can't satisfy the request
      try
      {
        printf("nohit\n");

        // allocate memory and convert cuda::pointer to raw pointer
        result = thrust::cuda::malloc<char>(num_bytes).get();
      }
      catch(std::runtime_error &e)
      {
        throw;
      }
    }

    // insert the allocated pointer into the allocated_blocks map
    allocated_blocks.insert(std::make_pair(result, num_bytes));

    return result;
  }

  void deallocate(char* ptr, size_t n)
  {
    // erase the allocated block from the allocated blocks map
    allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);

    // insert the block into the free blocks map
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

private:
  typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
  typedef std::map<char*, std::ptrdiff_t> allocated_blocks_type;

  free_blocks_type free_blocks;
  allocated_blocks_type allocated_blocks;

  void free_all()
  {
    printf("clean\n");

    // deallocate all outstanding blocks in both lists
    for (free_blocks_type::iterator i = free_blocks.begin();
         i != free_blocks.end(); i++)
    {
      // transform the pointer to cuda::pointer before calling cuda::free
      thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
    }

    for (allocated_blocks_type::iterator i = allocated_blocks.begin();
         i != allocated_blocks.end(); i++)
    {
      // transform the pointer to cuda::pointer before calling cuda::free
      thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
    }
  }
};

// int main()
// {
//   size_t n = 1 << 22;
//
//   thrust::host_vector<int> h_input(n);
//
//   // generate random input
//   thrust::generate(h_input.begin(), h_input.end(), rand);
//
//   thrust::system::cuda::vector<int> d_input = h_input;
//   thrust::system::cuda::vector<int> d_result(n);
//
//   size_t num_trials = 5;
//
//   for(size_t i = 0; i < num_trials; ++i)
//   {
//     // initialize data to sort
//     d_result = d_input;
//
//     // tag iterators with my_tag to cause invocations of our
//     // get_temporary_buffer and return_temporary_buffer
//     // during sort
//     thrust::sort(thrust::retag<my_tag>(d_result.begin()),
//                  thrust::retag<my_tag>(d_result.end()));
//
//     // ensure the result is sorted
//     assert(thrust::is_sorted(d_result.begin(), d_result.end()));
//   }
//
//   // free all allocations before the underlying
//   // device backend (e.g., CUDART) goes out of scope
//   g_allocator.free_all();
//
//   return 0;
// }

extern "C"
#ifdef _WIN32
__declspec( dllexport )
#endif
  void cuR_thrust_pow_cu( float* A_ptr, float* B_ptr, int* dims, float pow, cudaStream_t* stream_ptr ){
    // Thrust pointers
    thrust::device_ptr<float> t_A_ptr( A_ptr );
    thrust::device_ptr<float> t_B_ptr( B_ptr );

    // // Temporary storages
    // thrust::device_vector<float> t_tmp_cent( dims[0]*dims[1] );
    // thrust::device_vector<float> t_tmp_ones( dims[0], 1.0 );

    // cent ^ 2
    if( stream_ptr ){
      thrust::transform(
        thrust::cuda::par.on( *stream_ptr ),
        t_A_ptr,
        t_A_ptr + ( dims[0] * dims[1] ),
        t_B_ptr,
        power_functor( pow )
      );
    }else{
      thrust::transform(
        t_A_ptr,
        t_A_ptr + ( dims[0] * dims[1] ),
        t_B_ptr,
        power_functor( pow )
      );
    }

    // float* tmp_cent = thrust::raw_pointer_cast(t_tmp_cent.data());
    // float* tmp_ones = thrust::raw_pointer_cast(t_tmp_ones.data());
  }

// __global__
// void cuR_thrust_cmin_pos_cu_kern( thrust::device_ptr<float> t_A_ptr, thrust::device_ptr<int> t_x_ptr, int dims_0, int dims_1 ){
//   thrust::reduce_by_key(
//     thrust::cuda::par,
//
//     thrust::make_transform_iterator(
//       thrust::counting_iterator<int>( 0 ),
//       linear_index_to_col_index( dims_0 )
//     ),
//
//     thrust::make_transform_iterator(
//       thrust::counting_iterator<int>( 0 ),
//       linear_index_to_col_index( dims_0 )
//     ) + ( dims_0 * dims_1 ),
//
//     thrust::make_zip_iterator(
//       thrust::make_tuple(
//         t_A_ptr,
//         thrust::make_transform_iterator(
//           thrust::counting_iterator<int>( 0 ),
//           linear_index_to_row_index( dims_0 )
//         )
//       )
//     ),
//
//     thrust::make_discard_iterator(),
//
//     thrust::make_zip_iterator(
//       thrust::make_tuple(
//         thrust::make_discard_iterator(),
//         t_x_ptr
//       )
//     ),
//
//     thrust::equal_to<int>(),
//
//     thrust::minimum< thrust::tuple<float, int> >()
//   );
// }

extern "C"
#ifdef _WIN32
__declspec( dllexport )
#endif
  void cuR_thrust_cmin_pos_cu( float* A_ptr, int* x_ptr, int* dims, cudaStream_t* stream_ptr ){
    // Thrust pointers
    thrust::device_ptr<float> t_A_ptr( A_ptr );
    thrust::device_ptr<int>   t_x_ptr( x_ptr );

    // No arrayed host-side args!
    int dims_0 = dims[0];
    int dims_1 = dims[1];

    cached_allocator alloc;

    if( stream_ptr ){
      // cuR_thrust_cmin_pos_cu_kern<<<1, 1, 0, *stream_ptr>>>( t_A_ptr, t_x_ptr, dims_0, dims_1 );

      printf( "This\n" );

      thrust::reduce_by_key(
        thrust::cuda::par(alloc).on( *stream_ptr ),

        thrust::make_transform_iterator(
          thrust::counting_iterator<int>( 0 ),
          linear_index_to_col_index( dims_0 )
        ),

        thrust::make_transform_iterator(
          thrust::counting_iterator<int>( 0 ),
          linear_index_to_col_index( dims_0 )
        ) + ( dims_0 * dims_1 ),

        thrust::make_zip_iterator(
          thrust::make_tuple(
            t_A_ptr,
            thrust::make_transform_iterator(
              thrust::counting_iterator<int>( 0 ),
              linear_index_to_row_index( dims_0 )
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
      printf( "Thus\n" );

      thrust::reduce_by_key(
        thrust::cuda::par(alloc),

        thrust::make_transform_iterator(
          thrust::counting_iterator<int>( 0 ),
          linear_index_to_col_index( dims_0 )
        ),

        thrust::make_transform_iterator(
          thrust::counting_iterator<int>( 0 ),
          linear_index_to_col_index( dims_0 )
        ) + ( dims_0 * dims_1 ),

        thrust::make_zip_iterator(
          thrust::make_tuple(
            t_A_ptr,
            thrust::make_transform_iterator(
              thrust::counting_iterator<int>( 0 ),
              linear_index_to_row_index( dims_0 )
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



// extern "C"
// void cuR_thrust_table_cu( int* quant, int* perm, int* temp_quant, int* dims, int* weights, int* dims_weights ){
//   // Thrust pointers
//   thrust::device_ptr<int> t_ptr_quant( quant );
//   thrust::device_ptr<int> t_ptr_perm( perm );
//   thrust::device_ptr<int> t_ptr_temp_quant( temp_quant );
//   thrust::device_ptr<int> t_ptr_weights( weights );
//
//   // Initialize perm, copy quant to temp_quant
//   thrust::sequence( t_ptr_perm, t_ptr_perm + dims[0], 1 );
//   thrust::copy( t_ptr_quant, t_ptr_quant + dims[0], t_ptr_temp_quant );
//
//   // Stable sort temp_quant together with perm
//   thrust::stable_sort_by_key( t_ptr_temp_quant,
//                               t_ptr_temp_quant + dims[0],
//                               t_ptr_perm );
//
//   // Count occurences
//   thrust::reduce_by_key(
//     t_ptr_temp_quant,
//     t_ptr_temp_quant + dims[0],
//     thrust::make_constant_iterator( (int) 1 ),
//     thrust::make_discard_iterator(),
//     t_ptr_weights,
//     thrust::equal_to<int>(),
//     thrust::plus<int>()
//   );
// };

