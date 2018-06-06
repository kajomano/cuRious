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

#include <thrust/system/cuda/execution_policy.h>

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
void cuR_thrust_pow2_cu( float* A_ptr, float* B_ptr, int* dims, cudaStream_t* stream_ptr ){
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
      power_functor( 2.0 )
    );
  }else{
    thrust::transform(
      t_A_ptr,
      t_A_ptr + ( dims[0] * dims[1] ),
      t_B_ptr,
      power_functor( 2.0 )
    );
  }

  // float* tmp_cent = thrust::raw_pointer_cast(t_tmp_cent.data());
  // float* tmp_ones = thrust::raw_pointer_cast(t_tmp_ones.data());
};

extern "C"
#ifdef _WIN32
__declspec( dllexport )
#endif
void cuR_thrust_cmins_cu( float* A_ptr, int* x_ptr, int* dims, cudaStream_t* stream_ptr ){
  // Thrust pointers
  thrust::device_ptr<float> t_A_ptr( A_ptr );
  thrust::device_ptr<int>   t_x_ptr( x_ptr );

  // colmins
  if( stream_ptr ){
    thrust::reduce_by_key(
      thrust::cuda::par.on( *stream_ptr ),

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
};

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

