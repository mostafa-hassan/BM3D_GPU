#ifndef TENSOR_H
#define TENSOR_H
#include <iostream>
#include <iomanip>
#include <cfloat>

#include <cuda.h>
#include <cuda_runtime.h>

#include <boost/multi_array.hpp>

#include "image.h"

#define CUDA_ERR_CHECK(rv) {cuda_assert((rv), __FILE__, __LINE__);}
// Activate debug mode?
#define __tensor_debug__ 1

// Pack the coordinate and distance together to form a single structure
struct dist_t
{
	unsigned short m_this_patch_g_coord_x;
	unsigned short m_this_patch_g_coord_y;
	image_t::pixel_t m_dist;
	
	__host__ __device__ dist_t
	(
		// default constructor
	) : m_this_patch_g_coord_x (0),
		m_this_patch_g_coord_y (0),
		m_dist (FLT_MAX)
	{}
	__host__ __device__ dist_t
	(
		const unsigned short & this_patch_g_coord_x, 
		const unsigned short & this_patch_g_coord_y, 
		const image_t::pixel_t & dist
	) : m_this_patch_g_coord_x (this_patch_g_coord_x), 
		m_this_patch_g_coord_y (this_patch_g_coord_y),
		m_dist (dist)
	{}
	__forceinline__ __host__ __device__ dist_t & operator=(const dist_t & rhs)
	{
		m_this_patch_g_coord_x = rhs.m_this_patch_g_coord_x; 
		m_this_patch_g_coord_y = rhs.m_this_patch_g_coord_y;
		m_dist = rhs.m_dist;
		
		return *this;
	}
};
__forceinline__ __host__ __device__ bool operator< (const dist_t & lhs, const dist_t & rhs){return lhs.m_dist <  rhs.m_dist;}

template < typename data_t, unsigned dim >
struct tensor_t
{

	data_t * m_data_arr;
	unsigned m_size;
	unsigned m_extents[dim];

	// Constructor(s), Destructor
	 tensor_t();
	 tensor_t(const unsigned * const extents);
	 tensor_t(const boost::multi_array < data_t, dim > & ndarray);
	 tensor_t(const std::vector < data_t > & array);
	 tensor_t(const tensor_t < data_t, dim > & tensor);
	 tensor_t < data_t, dim > & operator=(const unsigned * const extents);
	 tensor_t < data_t, dim > & operator=(const boost::multi_array < data_t, dim > & ndarray);
	 tensor_t < data_t, dim > & operator=(const std::vector < data_t > & array);
	 tensor_t < data_t, dim > & operator=(const tensor_t < data_t, dim > & tensor);
	~tensor_t();
	// @brief output the data as a multi-dimensional array
	boost::multi_array < data_t, dim > to_ndarray() const;
};

inline void cuda_assert(const cudaError_t & err_code, const char * const fname, const unsigned line)
{
	if (err_code != cudaSuccess)
	{
		std::cerr << "CUDA assertion at line " << std::setw(5) << line << " of file " << fname << 
			" failed with error code \"" << cudaGetErrorString(err_code) << "\"." << std::endl;
		assert(0);
	}
}

#endif // TENSOR_H
