/*
 * File Description and Acknowledgments would be added upon completion
 */
#include <cmath>
#include <omp.h>

#include <vector>

#include "../inc/tensor.h"

/*******************************************************************************/
// Constructor(s), Destructor & Assignment Operator
/*******************************************************************************/

template < typename data_t, unsigned dim >
tensor_t < data_t, dim > :: tensor_t()
{
	m_data_arr = nullptr;
}

template < typename data_t, unsigned dim >
tensor_t < data_t, dim > :: tensor_t(const unsigned * const extents)
{

	for (unsigned i = 0; i < dim; ++i)
	{
		m_extents[i] = extents[i];
	}
	
	unsigned size = 1;
#pragma omp parallel for reduction (*:size)
	for (unsigned i = 0; i < dim; ++i)
	{
		size *= extents[i];
	}
	m_size = size;

	CUDA_ERR_CHECK(cudaMalloc(&m_data_arr,    m_size * sizeof(data_t)));
	CUDA_ERR_CHECK(cudaMemset( m_data_arr, 0, m_size * sizeof(data_t)));
}

template < typename data_t, unsigned dim >
tensor_t < data_t, dim > :: tensor_t(const boost::multi_array < data_t, dim > & ndarray)
{
#pragma omp parallel for
	for (unsigned i = 0; i < dim; ++i)
	{
		m_extents[i] = ndarray.shape()[i];
	}
	
	unsigned size = 1;
#pragma omp parallel for reduction (*:size)
	for (unsigned i = 0; i < dim; ++i)
	{
		size *= ndarray.shape()[i];
	}
	m_size = size;
	
	CUDA_ERR_CHECK(cudaMalloc(&m_data_arr,                 m_size * sizeof(data_t)));
	CUDA_ERR_CHECK(cudaMemcpy( m_data_arr, ndarray.data(), m_size * sizeof(data_t),
		cudaMemcpyHostToDevice));
}

template < typename data_t, unsigned dim >
tensor_t < data_t, dim > :: tensor_t(const std::vector < data_t > & array)
{
	assert(dim == 1);
	
	m_extents[0] = array.size(); m_size = array.size();
	
	CUDA_ERR_CHECK(cudaMalloc(&m_data_arr,               m_size * sizeof(data_t)));
	CUDA_ERR_CHECK(cudaMemcpy( m_data_arr, array.data(), m_size * sizeof(data_t),
		cudaMemcpyHostToDevice));
}

template < typename data_t, unsigned dim >
tensor_t < data_t, dim > :: tensor_t(const tensor_t < data_t, dim > & tensor)
{
	if (m_data_arr != nullptr)
	{
		CUDA_ERR_CHECK(cudaFree(m_data_arr));
	}
	
#pragma omp parallel for
	for (unsigned i = 0; i < dim; ++i)
	{
		m_extents[i] = tensor.m_extents[i];
	}
	
	m_size = tensor.m_size;
	
	CUDA_ERR_CHECK(cudaMalloc(&m_data_arr,                    m_size * sizeof(data_t)));
	CUDA_ERR_CHECK(cudaMemcpy( m_data_arr, tensor.m_data_arr, m_size * sizeof(data_t),
		cudaMemcpyDeviceToDevice));
}

template < typename data_t, unsigned dim >
tensor_t < data_t, dim > &
tensor_t < data_t, dim > ::operator=(const unsigned * const extents)
{
	if (m_data_arr != nullptr)
	{
		CUDA_ERR_CHECK(cudaFree(m_data_arr));
	}
	
#pragma omp parallel for 
	for (unsigned i = 0; i < dim; ++i)
	{
		m_extents[i] = extents[i];
	}
	
	unsigned size = 1;
#pragma omp parallel for reduction (*:size)
	for (unsigned i = 0; i < dim; ++i)
	{
		size *= extents[i];
	}
	m_size = size;
	
	CUDA_ERR_CHECK(cudaMalloc(&m_data_arr,    m_size * sizeof(data_t)));
	CUDA_ERR_CHECK(cudaMemset( m_data_arr, 0, m_size * sizeof(data_t)));
	
	return *this;
}

template < typename data_t, unsigned dim >
tensor_t < data_t, dim > &
tensor_t < data_t, dim > ::operator=(const boost::multi_array < data_t, dim > & ndarray)
{
	if (m_data_arr != nullptr)
	{
		CUDA_ERR_CHECK(cudaFree(m_data_arr));
	}
	
#pragma omp parallel for
	for (unsigned i = 0; i < dim; ++i)
	{
		m_extents[i] = ndarray.shape()[i];
	}
	
	unsigned size = 1;
#pragma omp parallel for reduction (*:size)
	for (unsigned i = 0; i < dim; ++i)
	{
		size *= ndarray.shape()[i];
	}
	m_size = size;
	
	CUDA_ERR_CHECK(cudaMalloc(&m_data_arr,                 m_size * sizeof(data_t)));
	CUDA_ERR_CHECK(cudaMemcpy( m_data_arr, ndarray.data(), m_size * sizeof(data_t),
		cudaMemcpyHostToDevice));
	
	return *this;
}

template < typename data_t, unsigned dim >
tensor_t < data_t, dim > &
tensor_t < data_t, dim > ::operator=(const std::vector < data_t > & array)
{
	if (m_data_arr != nullptr)
	{
		CUDA_ERR_CHECK(cudaFree(m_data_arr));
	}
	
	assert(dim == 1);
	
	m_extents[0] = array.size(); m_size = array.size();
	
	CUDA_ERR_CHECK(cudaMalloc(&m_data_arr,               m_size * sizeof(data_t)));
	CUDA_ERR_CHECK(cudaMemcpy( m_data_arr, array.data(), m_size * sizeof(data_t),
		cudaMemcpyHostToDevice));
	
	return *this;
}

template < typename data_t, unsigned dim >
tensor_t < data_t, dim > &
tensor_t < data_t, dim > ::operator=(const tensor_t < data_t, dim > & tensor)
{
	if (m_data_arr != nullptr)
	{
		CUDA_ERR_CHECK(cudaFree(m_data_arr));
	}
	
#pragma omp parallel for
	for (unsigned i = 0; i < dim; ++i)
	{
		m_extents[i] = tensor.m_extents[i];
	}
	
	m_size = tensor.m_size;
	
	CUDA_ERR_CHECK(cudaMalloc(&m_data_arr,                    m_size * sizeof(data_t)));
	CUDA_ERR_CHECK(cudaMemcpy( m_data_arr, tensor.m_data_arr, m_size * sizeof(data_t),
		cudaMemcpyDeviceToDevice));
	
	return *this;
}

template < typename data_t, unsigned dim >
tensor_t < data_t, dim > ::~tensor_t()
{
	if (m_data_arr != nullptr)
	{
		CUDA_ERR_CHECK(cudaFree(m_data_arr)); m_data_arr = nullptr;
	}
}

/*******************************************************************************/
// Conversion to Boost Multi-Array Object
/*******************************************************************************/
template < typename data_t, unsigned dim >
boost::multi_array < data_t, dim >
tensor_t < data_t, dim > ::to_ndarray() const
{
	std::vector < unsigned > extents (m_extents, m_extents + dim);
	
	boost::multi_array < data_t, dim > ndarray (extents);
	CUDA_ERR_CHECK(cudaMemcpy(ndarray.data(), m_data_arr, m_size * sizeof(data_t), cudaMemcpyDeviceToHost));

	return ndarray;
}

/*******************************************************************************/
// Template Class Instantiation
/*******************************************************************************/
template class tensor_t <    float, 1 >;
template class tensor_t <    float, 2 >;
template class tensor_t <    float, 3 >;
template class tensor_t <    float, 4 >;
template class tensor_t <    float, 5 >;
template class tensor_t <    float, 6 >;
template class tensor_t <    float, 7 >;
template class tensor_t <    float, 8 >;
template class tensor_t <    float, 9 >;
template class tensor_t <   float2, 1 >;
template class tensor_t <   float2, 2 >;
template class tensor_t <   float2, 3 >;
template class tensor_t <   float2, 4 >;
template class tensor_t <   float2, 5 >;
template class tensor_t < unsigned, 1 >;
template class tensor_t < unsigned, 2 >;
template class tensor_t < unsigned, 3 >;
template class tensor_t < unsigned, 4 >;
template class tensor_t < unsigned, 5 >;
template class tensor_t <   dist_t, 1 >;
template class tensor_t <   dist_t, 2 >;
template class tensor_t <   dist_t, 3 >;
template class tensor_t <   dist_t, 4 >;
template class tensor_t <   dist_t, 5 >;
