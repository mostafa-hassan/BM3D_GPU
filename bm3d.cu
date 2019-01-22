/*
 * File Description and Acknowledgments would be added upon completion
 */
#include <cmath>
#include <omp.h>
#include <ctime>

#include <cuda_texture_types.h>

#include "../inc/bm3d.h"

#define __config_specific_op__ 1

/*******************************************************************************/
// DCT, Hada Transform, Kaiser Coefficients
/*******************************************************************************/

static texture < image_t::pixel_t, 1, cudaReadModeElementType > l_tex_ref_dct_2d_coeff;     static tensor_t < image_t::pixel_t, 4 > l_dct_2d_coeff_ts;
static texture < image_t::pixel_t, 1, cudaReadModeElementType > l_tex_ref_had_tf_coeff;     static tensor_t < image_t::pixel_t, 3 > l_had_tf_coeff_ts;
static texture < image_t::pixel_t, 1, cudaReadModeElementType > l_tex_ref_kaiser_win_coeff; static tensor_t < image_t::pixel_t, 2 > l_kaiser_win_coeff_ts; 

/*******************************************************************************/
// Auxiliary Data Structure
/*******************************************************************************/

struct aux_dist_t
{
	unsigned m_l_threadIdx; image_t::pixel_t m_dist;
	
	__forceinline__ __device__          aux_dist_t & operator=(        aux_dist_t & rhs)
	{
		m_l_threadIdx = rhs.m_l_threadIdx; m_dist = rhs.m_dist;
		
		return *this;
	}
	__forceinline__ __device__          aux_dist_t & operator=(   const aux_dist_t & rhs)
	{
		m_l_threadIdx = rhs.m_l_threadIdx; m_dist = rhs.m_dist;
		
		return *this;
	}
	__forceinline__ __device__ volatile aux_dist_t & operator=(volatile aux_dist_t & rhs) volatile
	{
		m_l_threadIdx = rhs.m_l_threadIdx; m_dist = rhs.m_dist;
		
		return *this;
	}
};
__forceinline__ __device__ bool operator< (         aux_dist_t & lhs,          aux_dist_t & rhs){return lhs.m_dist <  rhs.m_dist;}
__forceinline__ __device__ bool operator< (   const aux_dist_t & lhs,    const aux_dist_t & rhs){return lhs.m_dist <  rhs.m_dist;}
__forceinline__ __device__ bool operator< (volatile aux_dist_t & lhs, volatile aux_dist_t & rhs){return lhs.m_dist <  rhs.m_dist;}

/*******************************************************************************/
// Zeroth Step: Pre-compute the DCT for All Patches
//******************************************************************************/
#if __config_specific_op__
// @brief pre-compute the DCT values of all the patches
// @param (output) img_pixel_arr_dct: DCT of all patches
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) img_pixel_arr_raw: input pixel array
// (shape: nchnls * height * width)
// @param ( input) width & height: shape of the input pixel array
static __global__ void cuda_precompute_dct
(
	      image_t::pixel_t * const __restrict__ img_pixel_arr_dct,
	const image_t::pixel_t * const __restrict__ img_pixel_arr_raw,
	const unsigned width, const unsigned height
)
{
	extern __shared__ image_t::pixel_t s_data_cuda_precompute_dct[];
	
	const unsigned l_threadIdx_xy  =   threadIdx.x  + threadIdx.y * blockDim.x , blockDim_xy  = blockDim.y * blockDim.x ;
	const unsigned l_threadIdx_xyz = l_threadIdx_xy + threadIdx.z * blockDim_xy, blockDim_xyz = blockDim.z * blockDim_xy;
	
	image_t::pixel_t * const s_img_pixel_arr_raw = s_data_cuda_precompute_dct;
	image_t::pixel_t * const s_img_pixel_arr_dct = s_img_pixel_arr_raw + blockDim_xyz;
	//**************************************************************************
	// pre-compute the offsets of multi-dimensional arrays
	const unsigned img_pixel_arr_dct_offset[3] = 
		{
			gridDim.y * gridDim.x * blockDim_xyz,
			            gridDim.x * blockDim_xyz,
			                        blockDim_xyz
		};
	//**************************************************************************
	// initialize the content of the shared memory
	if (blockIdx.x == gridDim.x - 1)
	{
		s_img_pixel_arr_raw[l_threadIdx_xyz] = threadIdx.z == 0 ? 
			img_pixel_arr_raw
			[
				 blockIdx.z                * height * width + 
				(blockIdx.y + threadIdx.y)          * width + 
				(blockIdx.x * 2 + threadIdx.z + threadIdx.x)
			] : 0.0;
	}
	else
	{
		s_img_pixel_arr_raw[l_threadIdx_xyz] = 
			img_pixel_arr_raw
			[
				 blockIdx.z                * height * width + 
				(blockIdx.y + threadIdx.y)          * width + 
				(blockIdx.x * 2 + threadIdx.z + threadIdx.x)
			];
	}
	s_img_pixel_arr_dct[l_threadIdx_xyz] = 0.0;
	//**************************************************************************
	for (unsigned i = 0; i < blockDim.y; ++i) // dummy variable
	{
		for (unsigned j = 0; j < blockDim.x; ++j) // dummy variable
		{
			const unsigned dummy_l_threadIdx_xy = j + i * blockDim.x;
			
			s_img_pixel_arr_dct[l_threadIdx_xyz] += s_img_pixel_arr_raw[dummy_l_threadIdx_xy + threadIdx.z * blockDim_xy] * 
				tex1Dfetch(l_tex_ref_dct_2d_coeff, dummy_l_threadIdx_xy + l_threadIdx_xy * blockDim_xy);
		} // dummy variable
	} // dummy variable
	img_pixel_arr_dct
	[
		   blockIdx.z * img_pixel_arr_dct_offset[0] + 
		   blockIdx.y * img_pixel_arr_dct_offset[1] + 
		   blockIdx.x * img_pixel_arr_dct_offset[2] + 
		l_threadIdx_xyz
	] = s_img_pixel_arr_dct[l_threadIdx_xyz];
}
#else // !__config_specific_op__
// @brief pre-compute the DCT values of all the patches
// @param (output) img_pixel_arr_dct: DCT of all patches
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) img_pixel_arr_raw: input pixel array
// (shape: nchnls * height * width)
// @param ( input) width & height: shape of the input pixel array
static __global__ void cuda_precompute_dct 
(
	      image_t::pixel_t * __restrict__ const img_pixel_arr_dct,
	const image_t::pixel_t * __restrict__ const img_pixel_arr_raw,
	const unsigned width, const unsigned height
)
{
	extern __shared__ image_t::pixel_t s_data_cuda_precompute_dct[];
	
	const unsigned l_threadIdx = threadIdx.x + threadIdx.y * blockDim.x, blockDim_xy = blockDim.y * blockDim.x;
	
	image_t::pixel_t * const s_img_pixel_arr_raw = s_data_cuda_precompute_dct;
	image_t::pixel_t * const s_img_pixel_arr_dct = s_img_pixel_arr_raw + blockDim_xy;
	//**************************************************************************
	// pre-compute the offsets of multi-dimensional array
	const unsigned img_pixel_arr_dct_offset[3] = 
		{
			gridDim.y * gridDim.x * blockDim_xy,
			            gridDim.x * blockDim_xy,
			                        blockDim_xy
		};
	//**************************************************************************
	// initialize the content of the shared memory
	s_img_pixel_arr_raw[l_threadIdx] = 
		img_pixel_arr_raw
		[
			 blockIdx.z                * height * width + 
			(blockIdx.y + threadIdx.y)          * width + 
			(blockIdx.x + threadIdx.x)
		];
	s_img_pixel_arr_dct[l_threadIdx] = 0.0;
	__syncthreads(); // end of initializing the shared memory
	//**************************************************************************
	for (unsigned i = 0; i < blockDim.y; ++i) // dummy variable
	{
		for (unsigned j = 0; j < blockDim.x; ++j) // dummy variable
		{
			const unsigned dummy_l_threadIdx = j + i * blockDim.x;
			
			s_img_pixel_arr_dct[l_threadIdx] += s_img_pixel_arr_raw[dummy_l_threadIdx] * 
				tex1Dfetch(l_tex_ref_dct_2d_coeff, dummy_l_threadIdx + l_threadIdx * blockDim_xy);
		} // dummy variable
	} // dummy variable
	img_pixel_arr_dct
	[
		   blockIdx.z * img_pixel_arr_dct_offset[0] + 
		   blockIdx.y * img_pixel_arr_dct_offset[1] + 
		   blockIdx.x * img_pixel_arr_dct_offset[2] + 
		l_threadIdx
	] = s_img_pixel_arr_dct[l_threadIdx];
}
#endif // __config_specific_op__
// @brief precompute the DCT values of all the patches
// @param (output) img_pixel_arr_dct_noisy: DCT of noisy image
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) img_pixel_arr_raw_noisy: noisy image
// (shape: nchnls * height * width)
// @param bm3d_param: BM3D parameters
void bm3d_precompute_dct_ht 
(
	      tensor_t < image_t::pixel_t, 5 > & img_pixel_arr_dct_noisy,
	const tensor_t < image_t::pixel_t, 3 > & img_pixel_arr_raw_noisy,
	const bm3d_param_t & bm3d_param
)
{
	// pre-compute the DCT coefficients and bind them to the texture memory
	boost::multi_array < image_t::pixel_t, 4 > dct_2d_coeff_ndarr
		(
			boost::extents
			[bm3d_param.m_patch_size]
			[bm3d_param.m_patch_size]
			[bm3d_param.m_patch_size]
			[bm3d_param.m_patch_size]
		);
	const image_t::pixel_t dct_2d_coeff_norm_coeff = 2.0 / bm3d_param.m_patch_size;
#pragma omp parallel for collapse (4)
	for (unsigned i = 0; i < bm3d_param.m_patch_size; ++i)
	{
		for (unsigned j = 0; j < bm3d_param.m_patch_size; ++j)
		{
			for (unsigned k = 0; k < bm3d_param.m_patch_size; ++k)
			{ 
				for (unsigned l = 0; l < bm3d_param.m_patch_size; ++l)
				{
					dct_2d_coeff_ndarr[i][j][k][l] =
						dct_2d_coeff_norm_coeff *
						cos((2 * k + 1) * i * M_PI / (2 * bm3d_param.m_patch_size)) *
						cos((2 * l + 1) * j * M_PI / (2 * bm3d_param.m_patch_size));
					if (i == 0)
					{
						dct_2d_coeff_ndarr[i][j][k][l] *= 1.0 / M_SQRT2;
					}
					if (j == 0)
					{
						dct_2d_coeff_ndarr[i][j][k][l] *= 1.0 / M_SQRT2;
					}
				} // dummy variable l
			} // dummy variable k
		} // j
	} // i
	l_dct_2d_coeff_ts = dct_2d_coeff_ndarr;
	CUDA_ERR_CHECK(cudaBindTexture(nullptr, l_tex_ref_dct_2d_coeff, l_dct_2d_coeff_ts.m_data_arr, l_dct_2d_coeff_ts.m_size * sizeof(image_t::pixel_t)));
	//**************************************************************************
#if __config_specific_op__
	assert(bm3d_param.m_patch_size == 4);
	//**************************************************************************
	cuda_precompute_dct
		<<<
			dim3
				(
					// # of patches along the x axis
					img_pixel_arr_dct_noisy.m_extents[2] / 2,
					// $ of patches along the y axis
					img_pixel_arr_dct_noisy.m_extents[1],
					// # of channels
					img_pixel_arr_dct_noisy.m_extents[0]
				)
			,
			dim3
				(
					bm3d_param.m_patch_size,
					bm3d_param.m_patch_size,
					2
				)
			,
			64 * sizeof(image_t::pixel_t)
		>>>
		(
			img_pixel_arr_dct_noisy.m_data_arr,
			img_pixel_arr_raw_noisy.m_data_arr,
			img_pixel_arr_raw_noisy.m_extents[2],
			img_pixel_arr_raw_noisy.m_extents[1]
		);
#else // !__config_specific_op__
	cuda_precompute_dct
		<<<
			dim3
				(
					// # of patches along the x axis
					img_pixel_arr_dct_noisy.m_extents[2],
					// # of patches along the y axis
					img_pixel_arr_dct_noisy.m_extents[1],
					// # of channels
					img_pixel_arr_dct_noisy.m_extents[0]
				)
			,
			dim3
				(
					bm3d_param.m_patch_size,
					bm3d_param.m_patch_size
				)
			,
			bm3d_param.m_patch_size * bm3d_param.m_patch_size * 2 * sizeof(image_t::pixel_t)
		>>>
		(
			img_pixel_arr_dct_noisy.m_data_arr,
			img_pixel_arr_raw_noisy.m_data_arr,
			img_pixel_arr_raw_noisy.m_extents[2], 
			img_pixel_arr_raw_noisy.m_extents[1]
		);
#endif // __config_specific_op__
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

/*******************************************************************************/
// Auxiliary Function for the 1st Step
/*******************************************************************************/

// @brief ceil the input value to the nearest power of 2, e.g. 15 -> 16
static __forceinline__ __host__ __device__ unsigned ceil_pow_2(const unsigned i)
{
	if (i <=    1) return    1;
	if (i <=    2) return    2;
	if (i <=    4) return    4;
	if (i <=    8) return    8;
	if (i <=   16) return   16;
	if (i <=   32) return   32;
	if (i <=   64) return   64;
	if (i <=  128) return  128;
	if (i <=  256) return  256;
	if (i <=  512) return  512;
	if (i <= 1024) return 1024;
	
	assert(0); return 0;
}

/*******************************************************************************/
// First Step: Compute the DCT and Sort the Distance Vector (Large Search Window Ver.)
/*******************************************************************************/

#define REGION____TOP__LEFT 0
#define REGION____TOP_RIGHT 1
#define REGION_BOTTOM__LEFT 2
#define REGION_BOTTOM_RIGHT 3

// @brief compute the distance vector based on DCT input, and sort the distance vector in ascending order
// Note that all these operations are ONLY done on the first channel
// @param (output) noisy_partial_dist_vec: partial sorted distance vector (based on noisy image)
// (shape: num_y_refs * num_x_refs * region * max_grp_size)
// @param ( input) img_pixel_arr_dct_noisy: DCT of noisy image
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) width & height: shape of the RAW pixel array
// @param ( input) ceil_blockDim_xy: ceiling of blockDim_xy to the nearest power of 2
// @param ( input) bm3d_param: BM3D parameters
static __global__ void cuda_compute_and_sort_partial_dct_dist_vec_1st_chnl_ht 
(
	                dist_t * const __restrict__ noisy_partial_dist_vec,
	const image_t::pixel_t * const __restrict__ img_pixel_arr_dct_noisy, 
	const unsigned width, const unsigned height, const unsigned ceil_blockDim_xy,
	const bm3d_param_t bm3d_param
)
{
	extern __shared__ char s_data_compute_and_sort_partial_dct_dist_vec_1st_chnl_ht[];

	const unsigned l_threadIdx = threadIdx.x + threadIdx.y * blockDim.x, blockDim_xy = blockDim.y * blockDim.x;
	//**************************************************************************
	// shared memory allocated for pixel_t values
	image_t::pixel_t * const s_data_ptr_pixel_t = (image_t::pixel_t *) s_data_compute_and_sort_partial_dct_dist_vec_1st_chnl_ht;
	// shared memory allocated for DCT distance
	          dist_t * const s_data_ptr_dist_t = (dist_t *)(s_data_ptr_pixel_t + (bm3d_param.m_patch_size * bm3d_param.m_patch_size + 1) / 2 * 2);
	// shared memory allocated for auxiliary data structure
	      aux_dist_t * const s_data_ptr_aux_dist_t = (aux_dist_t *)(s_data_ptr_dist_t + blockDim_xy);
	
	image_t::pixel_t * const s_reference_patch_dct = s_data_ptr_pixel_t;
	          dist_t * const s_dist_vec_raw_data   = s_data_ptr_dist_t;
	      aux_dist_t * const s_dist_vec_tmp_data   = s_data_ptr_aux_dist_t;
	//**************************************************************************
	// pre-compute the offsets of multi-dimensional array
	const unsigned img_pixel_arr_dct_noisy_offset[3] = 
		{
			(width - bm3d_param.m_patch_size + 2) * bm3d_param.m_patch_size * bm3d_param.m_patch_size,
			                                        bm3d_param.m_patch_size * bm3d_param.m_patch_size,
			                                                                  bm3d_param.m_patch_size
		};
	const unsigned partial_dist_vec_offset[3] = 
		{
			gridDim.x * gridDim.z * bm3d_param.m_max_grp_size,
			            gridDim.z * bm3d_param.m_max_grp_size,
			                        bm3d_param.m_max_grp_size
		};
	//**************************************************************************
	// compute the global coordinate of the starting point of the reference patch
	const unsigned refer_patch_g_coord_x = blockIdx.x * bm3d_param.m_reference_step;
	const unsigned refer_patch_g_coord_y = blockIdx.y * bm3d_param.m_reference_step;
	// this_patch_g_coord_* denotes global coordinate of the starting point of the patch of "this" thread
	unsigned this_patch_g_coord_x, this_patch_g_coord_y;
	// the following conditional statements take into account whether the window is located at the boundary or not
	if ((signed int)(refer_patch_g_coord_x - bm3d_param.m_hf_window_size) < 0)
	{
		if (blockIdx.z == REGION____TOP__LEFT || blockIdx.z == REGION_BOTTOM__LEFT) this_patch_g_coord_x = threadIdx.x             ;
		if (blockIdx.z == REGION____TOP_RIGHT || blockIdx.z == REGION_BOTTOM_RIGHT) this_patch_g_coord_x = threadIdx.x + blockDim.x;
	}
	else if (refer_patch_g_coord_x + bm3d_param.m_hf_window_size >=  width)
	{
		if (blockIdx.z == REGION____TOP__LEFT || blockIdx.z == REGION_BOTTOM__LEFT) this_patch_g_coord_x = threadIdx.x              +  width - 1 - 2 * bm3d_param.m_hf_window_size;
		if (blockIdx.z == REGION____TOP_RIGHT || blockIdx.z == REGION_BOTTOM_RIGHT) this_patch_g_coord_x = threadIdx.x + blockDim.x +  width - 1 - 2 * bm3d_param.m_hf_window_size;
	}
	else
	{
		if (blockIdx.z == REGION____TOP__LEFT || blockIdx.z == REGION_BOTTOM__LEFT) this_patch_g_coord_x = threadIdx.x              + refer_patch_g_coord_x - bm3d_param.m_hf_window_size;
		if (blockIdx.z == REGION____TOP_RIGHT || blockIdx.z == REGION_BOTTOM_RIGHT) this_patch_g_coord_x = threadIdx.x + blockDim.x + refer_patch_g_coord_x - bm3d_param.m_hf_window_size;
	}
	if ((signed int)(refer_patch_g_coord_y - bm3d_param.m_hf_window_size) < 0)
	{
		if (blockIdx.z == REGION____TOP__LEFT || blockIdx.z == REGION____TOP_RIGHT) this_patch_g_coord_y = threadIdx.y             ;
		if (blockIdx.z == REGION_BOTTOM__LEFT || blockIdx.z == REGION_BOTTOM_RIGHT) this_patch_g_coord_y = threadIdx.y + blockDim.y;
	}
	else
	if (refer_patch_g_coord_y + bm3d_param.m_hf_window_size >= height)
	{
		if (blockIdx.z == REGION____TOP__LEFT || blockIdx.z == REGION____TOP_RIGHT) this_patch_g_coord_y = threadIdx.y              + height - 1 - 2 * bm3d_param.m_hf_window_size;
		if (blockIdx.z == REGION_BOTTOM__LEFT || blockIdx.z == REGION_BOTTOM_RIGHT) this_patch_g_coord_y = threadIdx.y + blockDim.y + height - 1 - 2 * bm3d_param.m_hf_window_size;

	}
	else
	{
		if (blockIdx.z == REGION____TOP__LEFT || blockIdx.z == REGION____TOP_RIGHT) this_patch_g_coord_y = threadIdx.y              + refer_patch_g_coord_y - bm3d_param.m_hf_window_size;
		if (blockIdx.z == REGION_BOTTOM__LEFT || blockIdx.z == REGION_BOTTOM_RIGHT) this_patch_g_coord_y = threadIdx.y + blockDim.y + refer_patch_g_coord_y - bm3d_param.m_hf_window_size;
	}
	//**************************************************************************
	// initialize the shared memory value
	if (threadIdx.x < bm3d_param.m_patch_size && threadIdx.y < bm3d_param.m_patch_size)
	{
		s_reference_patch_dct[threadIdx.x + threadIdx.y * bm3d_param.m_patch_size] = 
			img_pixel_arr_dct_noisy
			[
				refer_patch_g_coord_y * img_pixel_arr_dct_noisy_offset[0] + 
				refer_patch_g_coord_x * img_pixel_arr_dct_noisy_offset[1] + 
			              threadIdx.y * img_pixel_arr_dct_noisy_offset[2] + 
			              threadIdx.x
			];
	} // if (threadIdx.x < bm3d_param.m_patch_size && threadIdx.y < bm3d_param.m_patch_size)
	if (l_threadIdx < ceil_blockDim_xy - blockDim_xy)
	{
		s_dist_vec_tmp_data[blockDim_xy + l_threadIdx].m_dist = FLT_MAX;
	}
	__syncthreads(); // end of initializing the shared memory value
	//**************************************************************************
	s_dist_vec_raw_data[l_threadIdx].m_this_patch_g_coord_x = this_patch_g_coord_x;
	s_dist_vec_raw_data[l_threadIdx].m_this_patch_g_coord_y = this_patch_g_coord_y;
	s_dist_vec_raw_data[l_threadIdx].m_dist = 0;
	// compute the raw distance vector (by "raw", we mean unsorted)
	for (unsigned i = 0; i < bm3d_param.m_patch_size; ++i)
	{
		for (unsigned j = 0; j < bm3d_param.m_patch_size; ++j)
		{
			image_t::pixel_t dct_val = 
				img_pixel_arr_dct_noisy
				[
					this_patch_g_coord_y * img_pixel_arr_dct_noisy_offset[0] + 
					this_patch_g_coord_x * img_pixel_arr_dct_noisy_offset[1] + 
					                   i * img_pixel_arr_dct_noisy_offset[2] + 
					                   j
				];
			image_t::pixel_t diff = dct_val - s_reference_patch_dct[j + i * bm3d_param.m_patch_size];
			
			s_dist_vec_raw_data[l_threadIdx].m_dist += diff * diff;
		} // patch local index x: j
	} // patch local index y: i
	//**************************************************************************
	for (unsigned i = 0; i < bm3d_param.m_max_grp_size; ++i)
	{
		s_dist_vec_tmp_data[l_threadIdx].m_l_threadIdx = l_threadIdx;
		s_dist_vec_tmp_data[l_threadIdx].m_dist        = s_dist_vec_raw_data[l_threadIdx].m_dist;
		__syncthreads();
		for (unsigned s = ceil_blockDim_xy / 2; s > 32; s /= 2)
		{
			if (l_threadIdx < s)
			{
				s_dist_vec_tmp_data[l_threadIdx] = 
					s_dist_vec_tmp_data[l_threadIdx] < s_dist_vec_tmp_data[l_threadIdx + s] ? 
					s_dist_vec_tmp_data[l_threadIdx] : s_dist_vec_tmp_data[l_threadIdx + s];
			}
			__syncthreads();
		}
		if (l_threadIdx < 32)
		{
			volatile aux_dist_t * sv_dist_vec_tmp_data = s_dist_vec_tmp_data;
			
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx + 32] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx + 32];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx + 16] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx + 16];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  8] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  8];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  4] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  4];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  2] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  2];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  1] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  1];
		}
		if (l_threadIdx == 0)
		{
			const unsigned min_l_threadIdx = s_dist_vec_tmp_data[0].m_l_threadIdx;
			
			noisy_partial_dist_vec
			[
				blockIdx.y * partial_dist_vec_offset[0] + 
				blockIdx.x * partial_dist_vec_offset[1] +
				blockIdx.z * partial_dist_vec_offset[2] + 
				         i
			] = s_dist_vec_raw_data[min_l_threadIdx];
			s_dist_vec_raw_data[min_l_threadIdx].m_dist = FLT_MAX;
		}
		__syncthreads();
	} // group index i
}

// @brief compute the distance vector based on raw input, and sort the distance vector in ascending order
// Note that all these operations are ONLY done on the first channel
// @param (output) basic_partial_dist_vec: partial sorted distance vector (based on basic estimate)
// (shape: num_y_refs * num_x_refs * region * max_grp_size)
// @param ( input) img_pixel_arr_raw_basic: basic estimate of the noisy image
// (shape: nchnls * height * width)
// @param ( input) width & height: shape of the raw pixel array
// @param ( input) ceil_blockDim_xy: ceiling of blockDim_xy to the nearest power of 2
// @param ( input) bm3d_param: BM3D parameters
static __global__ void cuda_compute_and_sort_partial_raw_dist_vec_1st_chnl_wn 
(
	                dist_t * const __restrict__ basic_partial_dist_vec,
	const image_t::pixel_t * const __restrict__ img_pixel_arr_raw_basic, 
	const unsigned width, const unsigned height, const unsigned ceil_blockDim_xy,
	const bm3d_param_t bm3d_param
)
{
	extern __shared__ char s_data_compute_and_sort_partial_raw_dist_vec_1st_chnl_wn[];

	const unsigned l_threadIdx = threadIdx.x + threadIdx.y * blockDim.x, blockDim_xy = blockDim.y * blockDim.x;
	const unsigned search_win_len = bm3d_param.m_hf_window_size + bm3d_param.m_patch_size / 2;
	//**************************************************************************
	// shared memory allocated for pixel_t values
	image_t::pixel_t * const s_data_ptr_pixel_t = (image_t::pixel_t *) s_data_compute_and_sort_partial_raw_dist_vec_1st_chnl_wn;
	// shared memory allocated for DCT distance
	          dist_t * const s_data_ptr_dist_t = (dist_t *)(s_data_ptr_pixel_t + (bm3d_param.m_patch_size * bm3d_param.m_patch_size + search_win_len * search_win_len + 1) / 2 * 2);
	// shared memory allocated for auxiliary data structure
	      aux_dist_t * const s_data_ptr_aux_dist_t = (aux_dist_t *)(s_data_ptr_dist_t + blockDim_xy);
	
	image_t::pixel_t * const s_reference_patch_raw = s_data_ptr_pixel_t;
	image_t::pixel_t * const s_img_pixel_arr_raw   = s_reference_patch_raw + bm3d_param.m_patch_size * bm3d_param.m_patch_size;
	          dist_t * const s_dist_vec_raw_data   = s_data_ptr_dist_t;
	      aux_dist_t * const s_dist_vec_tmp_data   = s_data_ptr_aux_dist_t;
	//**************************************************************************
	// pre-compute the offsets of multi-dimensional array
	const unsigned partial_dist_vec_offset[3] = 
		{
			gridDim.x * gridDim.z * bm3d_param.m_max_grp_size,
			            gridDim.z * bm3d_param.m_max_grp_size,
			                        bm3d_param.m_max_grp_size
		};
	//**************************************************************************
	// compute the global coordinate of the starting point of the reference patch
	const unsigned refer_patch_g_coord_x = blockIdx.x * bm3d_param.m_reference_step;
	const unsigned refer_patch_g_coord_y = blockIdx.y * bm3d_param.m_reference_step;
	// this_patch_g_coord_* denotes global coordinate of the starting point of the patch of "this" thread
	unsigned this_patch_g_coord_x, this_patch_g_coord_y;
	// the following conditional statements take into account whether the window is located at the boundary or not
	if ((signed int)(refer_patch_g_coord_x - bm3d_param.m_hf_window_size) < 0)
	{
		if (blockIdx.z == REGION____TOP__LEFT || blockIdx.z == REGION_BOTTOM__LEFT) this_patch_g_coord_x = threadIdx.x             ;
		if (blockIdx.z == REGION____TOP_RIGHT || blockIdx.z == REGION_BOTTOM_RIGHT) this_patch_g_coord_x = threadIdx.x + blockDim.x;
	}
	else if (refer_patch_g_coord_x + bm3d_param.m_hf_window_size >=  width)
	{
		if (blockIdx.z == REGION____TOP__LEFT || blockIdx.z == REGION_BOTTOM__LEFT) this_patch_g_coord_x = threadIdx.x              +  width - 1 - 2 * bm3d_param.m_hf_window_size;
		if (blockIdx.z == REGION____TOP_RIGHT || blockIdx.z == REGION_BOTTOM_RIGHT) this_patch_g_coord_x = threadIdx.x + blockDim.x +  width - 1 - 2 * bm3d_param.m_hf_window_size;
	}
	else
	{
		if (blockIdx.z == REGION____TOP__LEFT || blockIdx.z == REGION_BOTTOM__LEFT) this_patch_g_coord_x = threadIdx.x              + refer_patch_g_coord_x - bm3d_param.m_hf_window_size;
		if (blockIdx.z == REGION____TOP_RIGHT || blockIdx.z == REGION_BOTTOM_RIGHT) this_patch_g_coord_x = threadIdx.x + blockDim.x + refer_patch_g_coord_x - bm3d_param.m_hf_window_size;
	}
	if ((signed int)(refer_patch_g_coord_y - bm3d_param.m_hf_window_size) < 0)
	{
		if (blockIdx.z == REGION____TOP__LEFT || blockIdx.z == REGION____TOP_RIGHT) this_patch_g_coord_y = threadIdx.y             ;
		if (blockIdx.z == REGION_BOTTOM__LEFT || blockIdx.z == REGION_BOTTOM_RIGHT) this_patch_g_coord_y = threadIdx.y + blockDim.y;
	}
	else
	if (refer_patch_g_coord_y + bm3d_param.m_hf_window_size >= height)
	{
		if (blockIdx.z == REGION____TOP__LEFT || blockIdx.z == REGION____TOP_RIGHT) this_patch_g_coord_y = threadIdx.y              + height - 1 - 2 * bm3d_param.m_hf_window_size;
		if (blockIdx.z == REGION_BOTTOM__LEFT || blockIdx.z == REGION_BOTTOM_RIGHT) this_patch_g_coord_y = threadIdx.y + blockDim.y + height - 1 - 2 * bm3d_param.m_hf_window_size;

	}
	else
	{
		if (blockIdx.z == REGION____TOP__LEFT || blockIdx.z == REGION____TOP_RIGHT) this_patch_g_coord_y = threadIdx.y              + refer_patch_g_coord_y - bm3d_param.m_hf_window_size;
		if (blockIdx.z == REGION_BOTTOM__LEFT || blockIdx.z == REGION_BOTTOM_RIGHT) this_patch_g_coord_y = threadIdx.y + blockDim.y + refer_patch_g_coord_y - bm3d_param.m_hf_window_size;
	}
	//**************************************************************************
	// initialize the shared memory value
	if (threadIdx.x < bm3d_param.m_patch_size && threadIdx.y < bm3d_param.m_patch_size)
	{
		s_reference_patch_raw[threadIdx.x + threadIdx.y * bm3d_param.m_patch_size] = img_pixel_arr_raw_basic[(refer_patch_g_coord_x + threadIdx.x) + (refer_patch_g_coord_y + threadIdx.y) * width];
	}
	s_img_pixel_arr_raw[threadIdx.x + threadIdx.y * search_win_len] = img_pixel_arr_raw_basic[this_patch_g_coord_x + this_patch_g_coord_y * width];
	if (threadIdx.x == (blockDim.x - 1))
	{
		for (unsigned i = 1; i < bm3d_param.m_patch_size; ++i)
		{
			s_img_pixel_arr_raw[(threadIdx.x + i) + threadIdx.y * search_win_len] = img_pixel_arr_raw_basic[(this_patch_g_coord_x + i) + this_patch_g_coord_y * width];
		}
	}
	if (threadIdx.y == (blockDim.y - 1))
	{
		for (unsigned i = 1; i < bm3d_param.m_patch_size; ++i)
		{
			s_img_pixel_arr_raw[threadIdx.x + (threadIdx.y + i) * search_win_len] = img_pixel_arr_raw_basic[this_patch_g_coord_x + (this_patch_g_coord_y + i) * width];
		}
	}
	if (threadIdx.x == (blockDim.x - 1) && threadIdx.y == (blockDim.y - 1))
	{
		for (unsigned i = 1; i < bm3d_param.m_patch_size; ++i)
		{
			for (unsigned j = 1; j < bm3d_param.m_patch_size; ++j)
			{
				s_img_pixel_arr_raw[(threadIdx.x + j) + (threadIdx.y + i) * search_win_len] = img_pixel_arr_raw_basic[(this_patch_g_coord_x + j) + (this_patch_g_coord_y + i) * width];
			}
		}
	}
	if (l_threadIdx < ceil_blockDim_xy - blockDim_xy)
	{
		s_dist_vec_tmp_data[blockDim_xy + l_threadIdx].m_dist = FLT_MAX;
	}
	__syncthreads(); // end of the setup part of the code
	//**************************************************************************
	s_dist_vec_raw_data[l_threadIdx].m_this_patch_g_coord_x = this_patch_g_coord_x;
	s_dist_vec_raw_data[l_threadIdx].m_this_patch_g_coord_y = this_patch_g_coord_y;
	s_dist_vec_raw_data[l_threadIdx].m_dist = 0;
	// compute the raw distance vector (by "raw", we mean unsorted)
	for (unsigned i = 0; i < bm3d_param.m_patch_size; ++i)
	{
		for (unsigned j = 0; j < bm3d_param.m_patch_size; ++j)
		{
			image_t::pixel_t raw_val = s_img_pixel_arr_raw[(threadIdx.x + j) + (threadIdx.y + i) * search_win_len];
			image_t::pixel_t diff = raw_val - s_reference_patch_raw[j + i * bm3d_param.m_patch_size];
			
			s_dist_vec_raw_data[l_threadIdx].m_dist += diff * diff;
		} // patch local index x: j
	} // patch local index y: i
	//**************************************************************************
	for (unsigned i = 0; i < bm3d_param.m_max_grp_size; ++i)
	{
		s_dist_vec_tmp_data[l_threadIdx].m_l_threadIdx = l_threadIdx;
		s_dist_vec_tmp_data[l_threadIdx].m_dist        = s_dist_vec_raw_data[l_threadIdx].m_dist;
		__syncthreads();
		for (unsigned s = ceil_blockDim_xy / 2; s > 32; s /= 2)
		{
			if (l_threadIdx < s)
			{
				s_dist_vec_tmp_data[l_threadIdx] = 
					s_dist_vec_tmp_data[l_threadIdx] < s_dist_vec_tmp_data[l_threadIdx + s] ? 
					s_dist_vec_tmp_data[l_threadIdx] : s_dist_vec_tmp_data[l_threadIdx + s];
			}
			__syncthreads();
		}
		if (l_threadIdx < 32)
		{
			volatile aux_dist_t * sv_dist_vec_tmp_data = s_dist_vec_tmp_data;
			
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx + 32] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx + 32];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx + 16] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx + 16];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  8] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  8];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  4] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  4];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  2] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  2];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  1] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  1];
		}
		if (l_threadIdx == 0)
		{
			const unsigned min_l_threadIdx = s_dist_vec_tmp_data[0].m_l_threadIdx;
			
			basic_partial_dist_vec
			[
				blockIdx.y * partial_dist_vec_offset[0] + 
				blockIdx.x * partial_dist_vec_offset[1] +
				blockIdx.z * partial_dist_vec_offset[2] + 
				         i
			] = s_dist_vec_raw_data[min_l_threadIdx];
			s_dist_vec_raw_data[min_l_threadIdx].m_dist = FLT_MAX;
		}
		__syncthreads();
	} // group index i
}

#undef REGION____TOP__LEFT
#undef REGION____TOP_RIGHT
#undef REGION_BOTTOM__LEFT
#undef REGION_BOTTOM_RIGHT

// @brief pick up the smallest max_grp_size elements from all subregions
// @param (output) dist_vec: sorted distance vector
// (shape: num_y_refs * num_x_refs * max_grp_size)
// @param ( input) partial_dist_vec: partial sorted distance vector
// (shape: num_y_refs * num_x_refs * region * max_grp_size)
// @param ( input) bm3d_param: BM3D parameters
static __global__ void cuda_full_sort_dist_1st_chnl 
(
	      dist_t * const __restrict__         dist_vec,
	const dist_t * const __restrict__ partial_dist_vec,
	const bm3d_param_t bm3d_param
)
{
	const unsigned         dist_vec_offset[2] = 
		{
			gridDim.x * bm3d_param.m_max_grp_size,
			            bm3d_param.m_max_grp_size
		};
	const unsigned partial_dist_vec_offset[3] = 
		{
			gridDim.x * 4 * bm3d_param.m_max_grp_size,
			            4 * bm3d_param.m_max_grp_size,
			                bm3d_param.m_max_grp_size
		};
	      unsigned partial_dist_vec_count [4] = {0};
	//**************************************************************************
	for (unsigned i = 0; i < bm3d_param.m_max_grp_size; ++i)
	{
		// assume that the minimum distance comes from the top left sub-region
		image_t::pixel_t min_dist = 
			partial_dist_vec
			[
				blockIdx.y * partial_dist_vec_offset[0] + 
				blockIdx.x * partial_dist_vec_offset[1] + 
				             partial_dist_vec_count [0]
			].m_dist; unsigned min_index = 0;
		for (unsigned j = 1; j < 4; ++j)
		{
			if (min_dist > 
				partial_dist_vec
				[
					blockIdx.y * partial_dist_vec_offset[0] + 
					blockIdx.x * partial_dist_vec_offset[1] + 
				             j * partial_dist_vec_offset[2] + 
					             partial_dist_vec_count [j]
				].m_dist)
			{
				min_dist = 
					partial_dist_vec
					[
						blockIdx.y * partial_dist_vec_offset[0] + 
						blockIdx.x * partial_dist_vec_offset[1] + 
						         j * partial_dist_vec_offset[2] + 
						             partial_dist_vec_count [j]
					].m_dist;
				min_index = j;
			}
		}
		dist_vec
		[
			blockIdx.y * dist_vec_offset[0] + 
			blockIdx.x * dist_vec_offset[1] + 
			         i
		] = 
			partial_dist_vec
			[
				blockIdx.y * partial_dist_vec_offset[0] + 
				blockIdx.x * partial_dist_vec_offset[1] + 
				 min_index * partial_dist_vec_offset[2] + 
				partial_dist_vec_count[min_index]
			];
		++partial_dist_vec_count[min_index];
	}
}

// @brief compute the distance vector based on DCT input, and sort the distance vector in ascending order
// Note that all these operations are ONLY done on the first channel
// @param (output) noisy_dist_vec: sorted distance vector (based on noisy image)
// (shape: num_y_refs * num_x_refs * max_grp_size)
// @param ( input) img_pixel_arr_dct_noisy: DCT of noisy image
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) img_pixel_arr_raw_noisy: noisy image
// (shape: nchnls * height * width)
// @param ( input) bm3d_param: BM3D parameters
void bm3d_compute_and_sort_dct_dist_vec_1st_chnl_long_ht 
(
	      tensor_t <           dist_t, 3 > & noisy_dist_vec,
	const tensor_t < image_t::pixel_t, 5 > & img_pixel_arr_dct_noisy,
	const tensor_t < image_t::pixel_t, 3 > & img_pixel_arr_raw_noisy,
	const bm3d_param_t & bm3d_param
)
{
	unsigned partial_dist_vec_extents[4] = 
		{
			// # of reference patches along the y axis
			(img_pixel_arr_raw_noisy.m_extents[1] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
			// # of reference patches along the x axis
			(img_pixel_arr_raw_noisy.m_extents[2] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
			// # of sub-sections
			4,
			// maximum 3D group size
			bm3d_param.m_max_grp_size
		};
	tensor_t < dist_t, 4 > noisy_partial_dist_vec (partial_dist_vec_extents);
	//**************************************************************************
	const unsigned      blockDim_xy = 
		(bm3d_param.m_hf_window_size + 1 - bm3d_param.m_patch_size / 2) * 
		(bm3d_param.m_hf_window_size + 1 - bm3d_param.m_patch_size / 2);
	const unsigned ceil_blockDim_xy = ceil_pow_2(blockDim_xy);
	// due to the large size of the search window, we split the entire search window into 4 different sections
	// ------------------------------
	// |    top left |    top right |
	// |-----------------------------
	// | bottom left | bottom right |
	// ------------------------------
	cuda_compute_and_sort_partial_dct_dist_vec_1st_chnl_ht
		<<<
			dim3
				(
					// # of reference patches along the x axis
					(img_pixel_arr_raw_noisy.m_extents[2] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of reference patches along the y axis
					(img_pixel_arr_raw_noisy.m_extents[1] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of regions per reference block
					4
				)
			,
			dim3
				(
					bm3d_param.m_hf_window_size + 1 - bm3d_param.m_patch_size / 2,
					bm3d_param.m_hf_window_size + 1 - bm3d_param.m_patch_size / 2
				)
			,
			// dynamic shared memory allocation
			(bm3d_param.m_patch_size * bm3d_param.m_patch_size + 1) / 2 * 2 * sizeof(image_t::pixel_t) + 
			     blockDim_xy * sizeof(    dist_t) + 
			ceil_blockDim_xy * sizeof(aux_dist_t)
		>>>
		(
			noisy_partial_dist_vec.m_data_arr,
			img_pixel_arr_dct_noisy.m_data_arr, 
			img_pixel_arr_raw_noisy.m_extents[2],
			img_pixel_arr_raw_noisy.m_extents[1],
			ceil_blockDim_xy,
			bm3d_param
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
	//**************************************************************************
	// merge all the partially sorted distance vectors for a fully sorted one
	cuda_full_sort_dist_1st_chnl
		<<<
			dim3
				(
					// # of reference patches along the x axis
					(img_pixel_arr_raw_noisy.m_extents[2] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of reference patches along the y axis
					(img_pixel_arr_raw_noisy.m_extents[1] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step
				)
			,
			1
		>>>
		(
			noisy_dist_vec        .m_data_arr, 
			noisy_partial_dist_vec.m_data_arr,
			bm3d_param
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

// @brief compute the distance vector based on raw input, and sort the distance vector in ascending order
// Note that all these operations are ONLY done on the first channel
// @param (output) basic_dist_vec: sorted distance vector (based on basic estimate)
// (shape: num_y_refs * num_x_refs * max_grp_size)
// @param ( input) img_pixel_arr_raw_basic: basic estimate of the noisy image
// (shape: nchnls * height * width)
// @param ( input) bm3d_param: BM3D parameters
void bm3d_compute_and_sort_raw_dist_vec_1st_chnl_long_wn 
(
	      tensor_t <           dist_t, 3 > & basic_dist_vec,
	const tensor_t < image_t::pixel_t, 3 > & img_pixel_arr_raw_basic,
	const bm3d_param_t & bm3d_param
)
{
	unsigned partial_dist_vec_extents[4] = 
		{
			// # of reference patches along the y axis
			(img_pixel_arr_raw_basic.m_extents[1] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
			// # of reference patches along the x axis
			(img_pixel_arr_raw_basic.m_extents[2] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
			// # of sub-sections
			4,
			// maximum 3D group size
			bm3d_param.m_max_grp_size
		};
	tensor_t < dist_t, 4 > basic_partial_dist_vec (partial_dist_vec_extents);
	//**************************************************************************
	const unsigned      blockDim_xy = 
		(bm3d_param.m_hf_window_size + 1 - bm3d_param.m_patch_size / 2) * 
		(bm3d_param.m_hf_window_size + 1 - bm3d_param.m_patch_size / 2);
	const unsigned ceil_blockDim_xy = ceil_pow_2(blockDim_xy);
	// due to the large size of the search window, we split the entire search window into 4 different sections
	// ------------------------------
	// |    top left |    top right |
	// |-----------------------------
	// | bottom left | bottom right |
	// ------------------------------
	cuda_compute_and_sort_partial_raw_dist_vec_1st_chnl_wn
		<<<
			dim3
				(
					// # of reference patches along the x axis
					(img_pixel_arr_raw_basic.m_extents[2] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of reference patches along the y axis
					(img_pixel_arr_raw_basic.m_extents[1] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of regions per reference block
					4
				)
			,
			dim3
				(
					bm3d_param.m_hf_window_size + 1 - bm3d_param.m_patch_size / 2,
					bm3d_param.m_hf_window_size + 1 - bm3d_param.m_patch_size / 2
				)
			,
			// dynamic shared memory allocation
			(bm3d_param.m_patch_size * bm3d_param.m_patch_size + 
			 (bm3d_param.m_hf_window_size + bm3d_param.m_patch_size / 2) * 
			 (bm3d_param.m_hf_window_size + bm3d_param.m_patch_size / 2) + 1) / 2 * 2 * sizeof(image_t::pixel_t) + 
			     blockDim_xy * sizeof(    dist_t) + 
			ceil_blockDim_xy * sizeof(aux_dist_t)
		>>>
		(
			basic_partial_dist_vec.m_data_arr,
			img_pixel_arr_raw_basic.m_data_arr, 
			img_pixel_arr_raw_basic.m_extents[2],
			img_pixel_arr_raw_basic.m_extents[1],
			ceil_blockDim_xy,
			bm3d_param
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
	//**************************************************************************
	// merge all the partially sorted distance vectors for a fully sorted one
	cuda_full_sort_dist_1st_chnl
		<<<
			dim3
				(
					// # of reference patches along the x axis
					(img_pixel_arr_raw_basic.m_extents[2] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of reference patches along the y axis
					(img_pixel_arr_raw_basic.m_extents[1] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step
				)
			,
			1
		>>>
		(
			basic_dist_vec        .m_data_arr, 
			basic_partial_dist_vec.m_data_arr,
			bm3d_param
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

/*******************************************************************************/
// First Step: Compute the DCT and Sort the Distance Vector (Short Search Window Ver.)
/*******************************************************************************/

// @brief compute the distance vector based on DCT input, and sort the distance vector in ascending order
// Note that all these operations are ONLY done on the first channel
// @param (output) noisy_dist_vec: sorted distance vector (based on noisy image)
// (shape: num_y_refs * num_x_refs * max_grp_size)
// @param ( input) img_pixel_arr_dct_noisy: DCT of noisy image
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) width & height: shape of the RAW pixel array
// @param ( input) ceil_blockDim_xy: ceiling of blockDim_xy to the nearest power of 2
// @param ( input) bm3d_param: BM3D parameters
static __global__ void cuda_compute_and_sort_dct_dist_vec_1st_chnl_ht 
(
	                dist_t * const __restrict__ noisy_dist_vec,
	const image_t::pixel_t * const __restrict__ img_pixel_arr_dct_noisy, 
	const unsigned width, const unsigned height, const unsigned ceil_blockDim_xy,
	const bm3d_param_t bm3d_param
)
{
	extern __shared__ char s_data_compute_and_sort_dct_dist_vec_1st_chnl_ht[];

	const unsigned l_threadIdx = threadIdx.x + threadIdx.y * blockDim.x, blockDim_xy = blockDim.y * blockDim.x;
	//**************************************************************************
	// shared memory allocated for pixel_t values
	image_t::pixel_t * const s_data_ptr_pixel_t = (image_t::pixel_t *) s_data_compute_and_sort_dct_dist_vec_1st_chnl_ht;
	// shared memory allocated for DCT distance
	          dist_t * const s_data_ptr_dist_t = (dist_t *)(s_data_ptr_pixel_t + (bm3d_param.m_patch_size * bm3d_param.m_patch_size + 1) / 2 * 2);
	// shared memory allocated for auxiliary data structure
	      aux_dist_t * const s_data_ptr_aux_dist_t = (aux_dist_t *)(s_data_ptr_dist_t + blockDim_xy);
	
	image_t::pixel_t * const s_reference_patch_dct = s_data_ptr_pixel_t;
	          dist_t * const s_dist_vec_raw_data   = s_data_ptr_dist_t;
	      aux_dist_t * const s_dist_vec_tmp_data   = s_data_ptr_aux_dist_t;
	//**************************************************************************
	// pre-compute the offsets of multi-dimensional array
	const unsigned img_pixel_arr_dct_noisy_offset[3] = 
		{
			(width - bm3d_param.m_patch_size + 2) * bm3d_param.m_patch_size * bm3d_param.m_patch_size,
			                                        bm3d_param.m_patch_size * bm3d_param.m_patch_size,
			                                                                  bm3d_param.m_patch_size
		};
	const unsigned dist_vec_offset[2] = 
		{
			gridDim.x * bm3d_param.m_max_grp_size,
			            bm3d_param.m_max_grp_size
		};
	//**************************************************************************
	// compute the global coordinate of the starting point of the reference patch
	const unsigned refer_patch_g_coord_x = blockIdx.x * bm3d_param.m_reference_step;
	const unsigned refer_patch_g_coord_y = blockIdx.y * bm3d_param.m_reference_step;
	// this_patch_g_coord_* denotes global coordinate of the starting point of the patch of "this" thread
	unsigned this_patch_g_coord_x, this_patch_g_coord_y;
	// the following conditional statements take into account whether the window is located at the boundary or not
	if ((signed int)(refer_patch_g_coord_x - bm3d_param.m_hf_window_size) < 0)
	{
		this_patch_g_coord_x = threadIdx.x;
	}
	else if (refer_patch_g_coord_x + bm3d_param.m_hf_window_size >=  width)
	{
		this_patch_g_coord_x = threadIdx.x +  width - 1 - 2 * bm3d_param.m_hf_window_size;
	}
	else
	{
		this_patch_g_coord_x = threadIdx.x + refer_patch_g_coord_x - bm3d_param.m_hf_window_size;
	}
	if ((signed int)(refer_patch_g_coord_y - bm3d_param.m_hf_window_size) < 0)
	{
		this_patch_g_coord_y = threadIdx.y;
	}
	else
	if (refer_patch_g_coord_y + bm3d_param.m_hf_window_size >= height)
	{
		this_patch_g_coord_y = threadIdx.y + height - 1 - 2 * bm3d_param.m_hf_window_size;
	}
	else
	{
		this_patch_g_coord_y = threadIdx.y + refer_patch_g_coord_y - bm3d_param.m_hf_window_size;
	}
	//**************************************************************************
	// initialize the shared memory value
	if (l_threadIdx < bm3d_param.m_patch_size * bm3d_param.m_patch_size)
	{
		s_reference_patch_dct[l_threadIdx] = 
			img_pixel_arr_dct_noisy
			[
				refer_patch_g_coord_y * img_pixel_arr_dct_noisy_offset[0] + 
				refer_patch_g_coord_x * img_pixel_arr_dct_noisy_offset[1] + 
			              l_threadIdx
			];
	} // if (l_threadIdx < bm3d_param.m_patch_size * bm3d_param.m_patch_size)
	if (l_threadIdx < ceil_blockDim_xy - blockDim_xy)
	{
		s_dist_vec_tmp_data[blockDim_xy + l_threadIdx].m_dist = FLT_MAX;
	}
	__syncthreads(); // end of initializing the shared memory value
	//**************************************************************************
	s_dist_vec_raw_data[l_threadIdx].m_this_patch_g_coord_x = this_patch_g_coord_x;
	s_dist_vec_raw_data[l_threadIdx].m_this_patch_g_coord_y = this_patch_g_coord_y;
	s_dist_vec_raw_data[l_threadIdx].m_dist = 0;
	// compute the raw distance vector (by "raw", we mean unsorted)
	for (unsigned i = 0; i < bm3d_param.m_patch_size; ++i)
	{
		for (unsigned j = 0; j < bm3d_param.m_patch_size; ++j)
		{
			image_t::pixel_t dct_val = 
				img_pixel_arr_dct_noisy
				[
					this_patch_g_coord_y * img_pixel_arr_dct_noisy_offset[0] + 
					this_patch_g_coord_x * img_pixel_arr_dct_noisy_offset[1] + 
					                   i * img_pixel_arr_dct_noisy_offset[2] + 
					                   j
				];
			image_t::pixel_t diff = dct_val - s_reference_patch_dct[j + i * bm3d_param.m_patch_size];
			
			s_dist_vec_raw_data[l_threadIdx].m_dist += diff * diff;
		} // patch local index x: j
	} // patch local index y: i
	//**************************************************************************
	for (unsigned i = 0; i < bm3d_param.m_max_grp_size; ++i)
	{
		s_dist_vec_tmp_data[l_threadIdx].m_l_threadIdx = l_threadIdx;
		s_dist_vec_tmp_data[l_threadIdx].m_dist        = s_dist_vec_raw_data[l_threadIdx].m_dist;
		__syncthreads();
		for (unsigned s = ceil_blockDim_xy / 2; s > 32; s /= 2)
		{
			if (l_threadIdx < s)
			{
				s_dist_vec_tmp_data[l_threadIdx] = 
					s_dist_vec_tmp_data[l_threadIdx] < s_dist_vec_tmp_data[l_threadIdx + s] ? 
					s_dist_vec_tmp_data[l_threadIdx] : s_dist_vec_tmp_data[l_threadIdx + s];
			}
			__syncthreads();
		}
		if (l_threadIdx < 32)
		{
			volatile aux_dist_t * sv_dist_vec_tmp_data = s_dist_vec_tmp_data;
			
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx + 32] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx + 32];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx + 16] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx + 16];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  8] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  8];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  4] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  4];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  2] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  2];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  1] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  1];
		}
		if (l_threadIdx == 0)
		{
			const unsigned min_l_threadIdx = s_dist_vec_tmp_data[0].m_l_threadIdx;
			
			noisy_dist_vec
			[
				blockIdx.y * dist_vec_offset[0] + 
				blockIdx.x * dist_vec_offset[1] + 
				         i
			] = s_dist_vec_raw_data[min_l_threadIdx];
			s_dist_vec_raw_data[min_l_threadIdx].m_dist = FLT_MAX;
		}
		__syncthreads();
	} // group index i
}

// @brief compute the distance vector based on raw input, and sort the distance vector in ascending order
// Note that all these operations are ONLY done on the first channel
// @param (output) basic_dist_vec: sorted distance vector (based on basic estimate)
// (shape: num_y_refs * num_x_refs * max_grp_size)
// @param ( input) img_pixel_arr_raw_basic: basic estimate of the noisy image
// (shape: nchnls * height * width)
// @param ( input) width & height: shape of the raw pixel array
// @param ( input) ceil_blockDim_xy: ceiling of blockDim_xy to the nearest power of 2
// @param ( input) bm3d_param: BM3D parameters
static __global__ void cuda_compute_and_sort_raw_dist_vec_1st_chnl_wn 
(
	                dist_t * const __restrict__ basic_dist_vec,
	const image_t::pixel_t * const __restrict__ img_pixel_arr_raw_basic, 
	const unsigned width, const unsigned height, const unsigned ceil_blockDim_xy,
	const bm3d_param_t bm3d_param
)
{
	extern __shared__ char s_data_compute_raw_sort_dist_1st_chnl_short_wn[];

	const unsigned l_threadIdx = threadIdx.x + threadIdx.y * blockDim.x, blockDim_xy = blockDim.y * blockDim.x;
	const unsigned search_win_len = 2 * bm3d_param.m_hf_window_size + 1;
	//**************************************************************************
	// shared memory allocated for pixel_t values
	image_t::pixel_t * const s_data_ptr_pixel_t = (image_t::pixel_t *) s_data_compute_raw_sort_dist_1st_chnl_short_wn;
	// shared memory allocated for DCT distance
	          dist_t * const s_data_ptr_dist_t = (dist_t *)(s_data_ptr_pixel_t + (bm3d_param.m_patch_size * bm3d_param.m_patch_size + search_win_len * search_win_len + 1) / 2 * 2);
	// shared memory allocated for auxiliary data structure
	      aux_dist_t * const s_data_ptr_aux_dist_t = (aux_dist_t *)(s_data_ptr_dist_t + blockDim_xy);
	
	image_t::pixel_t * const s_reference_patch_raw = s_data_ptr_pixel_t;
	image_t::pixel_t * const s_img_pixel_arr_raw   = s_reference_patch_raw + bm3d_param.m_patch_size * bm3d_param.m_patch_size;
	          dist_t * const s_dist_vec_raw_data   = s_data_ptr_dist_t;
	      aux_dist_t * const s_dist_vec_tmp_data   = s_data_ptr_aux_dist_t;
	//**************************************************************************
	// pre-compute the offsets of multi-dimensional array
	const unsigned dist_vec_offset[2] = 
		{
			gridDim.x * bm3d_param.m_max_grp_size,
			            bm3d_param.m_max_grp_size 
		};
	//**************************************************************************
	// compute the global coordinate of the starting point of the reference patch
	const unsigned refer_patch_g_coord_x = blockIdx.x * bm3d_param.m_reference_step;
	const unsigned refer_patch_g_coord_y = blockIdx.y * bm3d_param.m_reference_step;
	// this_patch_g_coord_* denotes global coordinate of the starting point of the patch of "this" thread
	unsigned this_patch_g_coord_x, this_patch_g_coord_y;
	// the following conditional statements take into account whether the window is located at the boundary or not
	if ((signed int)(refer_patch_g_coord_x - bm3d_param.m_hf_window_size) < 0)
	{
		this_patch_g_coord_x = threadIdx.x;
		
	}
	else if (refer_patch_g_coord_x + bm3d_param.m_hf_window_size >=  width)
	{
		this_patch_g_coord_x = threadIdx.x +  width - 1 - 2 * bm3d_param.m_hf_window_size;
	}
	else
	{
		this_patch_g_coord_x = threadIdx.x + refer_patch_g_coord_x - bm3d_param.m_hf_window_size;
	}
	if ((signed int)(refer_patch_g_coord_y - bm3d_param.m_hf_window_size) < 0)
	{
		this_patch_g_coord_y = threadIdx.y;
	}
	else
	if (refer_patch_g_coord_y + bm3d_param.m_hf_window_size >= height)
	{
		this_patch_g_coord_y = threadIdx.y + height - 1 - 2 * bm3d_param.m_hf_window_size;
	}
	else
	{
		this_patch_g_coord_y = threadIdx.y + refer_patch_g_coord_y - bm3d_param.m_hf_window_size;
	}
	//**************************************************************************
	// initialize the shared memory value
	if (threadIdx.x < bm3d_param.m_patch_size && threadIdx.y < bm3d_param.m_patch_size)
	{
		s_reference_patch_raw[threadIdx.x + threadIdx.y * bm3d_param.m_patch_size] = img_pixel_arr_raw_basic[(refer_patch_g_coord_x + threadIdx.x) + (refer_patch_g_coord_y + threadIdx.y) * width];
	}
	s_img_pixel_arr_raw[threadIdx.x + threadIdx.y * search_win_len] = img_pixel_arr_raw_basic[this_patch_g_coord_x + this_patch_g_coord_y * width];
	if (threadIdx.x == (blockDim.x - 1))
	{
		for (unsigned i = 1; i < bm3d_param.m_patch_size; ++i)
		{
			s_img_pixel_arr_raw[(threadIdx.x + i) + threadIdx.y * search_win_len] = img_pixel_arr_raw_basic[(this_patch_g_coord_x + i) + this_patch_g_coord_y * width];
		}
	}
	if (threadIdx.y == (blockDim.y - 1))
	{
		for (unsigned i = 1; i < bm3d_param.m_patch_size; ++i)
		{
			s_img_pixel_arr_raw[threadIdx.x + (threadIdx.y + i) * search_win_len] = img_pixel_arr_raw_basic[this_patch_g_coord_x + (this_patch_g_coord_y + i) * width];
		}
	}
	if (threadIdx.x == (blockDim.x - 1) && threadIdx.y == (blockDim.y - 1))
	{
		for (unsigned i = 1; i < bm3d_param.m_patch_size; ++i)
		{
			for (unsigned j = 1; j < bm3d_param.m_patch_size; ++j)
			{
				s_img_pixel_arr_raw[(threadIdx.x + j) + (threadIdx.y + i) * search_win_len] = img_pixel_arr_raw_basic[(this_patch_g_coord_x + j) + (this_patch_g_coord_y + i) * width];
			}
		}
	}
	if (l_threadIdx < ceil_blockDim_xy - blockDim_xy)
	{
		s_dist_vec_tmp_data[blockDim_xy + l_threadIdx].m_dist = FLT_MAX;
	}
	__syncthreads(); // end of the setup part of the code
	//**************************************************************************
	s_dist_vec_raw_data[l_threadIdx].m_this_patch_g_coord_x = this_patch_g_coord_x;
	s_dist_vec_raw_data[l_threadIdx].m_this_patch_g_coord_y = this_patch_g_coord_y;
	s_dist_vec_raw_data[l_threadIdx].m_dist = 0;
	// compute the raw distance vector (by "raw", we mean unsorted)
	for (unsigned i = 0; i < bm3d_param.m_patch_size; ++i)
	{
		for (unsigned j = 0; j < bm3d_param.m_patch_size; ++j)
		{
			image_t::pixel_t raw_val = s_img_pixel_arr_raw[(threadIdx.x + j) + (threadIdx.y + i) * search_win_len];
			image_t::pixel_t diff = raw_val - s_reference_patch_raw[j + i * bm3d_param.m_patch_size];
			
			s_dist_vec_raw_data[l_threadIdx].m_dist += diff * diff;
		} // patch local index x: j
	} // patch local index y: i
	//**************************************************************************
	for (unsigned i = 0; i < bm3d_param.m_max_grp_size; ++i)
	{
		s_dist_vec_tmp_data[l_threadIdx].m_l_threadIdx = l_threadIdx;
		s_dist_vec_tmp_data[l_threadIdx].m_dist        = s_dist_vec_raw_data[l_threadIdx].m_dist;
		__syncthreads();
		for (unsigned s = ceil_blockDim_xy / 2; s > 32; s /= 2)
		{
			if (l_threadIdx < s)
			{
				s_dist_vec_tmp_data[l_threadIdx] = 
					s_dist_vec_tmp_data[l_threadIdx] < s_dist_vec_tmp_data[l_threadIdx + s] ? 
					s_dist_vec_tmp_data[l_threadIdx] : s_dist_vec_tmp_data[l_threadIdx + s];
			}
			__syncthreads();
		}
		if (l_threadIdx < 32)
		{
			volatile aux_dist_t * sv_dist_vec_tmp_data = s_dist_vec_tmp_data;
			
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx + 32] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx + 32];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx + 16] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx + 16];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  8] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  8];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  4] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  4];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  2] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  2];
			s_dist_vec_tmp_data[l_threadIdx] = s_dist_vec_tmp_data[l_threadIdx] < sv_dist_vec_tmp_data[l_threadIdx +  1] ? s_dist_vec_tmp_data[l_threadIdx] : sv_dist_vec_tmp_data[l_threadIdx +  1];
		}
		if (l_threadIdx == 0)
		{
			const unsigned min_l_threadIdx = s_dist_vec_tmp_data[0].m_l_threadIdx;
			
			basic_dist_vec
			[
				blockIdx.y * dist_vec_offset[0] + 
				blockIdx.x * dist_vec_offset[1] +
				         i
			] = s_dist_vec_raw_data[min_l_threadIdx];
			s_dist_vec_raw_data[min_l_threadIdx].m_dist = FLT_MAX;
		}
		__syncthreads();
	} // group index i
}

// @brief compute the distance vector based on DCT input, and sort the distance vector in ascending order
// Note that all these operations are ONLY done on the first channel
// @param (output) noisy_dist_vec: sorted distance vector (based on noisy image)
// (shape: num_y_refs * num_x_refs * max_grp_size)
// @param ( input) img_pixel_arr_dct_noisy: DCT of noisy image
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) img_pixel_arr_raw_noisy: noisy image
// (shape: nchnls * height * width)
// @param ( input) bm3d_param: BM3D parameters
void bm3d_compute_and_sort_dct_dist_vec_1st_chnl_short_ht
(
	      tensor_t <           dist_t, 3 > & noisy_dist_vec,
	const tensor_t < image_t::pixel_t, 5 > & img_pixel_arr_dct_noisy,
	const tensor_t < image_t::pixel_t, 3 > & img_pixel_arr_raw_noisy,
	const bm3d_param_t & bm3d_param
)
{
	const unsigned      blockDim_xy = 
		(2 * bm3d_param.m_hf_window_size + 2 - bm3d_param.m_patch_size) * 
		(2 * bm3d_param.m_hf_window_size + 2 - bm3d_param.m_patch_size);
	const unsigned ceil_blockDim_xy = ceil_pow_2(blockDim_xy);
	
	cuda_compute_and_sort_dct_dist_vec_1st_chnl_ht
		<<<
			dim3
				(
					// # of reference patches along the x axis
					(img_pixel_arr_raw_noisy.m_extents[2] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of reference patches along the y axis
					(img_pixel_arr_raw_noisy.m_extents[1] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step
				)
			,
			dim3
				(
					2 * bm3d_param.m_hf_window_size + 2 - bm3d_param.m_patch_size,
					2 * bm3d_param.m_hf_window_size + 2 - bm3d_param.m_patch_size
				)
			,
			// dynamic shared memory allocation
			(bm3d_param.m_patch_size * bm3d_param.m_patch_size + 1) / 2 * 2 * sizeof(image_t::pixel_t) + 
			     blockDim_xy * sizeof(    dist_t) + 
			ceil_blockDim_xy * sizeof(aux_dist_t)
		>>>
		(
			noisy_dist_vec.m_data_arr,
			img_pixel_arr_dct_noisy.m_data_arr, 
			img_pixel_arr_raw_noisy.m_extents[2],
			img_pixel_arr_raw_noisy.m_extents[1],
			ceil_blockDim_xy,
			bm3d_param
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

// @brief compute the distance vector based on raw input, and sort the distance vector in ascending order
// Note that all these operations are ONLY done on the first channel
// @param (output) basic_dist_vec: sorted distance vector (based on basic estimate)
// (shape: num_y_refs * num_x_refs * max_grp_size)
// @param ( input) img_pixel_arr_raw_basic: basic estimate of the noisy image
// (shape: nchnls * height * width)
// @param ( input) bm3d_param: BM3D parameters
void bm3d_compute_and_sort_raw_dist_vec_1st_chnl_short_wn 
(
	      tensor_t <           dist_t, 3 > & basic_dist_vec,
	const tensor_t < image_t::pixel_t, 3 > & img_pixel_arr_raw_basic,
	const bm3d_param_t & bm3d_param
)
{
	const unsigned      blockDim_xy = 
		(2 * bm3d_param.m_hf_window_size + 2 - bm3d_param.m_patch_size) * 
		(2 * bm3d_param.m_hf_window_size + 2 - bm3d_param.m_patch_size);
	const unsigned ceil_blockDim_xy = ceil_pow_2(blockDim_xy);
	
	cuda_compute_and_sort_raw_dist_vec_1st_chnl_wn
		<<<
			dim3
				(
					// # of reference patches along the x axis
					(img_pixel_arr_raw_basic.m_extents[2] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of reference patches along the y axis
					(img_pixel_arr_raw_basic.m_extents[1] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step
				)
			,
			dim3
				(
					2 * bm3d_param.m_hf_window_size + 2 - bm3d_param.m_patch_size,
					2 * bm3d_param.m_hf_window_size + 2 - bm3d_param.m_patch_size
				)
			,
			// dynamic shared memory allocation
			(bm3d_param.m_patch_size * bm3d_param.m_patch_size + 
			(2 * bm3d_param.m_hf_window_size + 1) * 
			(2 * bm3d_param.m_hf_window_size + 1) + 1) / 2 * 2 * sizeof(image_t::pixel_t) + 
			     blockDim_xy * sizeof(    dist_t) + 
			ceil_blockDim_xy * sizeof(aux_dist_t)
		>>>
		(
			basic_dist_vec.m_data_arr,
			img_pixel_arr_raw_basic.m_data_arr, 
			img_pixel_arr_raw_basic.m_extents[2],
			img_pixel_arr_raw_basic.m_extents[1],
			ceil_blockDim_xy,
			bm3d_param
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

/*******************************************************************************/
// Second Step: Form the 3D Group according to the Sorted Distance Vector
/*******************************************************************************/

// @brief filter distances of the first channel that match the threshold
// @param (output) num_of_matches_vec: number of matches for each reference patch (for ONLY the 1st chnl of noisy image)
// (shape: num_y_refs * num_x_refs)
// @param ( input) dist_vec: sorted distance vector obtained in previous step
// (shape: num_y_refs * num_x_refs * max_grp_size)
// @param ( input) bm3d_param: BM3D parameters
static __global__ void cuda_filter_matches_1st_chnl 
(
	        unsigned * const __restrict__ num_of_matches_vec,
	const     dist_t * const __restrict__ dist_vec,
	const bm3d_param_t bm3d_param
)
{
	extern __shared__ unsigned s_data_cuda_filter_matches_1st_chnl[];
	
	unsigned * const s_num_of_matches = s_data_cuda_filter_matches_1st_chnl;
	//**************************************************************************
	// initialize the content of the shared memory
	if (dist_vec[threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x].m_dist <= 
		bm3d_param.m_tau_match * bm3d_param.m_patch_size * bm3d_param.m_patch_size)
	{
		s_num_of_matches[threadIdx.x] = 1;
	}
	else
	{
		s_num_of_matches[threadIdx.x] = 0;
	}
	//**************************************************************************
	// increase the number of matches by 1 for EACH PATCH that matches the threshold
	volatile unsigned * sv_num_of_matches = s_num_of_matches;

	if (blockDim.x == 32)
	{
		if (threadIdx.x < 16)
		{
			sv_num_of_matches[threadIdx.x] += sv_num_of_matches[threadIdx.x + 16];
			sv_num_of_matches[threadIdx.x] += sv_num_of_matches[threadIdx.x +  8];
			sv_num_of_matches[threadIdx.x] += sv_num_of_matches[threadIdx.x +  4];
			sv_num_of_matches[threadIdx.x] += sv_num_of_matches[threadIdx.x +  2];
			sv_num_of_matches[threadIdx.x] += sv_num_of_matches[threadIdx.x +  1];
		}
	}
	if (blockDim.x == 16)
	{
		if (threadIdx.x < 8)
		{
			sv_num_of_matches[threadIdx.x] += sv_num_of_matches[threadIdx.x + 8];
			sv_num_of_matches[threadIdx.x] += sv_num_of_matches[threadIdx.x + 4];
			sv_num_of_matches[threadIdx.x] += sv_num_of_matches[threadIdx.x + 2];
			sv_num_of_matches[threadIdx.x] += sv_num_of_matches[threadIdx.x + 1];
		}
	}
	if (threadIdx.x == 0)
	{
		num_of_matches_vec[blockIdx.x + blockIdx.y * gridDim.x] = s_num_of_matches[0];
	}
}

// @brief get the number of matches for each reference patch
// @param (output) noisy_num_of_matches_vec: number of matches for each reference patch (for ONLY the 1st chnl of noisy image)
// (shape: num_y_refs * num_x_refs)
// @param ( input) noisy_dist_vec: sorted distance vector obtained in previous step
// (shape: num_y_refs * num_of_x_refs * max_grp_size)
// @param ( input) img_pixel_arr_dct_noisy: DCT of noisy image
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) bm3d_param: BM3D parameters
void bm3d_filter_matches_ht 
(
	      tensor_t <         unsigned, 2 > & noisy_num_of_matches_vec,
	const tensor_t <           dist_t, 3 > & noisy_dist_vec,
	const tensor_t < image_t::pixel_t, 5 > & img_pixel_arr_dct_noisy,
	const bm3d_param_t & bm3d_param
)
{
	cuda_filter_matches_1st_chnl
		<<<
			dim3
				(
					// # of reference patches along the x axis
					img_pixel_arr_dct_noisy.m_extents[2] / bm3d_param.m_reference_step,
					// # of reference patches along the y axis
					img_pixel_arr_dct_noisy.m_extents[1] / bm3d_param.m_reference_step
				)
			,
			bm3d_param.m_max_grp_size
			,
			bm3d_param.m_max_grp_size * sizeof(unsigned)
		>>>
		(
			noisy_num_of_matches_vec.m_data_arr,
			noisy_dist_vec          .m_data_arr,
			bm3d_param
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

// @brief get the number of matches for each reference patch, and at the same time, compute the DCT for all patches
// @param (output) img_pixel_arr_dct_basic: DCT of basic estimate
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param (output) basic_num_of_matches_vec: number of matches for each reference patch (for ONLY the 1st chnl of basic estimate)
// (shape: num_y_refs * num_x_refs)
// @param ( input) basic_dist_vec: sorted distance vector obtained in previous step
// (shape: num_y_refs * num_x_refs * max_grp_size)
// @param ( input) img_pixel_arr_raw_basic: basic estimate of noisy image
// (shape: nchnls * height * width)
// @param ( input) bm3d_param: BM3D parameters
void bm3d_filter_matches_compute_dct_wn 
(
	      tensor_t < image_t::pixel_t, 5 > & img_pixel_arr_dct_basic,
	      tensor_t <         unsigned, 2 > & basic_num_of_matches_vec,
	const tensor_t <           dist_t, 3 > & basic_dist_vec,
	const tensor_t < image_t::pixel_t, 3 > & img_pixel_arr_raw_basic,
	const bm3d_param_t & bm3d_param
)
{
	// pre-compute the DCT coefficients and bind them to the texture memory
	boost::multi_array < image_t::pixel_t, 4 > dct_2d_coeff_ndarr
		(
			boost::extents
			[bm3d_param.m_patch_size]
			[bm3d_param.m_patch_size]
			[bm3d_param.m_patch_size]
			[bm3d_param.m_patch_size]
		);
	const image_t::pixel_t dct_2d_coeff_norm_coeff = 2.0 / bm3d_param.m_patch_size;
#pragma omp parallel for collapse (4)
	for (unsigned i = 0; i < bm3d_param.m_patch_size; ++i)
	{
		for (unsigned j = 0; j < bm3d_param.m_patch_size; ++j)
		{
			for (unsigned k = 0; k < bm3d_param.m_patch_size; ++k)
			{
				for (unsigned l = 0; l < bm3d_param.m_patch_size; ++l)
				{
					dct_2d_coeff_ndarr[i][j][k][l] =
						dct_2d_coeff_norm_coeff *
						cos((2 * k + 1) * i * M_PI / (2 * bm3d_param.m_patch_size)) *
						cos((2 * l + 1) * j * M_PI / (2 * bm3d_param.m_patch_size));
					if (i == 0)
					{
						dct_2d_coeff_ndarr[i][j][k][l] *= 1.0 / M_SQRT2;
					}
					if (j == 0)
					{
						dct_2d_coeff_ndarr[i][j][k][l] *= 1.0 / M_SQRT2;
					}
				} // dummy variable l
			} // dummy variable k
		} // j
	} // i
	l_dct_2d_coeff_ts = dct_2d_coeff_ndarr;
	CUDA_ERR_CHECK(cudaBindTexture(nullptr, l_tex_ref_dct_2d_coeff, l_dct_2d_coeff_ts.m_data_arr, l_dct_2d_coeff_ts.m_size * sizeof(image_t::pixel_t)));
	//**************************************************************************
#if __config_specific_op__
	assert(bm3d_param.m_patch_size == 4);
	//**************************************************************************
	cuda_precompute_dct
		<<<
			dim3
				(
					// # of patches along the x axis
					img_pixel_arr_dct_basic.m_extents[2] / 2,
					// $ of patches along the y axis
					img_pixel_arr_dct_basic.m_extents[1],
					// # of channels
					img_pixel_arr_dct_basic.m_extents[0]
				)
			,
			dim3
				(
					bm3d_param.m_patch_size,
					bm3d_param.m_patch_size,
					2
				)
			,
			64 * sizeof(image_t::pixel_t)
		>>>
		(
			img_pixel_arr_dct_basic.m_data_arr,
			img_pixel_arr_raw_basic.m_data_arr,
			img_pixel_arr_raw_basic.m_extents[2],
			img_pixel_arr_raw_basic.m_extents[1]
		);
#else // !__config_specific_op__
	cuda_precompute_dct
		<<<
			dim3
				(
					// # of patches along the x axis
					img_pixel_arr_dct_basic.m_extents[2],
					// # of patches along the y axis
					img_pixel_arr_dct_basic.m_extents[1],
					// # of channels
					img_pixel_arr_dct_basic.m_extents[0]
				)
			,
			dim3
				(
					bm3d_param.m_patch_size,
					bm3d_param.m_patch_size
				)
			,
			bm3d_param.m_patch_size * bm3d_param.m_patch_size * 2 * sizeof(image_t::pixel_t)
		>>>
		(
			img_pixel_arr_dct_basic.m_data_arr,
			img_pixel_arr_raw_basic.m_data_arr,
			img_pixel_arr_raw_basic.m_extents[2], 
			img_pixel_arr_raw_basic.m_extents[1]
		);
#endif // __config_specific_op__
	cuda_filter_matches_1st_chnl
		<<<
			dim3
				(
					// # of reference patches along the x axis
					(img_pixel_arr_raw_basic.m_extents[2] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of reference patches along the y axis
					(img_pixel_arr_raw_basic.m_extents[1] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step
				)
			,
			bm3d_param.m_max_grp_size
			,
			bm3d_param.m_max_grp_size * sizeof(unsigned)
		>>>
		(
			basic_num_of_matches_vec.m_data_arr,
			basic_dist_vec          .m_data_arr,
			bm3d_param
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

/*******************************************************************************/
// Third Step: Hard Thresholding & Wiener Filtering
/*******************************************************************************/

// @brief floor the input value to the nearest power of 2, e.g. 15 -> 8
static __forceinline__ __host__ __device__ unsigned floor_pow_2(const unsigned i)
{
	if (i >= 16) return 16;
	if (i >=  8) return  8;
	if (i >=  4) return  4;
	if (i >=  2) return  2;
	if (i >=  1) return  1;
	
	return 0;
}

// @brief numerator / denominator
// @param (output) img_pixel_arr_output: numerator / denominator
// (shape: nchnls * height * width)
// @param ( input) numerator & denominator: stores aggregated weights
// (shape: nchnls * height * width)
static __global__ void cuda_numerator_div_denominator
(
	      image_t::pixel_t * const __restrict__ img_pixel_arr_output,
	const image_t::pixel_t * const __restrict__   numerator, 
	const image_t::pixel_t * const __restrict__ denominator,
	const image_t::pixel_t * const __restrict__ img_pixel_arr,
	const unsigned pixel_arr_size
)
{
	const unsigned l_threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (l_threadIdx < pixel_arr_size)
	{
		// assert(numerator[l_threadIdx] == numerator[l_threadIdx]);
		
		if (denominator[l_threadIdx] == 0.0)
		{
			img_pixel_arr_output[l_threadIdx] = img_pixel_arr[l_threadIdx];
		}
		else
		{
			img_pixel_arr_output[l_threadIdx] = numerator[l_threadIdx] / denominator[l_threadIdx];
		}
	}
}

// @brief apply the following calculations, in sequence
// 1. forward Hadamard Transform along the third dimension
// 2. Hard Thresholding
// 3. inverse Hadamard Transform along the third dimension
// 4. inverse Discrete Cosine Transform
// 5. weight aggregation
// @param (output) numerator & denominator: stores aggregated weights
// (shape: nchnls * height * width)
// @param ( input) img_pixel_arr_dct_noisy: DCT of noisy image
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) noisy_dist_vec: sorted distance vector (based on noisy image)
// (shape: num_y_refs * num_x_refs * max_grp_size)
// @param ( input) noisy_num_of_matches_vec: number of matches for each reference patch (for ONLY the 1st chnl of noisy image)
// (shape: num_y_refs * num_x_refs)
// @param ( input) sigma_table: standard deviation of each channel
// (shape: nchnls)
// @param ( input) width & height: shape of the RAW pixel array
// @param ( input) threshold: threshold value -> lambda
// @param ( input) bm3d_param: BM3D parameters
static __global__ void cuda_hadtf_hard_inv_hadtf_inv_dct
(
	      image_t::pixel_t * const __restrict__   numerator, 
	      image_t::pixel_t * const __restrict__ denominator,
	const image_t::pixel_t * const __restrict__ img_pixel_arr_dct_noisy,
	const           dist_t * const __restrict__ noisy_dist_vec,
	const         unsigned * const __restrict__ noisy_num_of_matches_vec,
	const image_t::pixel_t * const __restrict__ sigma_table,
	const unsigned width, const unsigned height,
	const image_t::pixel_t threshold, const bm3d_param_t bm3d_param
)
{
	extern __shared__ image_t::pixel_t s_data_hadtf_hard_inv_hadtf_inv_dct[];
	
	// number of nonzero coefficients after hard-thresholding (used for weight aggregation)
	const unsigned l_threadIdx_xy  =   threadIdx.x  + threadIdx.y * blockDim.x , blockDim_xy  = blockDim.y * blockDim.x ;
	const unsigned l_threadIdx_xyz = l_threadIdx_xy + threadIdx.z * blockDim_xy, blockDim_xyz = blockDim.z * blockDim_xy;
	// the following things are what we need to store in the shared memory
	// 1. 3D group based on noisy image
	// (total size: max_grp_size * patch_size * patch_size)
	// 2. Hadamard-Transformed 3D group
	// (total size: max_grp_size * patch_size * patch_size)
	// 3. flag that indicates whether the local coefficient is zero or not
	// (total size: max_grp_size * patch_size * patch_size)
	// 4. inverse-Hadamard-Transformed 3D group
	// (total size: max_grp_size * patch_size * patch_size)
	// 5. inverse-DCT 3D group
	// (total size: max_grp_size * patch_size * patch_size)
	image_t::pixel_t * const s_img_pixel_arr_noisy_match = s_data_hadtf_hard_inv_hadtf_inv_dct;
	image_t::pixel_t * const s_img_pixel_arr_noisy_hadtf = s_img_pixel_arr_noisy_match + blockDim_xyz;
	image_t::pixel_t * const s_nonzero_coeff_flag        = s_img_pixel_arr_noisy_hadtf + blockDim_xyz; 
	image_t::pixel_t * const s_img_pixel_arr_inv_hadtf   = s_nonzero_coeff_flag        + blockDim_xyz;
	image_t::pixel_t * const s_img_pixel_arr_inv_dct     = s_img_pixel_arr_inv_hadtf   + blockDim_xyz;
#if __config_specific_op__
	image_t::pixel_t * const s_had_tf_coeff              = s_img_pixel_arr_inv_dct     + blockDim_xyz;
#endif // __config_specific_op__
	//**************************************************************************
	// floor the number of matches to the closest power of 2
	const unsigned num_of_matches = floor_pow_2(noisy_num_of_matches_vec[blockIdx.x + blockIdx.y * gridDim.x]);
	// pre-compute the offsets of multi-dimensional array
	const unsigned l_tex_ref_dct_2d_coeff_offset[3] = 
		{
			bm3d_param.m_patch_size * bm3d_param.m_patch_size * bm3d_param.m_patch_size,
			                          bm3d_param.m_patch_size * bm3d_param.m_patch_size,
			                                                    bm3d_param.m_patch_size
		};
#if __config_specific_op__
	
#else
	const unsigned l_tex_ref_had_tf_coeff_offset[2] = 
		{
			bm3d_param.m_max_grp_size * bm3d_param.m_max_grp_size,
			                            bm3d_param.m_max_grp_size
		};
#endif // __config_specific_op__
	const unsigned img_pixel_arr_dct_offset[3] = 
		{
			(height - bm3d_param.m_patch_size + 2) * (width - bm3d_param.m_patch_size + 2) * blockDim_xy,
			                                         (width - bm3d_param.m_patch_size + 2) * blockDim_xy,
			                                                                                 blockDim_xy
		};
	//**************************************************************************
	// initialize the content of the shared memory
	const dist_t dist = noisy_dist_vec[threadIdx.z + blockIdx.x * blockDim.z + blockIdx.y * gridDim.x * blockDim.z];
	const unsigned this_patch_g_coord_x = dist.m_this_patch_g_coord_x;
	const unsigned this_patch_g_coord_y = dist.m_this_patch_g_coord_y;
	s_img_pixel_arr_noisy_match[l_threadIdx_xyz] = 
		img_pixel_arr_dct_noisy
		[
			                 blockIdx.z * img_pixel_arr_dct_offset[0] + 
			dist.m_this_patch_g_coord_y * img_pixel_arr_dct_offset[1] + 
			dist.m_this_patch_g_coord_x * img_pixel_arr_dct_offset[2] + 
			              l_threadIdx_xy
		];
	s_img_pixel_arr_noisy_hadtf[l_threadIdx_xyz] = 0.0;
	s_nonzero_coeff_flag       [l_threadIdx_xyz] = 0.0;
	s_img_pixel_arr_inv_hadtf  [l_threadIdx_xyz] = 0.0;
	s_img_pixel_arr_inv_dct    [l_threadIdx_xyz] = 0.0;
#if __config_specific_op__
	s_had_tf_coeff             [l_threadIdx_xyz] = tex1Dfetch(l_tex_ref_had_tf_coeff, (num_of_matches - 1) * blockDim_xyz + l_threadIdx_xyz);
#endif // __config_specific_op__
	__syncthreads(); // end of initializing the shared memory
	//**************************************************************************
	// 1. forward Hadamard Transform along the third dimension
#if __config_specific_op__
	for (unsigned i = 0; i < bm3d_param.m_max_grp_size; ++i) // dummy variable
	{
		s_img_pixel_arr_noisy_hadtf[l_threadIdx_xyz] += s_img_pixel_arr_noisy_match[l_threadIdx_xy + i * blockDim_xy] * s_had_tf_coeff[i + threadIdx.z * blockDim_xy];
	} // dummy variable i
#else
	for (unsigned i = 0; i < bm3d_param.m_max_grp_size; ++i) // dummy variable
	{
		s_img_pixel_arr_noisy_hadtf[l_threadIdx_xyz] += s_img_pixel_arr_noisy_match[l_threadIdx_xy + i * blockDim_xy] * 
			tex1Dfetch
				(
					l_tex_ref_had_tf_coeff, 
					(num_of_matches - 1) * l_tex_ref_had_tf_coeff_offset[0] + 
					        threadIdx.z  * l_tex_ref_had_tf_coeff_offset[1] + 
			                          i
				);
	} // dummy variable i
#endif // __config_specific_op__
	//**************************************************************************
	// 2. Hard Thresholding
	if (fabs(s_img_pixel_arr_noisy_hadtf[l_threadIdx_xyz]) <= threshold * sigma_table[blockIdx.z])
	{
		// if the value is below certain zero, make it zero
		s_img_pixel_arr_noisy_hadtf[l_threadIdx_xyz] = 0.0;
	}
	else
	{
		// count the number of nonzero coefficients for the entire 3D block
		s_nonzero_coeff_flag[l_threadIdx_xyz] = 1.0;
	}
	__syncthreads(); // end of Hard Thresholding
	// reduce the nonzero coefficient flag into the number of nonzero coefficients
	for (unsigned s = blockDim_xyz / 2; s > 32; s /= 2)
	{
		if (l_threadIdx_xyz < s)
		{
			s_nonzero_coeff_flag[l_threadIdx_xyz] += s_nonzero_coeff_flag[l_threadIdx_xyz + s];
		}
		__syncthreads();
	}
	if (l_threadIdx_xyz < 32)
	{
		volatile image_t::pixel_t * sv_nonzero_coeff_flag = s_nonzero_coeff_flag;
		
		s_nonzero_coeff_flag[l_threadIdx_xyz] += sv_nonzero_coeff_flag[l_threadIdx_xyz + 32];
		s_nonzero_coeff_flag[l_threadIdx_xyz] += sv_nonzero_coeff_flag[l_threadIdx_xyz + 16];
		s_nonzero_coeff_flag[l_threadIdx_xyz] += sv_nonzero_coeff_flag[l_threadIdx_xyz +  8];
		s_nonzero_coeff_flag[l_threadIdx_xyz] += sv_nonzero_coeff_flag[l_threadIdx_xyz +  4];
		s_nonzero_coeff_flag[l_threadIdx_xyz] += sv_nonzero_coeff_flag[l_threadIdx_xyz +  2];
		s_nonzero_coeff_flag[l_threadIdx_xyz] += sv_nonzero_coeff_flag[l_threadIdx_xyz +  1];
	}
	//**************************************************************************
	// 3. inverse Hadamard Transform along the third dimension
#if __config_specific_op__
	for (unsigned i = 0; i < bm3d_param.m_max_grp_size; ++i) // dummy variable
	{
		s_img_pixel_arr_inv_hadtf[l_threadIdx_xyz] += s_img_pixel_arr_noisy_hadtf[l_threadIdx_xy + i * blockDim_xy] * s_had_tf_coeff[threadIdx.z + i * blockDim_xy];
	} // dummy variable i
#else
	for (unsigned i = 0; i < bm3d_param.m_max_grp_size; ++i) // dummy variable
	{
		s_img_pixel_arr_inv_hadtf[l_threadIdx_xyz] += s_img_pixel_arr_noisy_hadtf[l_threadIdx_xy + i * blockDim_xy] *
			tex1Dfetch
				(
					l_tex_ref_had_tf_coeff, 
					(num_of_matches - 1) * l_tex_ref_had_tf_coeff_offset[0] + 
					                  i  * l_tex_ref_had_tf_coeff_offset[1] + 
					        threadIdx.z
				);
	} // dummy variable i
#endif // __config_specific_op__
#if __config_specific_op__
	
#else // __config_specific_op__
	__syncthreads(); // end of the inverse Hadamard Transform
#endif // __config_specific_op__
	//**************************************************************************
	// 4. inverse Discrete Cosine Transform
	for (unsigned i = 0; i < bm3d_param.m_patch_size; ++i) // dummy variable
	{
		for (unsigned j = 0; j < bm3d_param.m_patch_size; ++j) // dummy variable
		{
			s_img_pixel_arr_inv_dct[l_threadIdx_xyz] += s_img_pixel_arr_inv_hadtf[j + i * blockDim.x + threadIdx.z * blockDim_xy] * 
				tex1Dfetch
					(
						l_tex_ref_dct_2d_coeff, 
						          i * l_tex_ref_dct_2d_coeff_offset[0] +
						          j * l_tex_ref_dct_2d_coeff_offset[1] + 
						threadIdx.y * l_tex_ref_dct_2d_coeff_offset[2] + 
						threadIdx.x
					);
		} // dummy variable j
	} // dummy variable i
	//**************************************************************************
	// 5. weight aggregation
	const image_t::pixel_t kaiser_win_coeff = threadIdx.z < num_of_matches ? tex1Dfetch(l_tex_ref_kaiser_win_coeff, threadIdx.x + threadIdx.y * blockDim.x) : 0;
	
	image_t::pixel_t weight_coeff;
		
	if (s_nonzero_coeff_flag[0] == 0)
	{	
		weight_coeff = 1.0;
	}
	else
	{
		weight_coeff = 1.0 / s_nonzero_coeff_flag[0];
	}
	//**************************************************************************
	atomicAdd
		(
			&  numerator
			[
				                         blockIdx.z  * height * width + 
				(this_patch_g_coord_y + threadIdx.y)          * width +
				(this_patch_g_coord_x + threadIdx.x)
			], weight_coeff * kaiser_win_coeff * s_img_pixel_arr_inv_dct[l_threadIdx_xyz]
		);
	atomicAdd
		(
			&denominator
			[
										 blockIdx.z  * height * width + 
				(this_patch_g_coord_y + threadIdx.y)          * width + 
				(this_patch_g_coord_x + threadIdx.x)
			], weight_coeff * kaiser_win_coeff
		);
}

// @brief apply the following calculations, in sequence
// 1. forward Hadamard Transform along the third dimension
// 2. Wiener Filtering
// 3. inverse Hadamard Transform along the third dimension
// 4. inverse Discrete Cosine Transform
// 5. weight aggregation
// @param (output) numerator & denominator: stores aggregated weights
// (shape: nchnls * height * width)
// @param ( input) img_pixel_arr_dct_basic: DCT of basic estimate
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) img_pixel_arr_dct_noisy: DCT of noisy image
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) basic_dist_vec: sorted distance vector (based on basic estimate)
// (shape: num_y_refs * num_x_refs * max_grp_size)
// @param ( input) basic_num_of_matches_vec: number of matches for each reference patch (for ONLY the 1st chnl of basic estimate)
// (shape: num_y_refs * num_x_refs)
// @param ( input) sigma_table: standard deviation of each channel
// (shape: nchnls)
// @param ( input) width & height: shape of the raw pixel array
// @param ( input) bm3d_param: BM3D parameters
static __global__ void cuda_hadtf_wiener_inv_hadtf_inv_dct
(
	      image_t::pixel_t * const __restrict__   numerator, 
	      image_t::pixel_t * const __restrict__ denominator,
	const image_t::pixel_t * const __restrict__ img_pixel_arr_dct_basic,
	const image_t::pixel_t * const __restrict__ img_pixel_arr_dct_noisy,
	const           dist_t * const __restrict__ basic_dist_vec,
	const         unsigned * const __restrict__ basic_num_of_matches_vec,
	const image_t::pixel_t * const __restrict__ sigma_table, 
	const unsigned width, const unsigned height,
	const bm3d_param_t bm3d_param
)
{
	extern __shared__ image_t::pixel_t s_data_hadtf_wiener_inv_hadtf_inv_dct[];
	
	const unsigned l_threadIdx_xy  =   threadIdx.x  + threadIdx.y * blockDim.x , blockDim_xy  = blockDim.y * blockDim.x ;
	const unsigned l_threadIdx_xyz = l_threadIdx_xy + threadIdx.z * blockDim_xy, blockDim_xyz = blockDim.z * blockDim_xy;
	// the following things are what we need to store in the shared memory
	// 1. 3D group based on basic estimate
	// (total size: max_grp_size * patch_size * patch_size)
	// 2. 3D group based on noisy image
	// (total size: max_grp_size * patch_size * patch_size)
	// 3. Hadamard-Transformed 3D group (basic)
	// (total size: max_grp_size * patch_size * patch_size)
	// 4. Hadamard-Transformed 3D group (noisy)
	// (total size: max_grp_size * patch_size * patch_size)
	// 5. Wiener weight coefficient
	// (total size: max_grp_size * patch_size * patch_size)
	// 6. inverse-Hadamard-Transformed 3D group
	// (total size: max_grp_size * patch_size * patch_size)
	// 7. inverse-DCT 3D group
	// (total size: max_grp_size * patch_size * patch_size)
	image_t::pixel_t * const s_img_pixel_arr_basic_match = s_data_hadtf_wiener_inv_hadtf_inv_dct;
	image_t::pixel_t * const s_img_pixel_arr_noisy_match = s_img_pixel_arr_basic_match + blockDim_xyz;
	image_t::pixel_t * const s_img_pixel_arr_basic_hadtf = s_img_pixel_arr_noisy_match + blockDim_xyz;
	image_t::pixel_t * const s_img_pixel_arr_noisy_hadtf = s_img_pixel_arr_basic_hadtf + blockDim_xyz;
	image_t::pixel_t * const s_wiener_weight_coeff       = s_img_pixel_arr_noisy_hadtf + blockDim_xyz;
	image_t::pixel_t * const s_img_pixel_arr_inv_hadtf   = s_wiener_weight_coeff       + blockDim_xyz;
	image_t::pixel_t * const s_img_pixel_arr_inv_dct     = s_img_pixel_arr_inv_hadtf   + blockDim_xyz;
#if __config_specific_op__
	image_t::pixel_t * const s_had_tf_coeff              = s_img_pixel_arr_inv_dct     + blockDim_xyz;
#endif // __config_specific_op__
	//**************************************************************************
	// floor the number of matches to the closest power of 2
	const unsigned num_of_matches = floor_pow_2(basic_num_of_matches_vec[blockIdx.x + blockIdx.y * gridDim.x]);
	// pre-compute the offsets of multi-dimensional array
	const unsigned l_tex_ref_dct_2d_coeff_offset[3] = 
		{
			bm3d_param.m_patch_size * bm3d_param.m_patch_size * bm3d_param.m_patch_size,
			                          bm3d_param.m_patch_size * bm3d_param.m_patch_size,
			                                                    bm3d_param.m_patch_size
		};
#if __config_specific_op__
	
#else
	const unsigned l_tex_ref_had_tf_coeff_offset[2] = 
		{
			bm3d_param.m_max_grp_size * bm3d_param.m_max_grp_size,
			                            bm3d_param.m_max_grp_size
		};
#endif // __config_specific_op__
	const unsigned img_pixel_arr_dct_offset[3] = 
		{
			(height - bm3d_param.m_patch_size + 2) * (width - bm3d_param.m_patch_size + 2) * blockDim_xy,
			                                         (width - bm3d_param.m_patch_size + 2) * blockDim_xy,
			                                                                                 blockDim_xy
		};
	//**************************************************************************
	// initialize the content of the shared memory
	const dist_t dist = basic_dist_vec[threadIdx.z + blockIdx.x * blockDim.z + blockIdx.y * gridDim.x * blockDim.z];
	const unsigned this_patch_g_coord_x = dist.m_this_patch_g_coord_x;
	const unsigned this_patch_g_coord_y = dist.m_this_patch_g_coord_y;
	s_img_pixel_arr_basic_match[l_threadIdx_xyz] = 
		img_pixel_arr_dct_basic
		[
			                 blockIdx.z * img_pixel_arr_dct_offset[0] + 
			dist.m_this_patch_g_coord_y * img_pixel_arr_dct_offset[1] + 
			dist.m_this_patch_g_coord_x * img_pixel_arr_dct_offset[2] + 
			              l_threadIdx_xy
		];
	s_img_pixel_arr_noisy_match[l_threadIdx_xyz] = 
		img_pixel_arr_dct_noisy
		[
			                 blockIdx.z * img_pixel_arr_dct_offset[0] + 
			dist.m_this_patch_g_coord_y * img_pixel_arr_dct_offset[1] + 
			dist.m_this_patch_g_coord_x * img_pixel_arr_dct_offset[2] + 
			              l_threadIdx_xy
		];
	s_img_pixel_arr_basic_hadtf[l_threadIdx_xyz] = 0.0;
	s_img_pixel_arr_noisy_hadtf[l_threadIdx_xyz] = 0.0;
	s_img_pixel_arr_inv_hadtf  [l_threadIdx_xyz] = 0.0;
	s_img_pixel_arr_inv_dct    [l_threadIdx_xyz] = 0.0;
#if __config_specific_op__
	s_had_tf_coeff             [l_threadIdx_xyz] = tex1Dfetch(l_tex_ref_had_tf_coeff, (num_of_matches - 1) * blockDim_xyz + l_threadIdx_xyz);
#endif // __config_specific_op__
	__syncthreads(); // end of initializing the shared memory
	//**************************************************************************
	// 1. forward Hadamard Transform along the third dimension
#if __config_specific_op__
	for (unsigned i = 0; i < bm3d_param.m_max_grp_size; ++i) // dummy variable
	{
		s_img_pixel_arr_basic_hadtf[l_threadIdx_xyz] += s_img_pixel_arr_basic_match[l_threadIdx_xy + i * blockDim_xy] * s_had_tf_coeff[i + threadIdx.z * blockDim_xy];
		s_img_pixel_arr_noisy_hadtf[l_threadIdx_xyz] += s_img_pixel_arr_noisy_match[l_threadIdx_xy + i * blockDim_xy] * s_had_tf_coeff[i + threadIdx.z * blockDim_xy];
	} // dummy variable i
#else
	for (unsigned i = 0; i < bm3d_param.m_max_grp_size; ++i) // dummy variable
	{
		s_img_pixel_arr_basic_hadtf[l_threadIdx_xyz] += s_img_pixel_arr_basic_match[l_threadIdx_xy + i * blockDim_xy] * 
			tex1Dfetch
				(
					l_tex_ref_had_tf_coeff,
					(num_of_matches - 1) * l_tex_ref_had_tf_coeff_offset[0] + 
					        threadIdx.z  * l_tex_ref_had_tf_coeff_offset[1] + 
					                  i
				);
		s_img_pixel_arr_noisy_hadtf[l_threadIdx_xyz] += s_img_pixel_arr_noisy_match[l_threadIdx_xy + i * blockDim_xy] * 
			tex1Dfetch
				(
					l_tex_ref_had_tf_coeff,
					(num_of_matches - 1) * l_tex_ref_had_tf_coeff_offset[0] + 
					        threadIdx.z  * l_tex_ref_had_tf_coeff_offset[1] + 
					                  i
				);
	} // dummy variable i
#endif // __confic_specific_op__
	//**************************************************************************
	// 2. Wiener Filtering
	s_wiener_weight_coeff[l_threadIdx_xyz] = 
	 	(s_img_pixel_arr_basic_hadtf[l_threadIdx_xyz] * s_img_pixel_arr_basic_hadtf[l_threadIdx_xyz]) / 
	 	(s_img_pixel_arr_basic_hadtf[l_threadIdx_xyz] * s_img_pixel_arr_basic_hadtf[l_threadIdx_xyz] + sigma_table[blockIdx.z] * sigma_table[blockIdx.z]);
	s_img_pixel_arr_noisy_hadtf[l_threadIdx_xyz] *= s_wiener_weight_coeff[l_threadIdx_xyz];
	//**************************************************************************
	s_wiener_weight_coeff[l_threadIdx_xyz] = s_wiener_weight_coeff[l_threadIdx_xyz] * s_wiener_weight_coeff[l_threadIdx_xyz];
	__syncthreads();
	// wiener weight coefficient L2 norm reduction
	for (unsigned s = blockDim_xyz / 2; s > 32; s /= 2)
	{
		if (l_threadIdx_xyz < s)
		{
			s_wiener_weight_coeff[l_threadIdx_xyz] += s_wiener_weight_coeff[l_threadIdx_xyz + s];
		}
		__syncthreads();
	}
	if (l_threadIdx_xyz < 32)
	{
		volatile image_t::pixel_t * sv_wiener_weight_coeff = s_wiener_weight_coeff;
		
		s_wiener_weight_coeff[l_threadIdx_xyz] += sv_wiener_weight_coeff[l_threadIdx_xyz + 32];
		s_wiener_weight_coeff[l_threadIdx_xyz] += sv_wiener_weight_coeff[l_threadIdx_xyz + 16];
		s_wiener_weight_coeff[l_threadIdx_xyz] += sv_wiener_weight_coeff[l_threadIdx_xyz +  8];
		s_wiener_weight_coeff[l_threadIdx_xyz] += sv_wiener_weight_coeff[l_threadIdx_xyz +  4];
		s_wiener_weight_coeff[l_threadIdx_xyz] += sv_wiener_weight_coeff[l_threadIdx_xyz +  2];
		s_wiener_weight_coeff[l_threadIdx_xyz] += sv_wiener_weight_coeff[l_threadIdx_xyz +  1];
	}
	//**************************************************************************
	// 3. inverse Hadamard Transform along the third dimension
#if __config_specific_op__
	for (unsigned i = 0; i < bm3d_param.m_max_grp_size; ++i) // dummy variable
	{
		s_img_pixel_arr_inv_hadtf[l_threadIdx_xyz] += s_img_pixel_arr_noisy_hadtf[l_threadIdx_xy + i * blockDim_xy] * s_had_tf_coeff[threadIdx.z + i * blockDim_xy];
	} // dummy variable i
#else // __config_specific_op__
	for (unsigned i = 0; i < bm3d_param.m_max_grp_size; ++i)
	{
		s_img_pixel_arr_inv_hadtf[l_threadIdx_xyz] += s_img_pixel_arr_noisy_hadtf[l_threadIdx_xy + i * blockDim_xy] * 
			tex1Dfetch
				(
					l_tex_ref_had_tf_coeff,
					(num_of_matches - 1) * l_tex_ref_had_tf_coeff_offset[0] + 
					                  i  * l_tex_ref_had_tf_coeff_offset[1] + 
					        threadIdx.z
				);
	}
#endif // __config_specific_op__
#if __config_specific_op__
	
#else // __config_specific_op__
	__syncthreads(); // end of the inverse Hadamard Transform
#endif // __config_specific_op__
	//**************************************************************************
	// 4. inverse Discrete Cosine Transform
	for (unsigned i = 0; i < bm3d_param.m_patch_size; ++i) // dummy variable
	{
		for (unsigned j = 0; j < bm3d_param.m_patch_size; ++j) // dummy variable
		{
			s_img_pixel_arr_inv_dct[l_threadIdx_xyz] += s_img_pixel_arr_inv_hadtf[j + i * blockDim.x + threadIdx.z * blockDim_xy] * 
				tex1Dfetch
					(
						l_tex_ref_dct_2d_coeff,
						          i * l_tex_ref_dct_2d_coeff_offset[0] + 
						          j * l_tex_ref_dct_2d_coeff_offset[1] + 
						threadIdx.y * l_tex_ref_dct_2d_coeff_offset[2] + 
						threadIdx.x
					);
		} // dummy variable j
	} // dummy variable i
	//**************************************************************************
	// 5. weight aggregation
	const image_t::pixel_t kaiser_win_coeff = threadIdx.z < num_of_matches ? tex1Dfetch(l_tex_ref_kaiser_win_coeff, threadIdx.x + threadIdx.y * blockDim.x) : 0;
	//**************************************************************************
	if (s_wiener_weight_coeff[0] == 0.0)
	{
		s_wiener_weight_coeff[0] = 1.0;
	}
	//**************************************************************************
	atomicAdd
		(
			&  numerator
			[
										 blockIdx.z  * height * width + 
				(this_patch_g_coord_y + threadIdx.y)          * width + 
				(this_patch_g_coord_x + threadIdx.x)
			], 1.0 / s_wiener_weight_coeff[0] * kaiser_win_coeff * s_img_pixel_arr_inv_dct[l_threadIdx_xyz]
		);
	atomicAdd
		(
			&denominator
			[
				                         blockIdx.z  * height * width + 
				(this_patch_g_coord_y + threadIdx.y)          * width + 
				(this_patch_g_coord_x + threadIdx.x)
			], 1.0 / s_wiener_weight_coeff[0] * kaiser_win_coeff
		);
}

// @brief apply the following calculations, in sequence
// 1. forward Hadamard Transform along the third dimension
// 2. Hard Thresholding
// 3. inverse Hadamard Transform along the third dimension
// 4. inverse Discrete Cosine Transform
// 5. weight aggregation
// @param (output) img_pixel_arr_basic: basic estimate of the noisy image
// (shape: nchnls * height * width)
// @param ( input) img_pixel_arr_dct_noisy: DCT of noisy image
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) noisy_dist_vec: sorted distance vector (based on noisy image)
// (shape: num_y_refs * num_x_refs * max_grp_size)
// @param ( input) noisy_num_of_matches_vec: number of matches for each reference patch (for ONLY the 1st chnl of noisy image)
// (shape: num_y_refs * num_x_refs)
// @param ( input) img_pixel_arr_raw_noisy: noisy image
// (shape: nchnls * height * width)
// @param ( input) sigma_table: standard deviation of each channel
// (shape: nchnls)
// @param ( input) threshold: threshold value -> lambda
// @param ( input) bm3d_param: BM3D parameters
void bm3d_hadtf_hard_inv_hadtf_inv_dct
(
          tensor_t < image_t::pixel_t, 3 > & img_pixel_arr_raw_basic,
	const tensor_t < image_t::pixel_t, 5 > & img_pixel_arr_dct_noisy,
	const tensor_t <           dist_t, 3 > & noisy_dist_vec,
	const tensor_t <         unsigned, 2 > & noisy_num_of_matches_vec,
	const tensor_t < image_t::pixel_t, 3 > & img_pixel_arr_raw_noisy,
	const tensor_t < image_t::pixel_t, 1 > & sigma_table,
	const image_t::pixel_t & threshold, const bm3d_param_t & bm3d_param
)
{
	// pre-compute the Hadamard Transform coefficients and bind them to the texture memory
	boost::multi_array < image_t::pixel_t, 3 > had_tf_coeff_ndarr
		(
			boost::extents
			[bm3d_param.m_max_grp_size]
			[bm3d_param.m_max_grp_size]
			[bm3d_param.m_max_grp_size]
		);
#pragma omp parallel for
	for (unsigned i = 1; i <= bm3d_param.m_max_grp_size; ++i)
	{
		unsigned floor_i = floor_pow_2(i); image_t::pixel_t sqrt_size = sqrt(floor_i);
		
		// initialize coefficients in the first row
		for (unsigned j = 0; j < floor_i; ++j) had_tf_coeff_ndarr[i - 1][0][j] = 1.0 / sqrt_size;
		// initialize coefficients in other rows
		for (unsigned j = 1, pattern = floor_i; j < floor_i;)
		{
			// "pattern" means the period and "start" denotes the starting element
			for (unsigned k = 0, half = pattern / 2, start = 0; k < floor_i / pattern; ++j, ++k, start += pattern)
			{
				for (unsigned l = start       ; l < start +    half; ++l) had_tf_coeff_ndarr[i - 1][j][l] =   1.0 / sqrt_size;
				for (unsigned l = start + half; l < start + pattern; ++l) had_tf_coeff_ndarr[i - 1][j][l] = - 1.0 / sqrt_size;
			}
			if (pattern > 2)
			{
				sqrt_size /= M_SQRT2; pattern /= 2;
			} // k
		} // j
	} // i
	l_had_tf_coeff_ts = had_tf_coeff_ndarr;
	CUDA_ERR_CHECK(cudaBindTexture(nullptr, l_tex_ref_had_tf_coeff, l_had_tf_coeff_ts.m_data_arr, l_had_tf_coeff_ts.m_size * sizeof(image_t::pixel_t)));
	//**************************************************************************
	assert(bm3d_param.m_patch_size == 4 || bm3d_param.m_patch_size == 8);
	// pre-compute the Kaiser Window coefficients
	boost::multi_array < image_t::pixel_t, 2 > kaiser_win_coeff_ndarr (boost::extents[bm3d_param.m_patch_size][bm3d_param.m_patch_size]);
	
	if (bm3d_param.m_patch_size == 4)
	{
		kaiser_win_coeff_ndarr[0][0] = 0.19245769; kaiser_win_coeff_ndarr[0][1] = 0.40549041; kaiser_win_coeff_ndarr[0][2] = 0.40549041; kaiser_win_coeff_ndarr[0][3] = 0.19245796;
		kaiser_win_coeff_ndarr[1][0] = 0.40549041; kaiser_win_coeff_ndarr[1][1] = 0.85433049; kaiser_win_coeff_ndarr[1][2] = 0.85433049; kaiser_win_coeff_ndarr[1][3] = 0.40549041;
		kaiser_win_coeff_ndarr[2][0] = 0.40549041; kaiser_win_coeff_ndarr[2][1] = 0.85433049; kaiser_win_coeff_ndarr[2][2] = 0.85433049; kaiser_win_coeff_ndarr[2][3] = 0.40549041;
		kaiser_win_coeff_ndarr[3][0] = 0.19245769; kaiser_win_coeff_ndarr[3][1] = 0.40549041; kaiser_win_coeff_ndarr[3][2] = 0.40549041; kaiser_win_coeff_ndarr[3][3] = 0.19245769;
	}
	if (bm3d_param.m_patch_size == 8)
	{
		kaiser_win_coeff_ndarr[0][0] = 0.19245769; kaiser_win_coeff_ndarr[0][1] = 0.29888631; kaiser_win_coeff_ndarr[0][2] = 0.38465216; kaiser_win_coeff_ndarr[0][3] = 0.43247046; kaiser_win_coeff_ndarr[0][4] = 0.43247046; kaiser_win_coeff_ndarr[0][5] = 0.38465216; kaiser_win_coeff_ndarr[0][6] = 0.29888631; kaiser_win_coeff_ndarr[0][7] = 0.19245769;
		kaiser_win_coeff_ndarr[1][0] = 0.29888631; kaiser_win_coeff_ndarr[1][1] = 0.46416969; kaiser_win_coeff_ndarr[1][2] = 0.59736384; kaiser_win_coeff_ndarr[1][3] = 0.67162554; kaiser_win_coeff_ndarr[1][4] = 0.67162554; kaiser_win_coeff_ndarr[1][5] = 0.59736384; kaiser_win_coeff_ndarr[1][6] = 0.46416969; kaiser_win_coeff_ndarr[1][7] = 0.29888631;
		kaiser_win_coeff_ndarr[2][0] = 0.38465216; kaiser_win_coeff_ndarr[2][1] = 0.59736384; kaiser_win_coeff_ndarr[2][2] = 0.76877824; kaiser_win_coeff_ndarr[2][3] = 0.86434944; kaiser_win_coeff_ndarr[2][4] = 0.86434944; kaiser_win_coeff_ndarr[2][5] = 0.76877824; kaiser_win_coeff_ndarr[2][6] = 0.59736384; kaiser_win_coeff_ndarr[2][7] = 0.38465216;
		kaiser_win_coeff_ndarr[3][0] = 0.43247046; kaiser_win_coeff_ndarr[3][1] = 0.67162554; kaiser_win_coeff_ndarr[3][2] = 0.86434944; kaiser_win_coeff_ndarr[3][3] = 0.97180164; kaiser_win_coeff_ndarr[3][4] = 0.97180164; kaiser_win_coeff_ndarr[3][5] = 0.86434944; kaiser_win_coeff_ndarr[3][6] = 0.67162554; kaiser_win_coeff_ndarr[3][7] = 0.43247046;
		kaiser_win_coeff_ndarr[4][0] = 0.43247046; kaiser_win_coeff_ndarr[4][1] = 0.67162554; kaiser_win_coeff_ndarr[4][2] = 0.86434944; kaiser_win_coeff_ndarr[4][3] = 0.97180164; kaiser_win_coeff_ndarr[4][4] = 0.97180164; kaiser_win_coeff_ndarr[4][5] = 0.86434944; kaiser_win_coeff_ndarr[4][6] = 0.67162554; kaiser_win_coeff_ndarr[4][7] = 0.43247046;
		kaiser_win_coeff_ndarr[5][0] = 0.38465216; kaiser_win_coeff_ndarr[5][1] = 0.59736384; kaiser_win_coeff_ndarr[5][2] = 0.76877824; kaiser_win_coeff_ndarr[5][3] = 0.86434944; kaiser_win_coeff_ndarr[5][4] = 0.86434944; kaiser_win_coeff_ndarr[5][5] = 0.76877824; kaiser_win_coeff_ndarr[5][6] = 0.59736384; kaiser_win_coeff_ndarr[5][7] = 0.38465216;
		kaiser_win_coeff_ndarr[6][0] = 0.29888631; kaiser_win_coeff_ndarr[6][1] = 0.46416969; kaiser_win_coeff_ndarr[6][2] = 0.59736384; kaiser_win_coeff_ndarr[6][3] = 0.67162554; kaiser_win_coeff_ndarr[6][4] = 0.67162554; kaiser_win_coeff_ndarr[6][5] = 0.59736384; kaiser_win_coeff_ndarr[6][6] = 0.46416969; kaiser_win_coeff_ndarr[6][7] = 0.29888631;
		kaiser_win_coeff_ndarr[7][0] = 0.19245769; kaiser_win_coeff_ndarr[7][1] = 0.29888631; kaiser_win_coeff_ndarr[7][2] = 0.38465216; kaiser_win_coeff_ndarr[7][3] = 0.43247046; kaiser_win_coeff_ndarr[7][4] = 0.43247046; kaiser_win_coeff_ndarr[7][5] = 0.38465216; kaiser_win_coeff_ndarr[7][6] = 0.29888631; kaiser_win_coeff_ndarr[7][7] = 0.19245769;
	}
	l_kaiser_win_coeff_ts = kaiser_win_coeff_ndarr;
	CUDA_ERR_CHECK(cudaBindTexture(nullptr, l_tex_ref_kaiser_win_coeff, l_kaiser_win_coeff_ts.m_data_arr, l_kaiser_win_coeff_ts.m_size * sizeof(image_t::pixel_t)));
	//**************************************************************************
	// numerator and denominator are used for weight aggregation process
	tensor_t < image_t::pixel_t, 3 >   numerator (img_pixel_arr_raw_noisy.m_extents);
	tensor_t < image_t::pixel_t, 3 > denominator (img_pixel_arr_raw_noisy.m_extents);
	//**************************************************************************
#if __config_specific_op__
	assert(bm3d_param.m_max_grp_size == 16 && bm3d_param.m_patch_size == 4);
	//**************************************************************************
	cuda_hadtf_hard_inv_hadtf_inv_dct
		<<<
			dim3
				(
					// # of reference patches along the x axis
					(img_pixel_arr_raw_noisy.m_extents[2] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of reference patches along the y axis
					(img_pixel_arr_raw_noisy.m_extents[1] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of channels
					 img_pixel_arr_raw_noisy.m_extents[0]
				)
			,
			dim3
				(
					bm3d_param.m_patch_size, 
					bm3d_param.m_patch_size, 
					bm3d_param.m_max_grp_size
				)
			,
			(bm3d_param.m_patch_size * 
			 bm3d_param.m_patch_size * 
			 bm3d_param.m_max_grp_size * 6) * 
			sizeof(image_t::pixel_t)
		>>>
		(
			numerator.m_data_arr, denominator.m_data_arr,
			img_pixel_arr_dct_noisy .m_data_arr,
			noisy_dist_vec          .m_data_arr,
			noisy_num_of_matches_vec.m_data_arr,
			sigma_table             .m_data_arr, 
			img_pixel_arr_raw_noisy.m_extents[2], img_pixel_arr_raw_noisy.m_extents[1],
			threshold, bm3d_param
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
#else // __config_specific_op__
	cuda_hadtf_hard_inv_hadtf_inv_dct
		<<<
			dim3
				(
					// # of reference patches along the x axis
					(img_pixel_arr_raw_noisy.m_extents[2] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of reference patches along the y axis
					(img_pixel_arr_raw_noisy.m_extents[1] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of channels
					 img_pixel_arr_raw_noisy.m_extents[0]
				)
			,
			dim3
				(
					bm3d_param.m_patch_size, 
					bm3d_param.m_patch_size, 
					bm3d_param.m_max_grp_size
				)
			,
			(bm3d_param.m_patch_size * 
			 bm3d_param.m_patch_size * 
			 bm3d_param.m_max_grp_size * 5) * 
			sizeof(image_t::pixel_t)
		>>>
		(
			numerator.m_data_arr, denominator.m_data_arr,
			img_pixel_arr_dct_noisy .m_data_arr,
			noisy_dist_vec          .m_data_arr,
			noisy_num_of_matches_vec.m_data_arr,
			sigma_table             .m_data_arr, 
			img_pixel_arr_raw_noisy.m_extents[2], img_pixel_arr_raw_noisy.m_extents[1],
			threshold, bm3d_param
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
#endif // __config_specific_op__
	//**************************************************************************
	// numerator / denominator
	cuda_numerator_div_denominator <<< img_pixel_arr_raw_noisy.m_size / 32 + 1, 32 >>>
		(
			img_pixel_arr_raw_basic.m_data_arr,
			numerator.m_data_arr, denominator.m_data_arr,
			img_pixel_arr_raw_noisy.m_data_arr,
			img_pixel_arr_raw_noisy.m_size
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
	//**************************************************************************
	CUDA_ERR_CHECK(cudaUnbindTexture(l_tex_ref_dct_2d_coeff)); l_dct_2d_coeff_ts.~tensor_t();
	CUDA_ERR_CHECK(cudaUnbindTexture(l_tex_ref_had_tf_coeff)); l_had_tf_coeff_ts.~tensor_t();
	CUDA_ERR_CHECK(cudaUnbindTexture(l_tex_ref_kaiser_win_coeff)); l_kaiser_win_coeff_ts.~tensor_t();
}

// @brief apply the following calculations, in sequence
// 1. forward Hadamard Transform along the third dimension
// 2. Wiener Filtering
// 3. inverse Hadamard Transform along the third dimension
// 4. inverse Discrete Cosine Transform
// 5. weight aggregation
// @param (output) img_pixel_arr_raw_final: final estimate of the noisy image
// (shape: nchnls * height * width)
// @param ( input) img_pixel_arr_dct_basic: DCT of basic estimate
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) img_pixel_arr_dct_noisy: DCT of noisy image
// (shape: nchnls * height(approx.) * width(approx.) * patch_size * patch_size)
// @param ( input) basic_dist_vec: sorted distance vector (based on basic estimate)
// (shape: num_y_refs * num_x_refs * max_grp_size)
// @param ( input) basic_num_of_matches_vec: number of matches for each reference patch (for ONLY the 1st chnl of basic estimate)
// (shape: num_y_refs * num_x_refs)
// @param ( input) img_pixel_arr_raw_basic: basic estimate of the noisy image
// (shape: nchnls * height * width)
// @param ( input) sigma_table: standard deviation of each channel
// (shape: nchnls)
// @param ( input) bm3d_param: BM3D parameters
void bm3d_hadtf_wiener_inv_hadtf_inv_dct
(
	      tensor_t < image_t::pixel_t, 3 > & img_pixel_arr_raw_final,
    const tensor_t < image_t::pixel_t, 5 > & img_pixel_arr_dct_basic,
	const tensor_t < image_t::pixel_t, 5 > & img_pixel_arr_dct_noisy,
	const tensor_t <           dist_t, 3 > & basic_dist_vec,
	const tensor_t <         unsigned, 2 > & basic_num_of_matches_vec,
	const tensor_t < image_t::pixel_t, 3 > & img_pixel_arr_raw_basic,
	const tensor_t < image_t::pixel_t, 1 > & sigma_table,
	const bm3d_param_t & bm3d_param
)
{
	// pre-compute the Hadamard Transform coefficients and bind them to the texture memory
	boost::multi_array < image_t::pixel_t, 3 > had_tf_coeff_ndarr
		(
			boost::extents
			[bm3d_param.m_max_grp_size]
			[bm3d_param.m_max_grp_size]
			[bm3d_param.m_max_grp_size]
		);
#pragma omp parallel for
	for (unsigned i = 1; i <= bm3d_param.m_max_grp_size; ++i)
	{
		unsigned floor_i = floor_pow_2(i); image_t::pixel_t sqrt_size = sqrt(floor_i);
		
		// initialize coefficients in the first row
		for (unsigned j = 0; j < floor_i; ++j) had_tf_coeff_ndarr[i - 1][0][j] = 1.0 / sqrt_size;
		// initialize coefficients in other rows
		for (unsigned j = 1, pattern = floor_i; j < floor_i;)
		{
			// "pattern" means the period and "start" denotes the starting element
			for (unsigned k = 0, half = pattern / 2, start = 0; k < floor_i / pattern; ++j, ++k, start += pattern)
			{
				for (unsigned l = start       ; l < start +    half; ++l) had_tf_coeff_ndarr[i - 1][j][l] =   1.0 / sqrt_size;
				for (unsigned l = start + half; l < start + pattern; ++l) had_tf_coeff_ndarr[i - 1][j][l] = - 1.0 / sqrt_size;
			}
			if (pattern > 2)
			{
				sqrt_size /= M_SQRT2; pattern /= 2;
			} // k
		} // j
	} // i
	l_had_tf_coeff_ts = had_tf_coeff_ndarr;
	CUDA_ERR_CHECK(cudaBindTexture(nullptr, l_tex_ref_had_tf_coeff, l_had_tf_coeff_ts.m_data_arr, l_had_tf_coeff_ts.m_size * sizeof(image_t::pixel_t)));
	//**************************************************************************
	assert(bm3d_param.m_patch_size == 4 || bm3d_param.m_patch_size == 8);
	// pre-compute the Kaiser Window coefficients
	boost::multi_array < image_t::pixel_t, 2 > kaiser_win_coeff_ndarr (boost::extents[bm3d_param.m_patch_size][bm3d_param.m_patch_size]);
	
	if (bm3d_param.m_patch_size == 4)
	{
		kaiser_win_coeff_ndarr[0][0] = 0.19245769; kaiser_win_coeff_ndarr[0][1] = 0.40549041; kaiser_win_coeff_ndarr[0][2] = 0.40549041; kaiser_win_coeff_ndarr[0][3] = 0.19245796;
		kaiser_win_coeff_ndarr[1][0] = 0.40549041; kaiser_win_coeff_ndarr[1][1] = 0.85433049; kaiser_win_coeff_ndarr[1][2] = 0.85433049; kaiser_win_coeff_ndarr[1][3] = 0.40549041;
		kaiser_win_coeff_ndarr[2][0] = 0.40549041; kaiser_win_coeff_ndarr[2][1] = 0.85433049; kaiser_win_coeff_ndarr[2][2] = 0.85433049; kaiser_win_coeff_ndarr[2][3] = 0.40549041;
		kaiser_win_coeff_ndarr[3][0] = 0.19245769; kaiser_win_coeff_ndarr[3][1] = 0.40549041; kaiser_win_coeff_ndarr[3][2] = 0.40549041; kaiser_win_coeff_ndarr[3][3] = 0.19245769;
	}
	if (bm3d_param.m_patch_size == 8)
	{
		kaiser_win_coeff_ndarr[0][0] = 0.19245769; kaiser_win_coeff_ndarr[0][1] = 0.29888631; kaiser_win_coeff_ndarr[0][2] = 0.38465216; kaiser_win_coeff_ndarr[0][3] = 0.43247046; kaiser_win_coeff_ndarr[0][4] = 0.43247046; kaiser_win_coeff_ndarr[0][5] = 0.38465216; kaiser_win_coeff_ndarr[0][6] = 0.29888631; kaiser_win_coeff_ndarr[0][7] = 0.19245769;
		kaiser_win_coeff_ndarr[1][0] = 0.29888631; kaiser_win_coeff_ndarr[1][1] = 0.46416969; kaiser_win_coeff_ndarr[1][2] = 0.59736384; kaiser_win_coeff_ndarr[1][3] = 0.67162554; kaiser_win_coeff_ndarr[1][4] = 0.67162554; kaiser_win_coeff_ndarr[1][5] = 0.59736384; kaiser_win_coeff_ndarr[1][6] = 0.46416969; kaiser_win_coeff_ndarr[1][7] = 0.29888631;
		kaiser_win_coeff_ndarr[2][0] = 0.38465216; kaiser_win_coeff_ndarr[2][1] = 0.59736384; kaiser_win_coeff_ndarr[2][2] = 0.76877824; kaiser_win_coeff_ndarr[2][3] = 0.86434944; kaiser_win_coeff_ndarr[2][4] = 0.86434944; kaiser_win_coeff_ndarr[2][5] = 0.76877824; kaiser_win_coeff_ndarr[2][6] = 0.59736384; kaiser_win_coeff_ndarr[2][7] = 0.38465216;
		kaiser_win_coeff_ndarr[3][0] = 0.43247046; kaiser_win_coeff_ndarr[3][1] = 0.67162554; kaiser_win_coeff_ndarr[3][2] = 0.86434944; kaiser_win_coeff_ndarr[3][3] = 0.97180164; kaiser_win_coeff_ndarr[3][4] = 0.97180164; kaiser_win_coeff_ndarr[3][5] = 0.86434944; kaiser_win_coeff_ndarr[3][6] = 0.67162554; kaiser_win_coeff_ndarr[3][7] = 0.43247046;
		kaiser_win_coeff_ndarr[4][0] = 0.43247046; kaiser_win_coeff_ndarr[4][1] = 0.67162554; kaiser_win_coeff_ndarr[4][2] = 0.86434944; kaiser_win_coeff_ndarr[4][3] = 0.97180164; kaiser_win_coeff_ndarr[4][4] = 0.97180164; kaiser_win_coeff_ndarr[4][5] = 0.86434944; kaiser_win_coeff_ndarr[4][6] = 0.67162554; kaiser_win_coeff_ndarr[4][7] = 0.43247046;
		kaiser_win_coeff_ndarr[5][0] = 0.38465216; kaiser_win_coeff_ndarr[5][1] = 0.59736384; kaiser_win_coeff_ndarr[5][2] = 0.76877824; kaiser_win_coeff_ndarr[5][3] = 0.86434944; kaiser_win_coeff_ndarr[5][4] = 0.86434944; kaiser_win_coeff_ndarr[5][5] = 0.76877824; kaiser_win_coeff_ndarr[5][6] = 0.59736384; kaiser_win_coeff_ndarr[5][7] = 0.38465216;
		kaiser_win_coeff_ndarr[6][0] = 0.29888631; kaiser_win_coeff_ndarr[6][1] = 0.46416969; kaiser_win_coeff_ndarr[6][2] = 0.59736384; kaiser_win_coeff_ndarr[6][3] = 0.67162554; kaiser_win_coeff_ndarr[6][4] = 0.67162554; kaiser_win_coeff_ndarr[6][5] = 0.59736384; kaiser_win_coeff_ndarr[6][6] = 0.46416969; kaiser_win_coeff_ndarr[6][7] = 0.29888631;
		kaiser_win_coeff_ndarr[7][0] = 0.19245769; kaiser_win_coeff_ndarr[7][1] = 0.29888631; kaiser_win_coeff_ndarr[7][2] = 0.38465216; kaiser_win_coeff_ndarr[7][3] = 0.43247046; kaiser_win_coeff_ndarr[7][4] = 0.43247046; kaiser_win_coeff_ndarr[7][5] = 0.38465216; kaiser_win_coeff_ndarr[7][6] = 0.29888631; kaiser_win_coeff_ndarr[7][7] = 0.19245769;
	}
	l_kaiser_win_coeff_ts = kaiser_win_coeff_ndarr;
	CUDA_ERR_CHECK(cudaBindTexture(nullptr, l_tex_ref_kaiser_win_coeff, l_kaiser_win_coeff_ts.m_data_arr, l_kaiser_win_coeff_ts.m_size * sizeof(image_t::pixel_t)));
	//**************************************************************************
	// numerator and denominator are used for weight aggregation process
	tensor_t < image_t::pixel_t, 3 >   numerator (img_pixel_arr_raw_basic.m_extents);
	tensor_t < image_t::pixel_t, 3 > denominator (img_pixel_arr_raw_basic.m_extents);
	//**************************************************************************
#if __config_specific_op__
	assert(bm3d_param.m_max_grp_size == 16 && bm3d_param.m_patch_size == 4);
	//**************************************************************************
	cuda_hadtf_wiener_inv_hadtf_inv_dct
		<<<
			dim3
				(
					// # of reference patches along the x axis
					(img_pixel_arr_raw_basic.m_extents[2] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of reference patches along the y axis
					(img_pixel_arr_raw_basic.m_extents[1] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of channels
					 img_pixel_arr_raw_basic.m_extents[0]
				)
			,
			dim3
				(
					bm3d_param.m_patch_size,
					bm3d_param.m_patch_size,
					bm3d_param.m_max_grp_size
				)
			,
			(bm3d_param.m_patch_size *
			 bm3d_param.m_patch_size * 
			 bm3d_param.m_max_grp_size * 8) *
			sizeof(image_t::pixel_t)
		>>>
		(
			numerator.m_data_arr, denominator.m_data_arr,
			img_pixel_arr_dct_basic .m_data_arr,
			img_pixel_arr_dct_noisy .m_data_arr,
			basic_dist_vec          .m_data_arr,
			basic_num_of_matches_vec.m_data_arr,
			sigma_table             .m_data_arr, 
			img_pixel_arr_raw_basic.m_extents[2], img_pixel_arr_raw_basic.m_extents[1],
			bm3d_param
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
#else // __config_specific_op__
	cuda_hadtf_wiener_inv_hadtf_inv_dct
		<<<
			dim3
				(
					// # of reference patches along the x axis
					(img_pixel_arr_raw_basic.m_extents[2] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of reference patches along the y axis
					(img_pixel_arr_raw_basic.m_extents[1] - bm3d_param.m_patch_size + 1) / bm3d_param.m_reference_step,
					// # of channels
					 img_pixel_arr_raw_basic.m_extents[0]
				)
			,
			dim3
				(
					bm3d_param.m_patch_size,
					bm3d_param.m_patch_size,
					bm3d_param.m_max_grp_size
				)
			,
			(bm3d_param.m_patch_size *
			 bm3d_param.m_patch_size * 
			 bm3d_param.m_max_grp_size * 7) *
			sizeof(image_t::pixel_t)
		>>>
		(
			numerator.m_data_arr, denominator.m_data_arr,
			img_pixel_arr_dct_basic .m_data_arr,
			img_pixel_arr_dct_noisy .m_data_arr,
			basic_dist_vec          .m_data_arr,
			basic_num_of_matches_vec.m_data_arr,
			sigma_table             .m_data_arr, 
			img_pixel_arr_raw_basic.m_extents[2], img_pixel_arr_raw_basic.m_extents[1],
			bm3d_param
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
#endif // __config_specific_op__
	//**************************************************************************
	// numerator / denominator
	cuda_numerator_div_denominator
		<<<	img_pixel_arr_raw_basic.m_size / 32 + 1, 32 >>>
		(
			img_pixel_arr_raw_final.m_data_arr,
			numerator.m_data_arr, denominator.m_data_arr,
			img_pixel_arr_raw_basic.m_data_arr,
			img_pixel_arr_raw_basic.m_size
		);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
	//**************************************************************************
	CUDA_ERR_CHECK(cudaUnbindTexture(l_tex_ref_dct_2d_coeff)); l_dct_2d_coeff_ts.~tensor_t();
	CUDA_ERR_CHECK(cudaUnbindTexture(l_tex_ref_had_tf_coeff)); l_had_tf_coeff_ts.~tensor_t();
	CUDA_ERR_CHECK(cudaUnbindTexture(l_tex_ref_kaiser_win_coeff)); l_kaiser_win_coeff_ts.~tensor_t();
}
