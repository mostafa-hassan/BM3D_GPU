#ifndef BM3D_H
#define BM3D_H

#include "tensor.h"

/*******************************************************************************/
// Auxiliary Data Structures
/*******************************************************************************/

struct comp_result_t
{
	bool lt; // less than
	bool et; // equal to
};

// Pack all the parameters together to form a single structure
struct bm3d_param_t
{
	unsigned			m_hf_window_size;
	unsigned			m_patch_size;
	unsigned			m_max_grp_size;
	unsigned			m_reference_step;
	image_t::pixel_t	m_tau_match;
	bm3d_param_t
		(
			const unsigned & hf_window_size, 
			const unsigned & patch_size,
			const unsigned & max_grp_size,
			const unsigned & reference_step,
			const image_t::pixel_t & tau_match
		) : m_hf_window_size(hf_window_size),
			m_patch_size    (patch_size    ),
			m_max_grp_size  (max_grp_size  ),
			m_reference_step(reference_step),
			m_tau_match     (tau_match     )
	{}
};

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

#endif // BM3D_H
