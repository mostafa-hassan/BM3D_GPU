#include "../inc/image.h"
#include "../inc/bm3d.h"

#include <string>
#include <utility>
#include <cstring>
#include <iostream>
#include <fstream>
using namespace std;

#include <chrono>
using namespace std::chrono;

static const image_t::pixel_t sigma = 20.0;
static const image_t::color_transform_t color_tf_mode_forward = image_t::color_transform_t::RGB_2_OPP;
static const image_t::color_transform_t color_tf_mode_inverse = image_t::color_transform_t::OPP_2_RGB;
static const image_t::pixel_t threshold_ht = 2.7;
// @param1 hf_window_size: half of the search window size
// @param2 patch_size: size of the patch
// @param3 max_grp_size: maximum number of similar patches allowed
// @param4 reference_step: step between reference patches
// @param5 tau_match: threshold value in which two patches are considered similar
const unsigned hf_window_size_ht = 24;
const unsigned hf_window_size_wn = 19;
const unsigned patch_size_ht = 4;
const unsigned patch_size_wn = 4;
const unsigned max_grp_size_ht = 16;
const unsigned max_grp_size_wn = 16;
const unsigned reference_step_ht = 1;
const unsigned reference_step_wn = 1;
const image_t::pixel_t tau_match_ht = 2500;
const image_t::pixel_t tau_match_wn =  400;
const image_t::pixel_t scaled_tau_match_ht = tau_match_ht * patch_size_ht * patch_size_ht;
const image_t::pixel_t scaled_tau_match_wn = tau_match_wn * patch_size_wn * patch_size_wn;

static const bm3d_param_t bm3d_param_ht (hf_window_size_ht, patch_size_ht, max_grp_size_ht, reference_step_ht, tau_match_ht);
static const bm3d_param_t bm3d_param_wn (hf_window_size_wn, patch_size_wn, max_grp_size_wn, reference_step_wn, tau_match_wn);

std::vector < std::pair < std::string, float > > time_vec_gpu;



const unsigned tileWidth = 1000;
const unsigned tileHeight = 1000;

int main(int argc, char *argv[])
{
		if( argc < 5 )
		{
			std::cout << "Please enter image file name, format, input directory and out directory" << std::endl;
			exit (-1);
		}

		const std::string img_fname (argv[1]);
		const std::string img_format (argv[2]);
		const std::string in_dir (argv[3]);
		const std::string out_dir (argv[4]);

		std::cout << "Running BM3D using GPU for file: " << img_fname << std::endl;
		std::cout << "nHard: " << hf_window_size_ht << " nWien: " << hf_window_size_wn << " kHard: " << patch_size_ht << " kWien: " << patch_size_wn << " NHard: " << max_grp_size_ht <<  " NWien: " << max_grp_size_wn << " pHard: " << reference_step_ht << " pWien: " << reference_step_wn << std::endl;
		//**************************************************************************
		// Image Setup
		//**************************************************************************
		std::cout << "Setting up the data structure ..." << std::endl;

		image_t raw_img, noisy_img;
		// read the pixel values from the PNG file, after that, add a Gaussian noise to noisy_img to make it noisy
		raw_img.load(in_dir, out_dir, img_fname, img_format);
		noisy_img.load(in_dir, out_dir, img_fname, img_format);
		noisy_img.add_noise(sigma);
		// noisy_img.save("./png/" + img_fname + "/" + img_fname + "_noisy.png");
		std::cout << "\tnchnls: " << raw_img.m_img_pixel_arr.shape()[0] << std::endl;
		std::cout << "\theight: " << raw_img.m_img_pixel_arr.shape()[1] << std::endl;
		std::cout << "\twidth : " << raw_img.m_img_pixel_arr.shape()[2] << std::endl;
		unsigned nchnls = raw_img.m_img_pixel_arr.shape()[0];
		unsigned height = raw_img.m_img_pixel_arr.shape()[1];
		unsigned width = raw_img.m_img_pixel_arr.shape()[2];

		// perform color transform if it is an RGB image
		if (raw_img.m_img_pixel_arr.shape()[0] == 3)
		{
			  raw_img.color_transform(color_tf_mode_forward);
			noisy_img.color_transform(color_tf_mode_forward);
		}
		tensor_t < image_t::pixel_t, 1 > sigma_table (raw_img.get_sigma_table(sigma, color_tf_mode_forward));
		//**************************************************************************
		// CUDA Function Call
		//**************************************************************************
		cudaDeviceProp deviceProp; // get the device properties
		CUDA_ERR_CHECK(cudaSetDevice(0));
		CUDA_ERR_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
		//**************************************************************************
		time_vec_gpu.push_back(std::make_pair < std::string, float > ("bm3d_precompute_dct_ht", 0));
		time_vec_gpu.push_back(std::make_pair < std::string, float > ("bm3d_compute_and_sort_dct_dist_vec_1st_chnl_ht", 0));
		time_vec_gpu.push_back(std::make_pair < std::string, float > ("bm3d_hadtf_hard_inv_hadtf_inv_dct", 0));
		time_vec_gpu.push_back(std::make_pair < std::string, float > ("bm3d_compute_and_sort_raw_dist_vec_1st_chnl_wn", 0));
		time_vec_gpu.push_back(std::make_pair < std::string, float > ("bm3d_filter_matches_compute_dct_wn", 0));
		time_vec_gpu.push_back(std::make_pair < std::string, float > ("bm3d_hadtf_wiener_inv_hadtf_inv_dct", 0));
		image_t basic_img, final_img;
		final_img.setDimensions(nchnls, height, width);
		basic_img.setDimensions(nchnls, height, width);
		unsigned maxTileX = width/tileWidth;
		unsigned maxTileY = height/tileHeight;
		bool integLastXTile = false;
		bool integLastYTile = false;
		//boundry tiles are soo small (less than 200 size) ? integrate them with the ones before
		if (width - maxTileX*tileWidth < 200 )
		{
			maxTileX = maxTileX -1;
			integLastXTile = true;
		}
		if (height - maxTileY*tileHeight < 200 )
		{
			maxTileY = maxTileY -1;
			integLastYTile = true;
		}
		std::cout << "X maxtiles = " << maxTileX << std::endl;
		std::cout << "Y maxtiles = " << maxTileY << std::endl;
		for (unsigned tileY = 0; tileY <=  maxTileY; tileY++)
		{
			for (unsigned tileX = 0; tileX <=  maxTileX; tileX++)
			{

				image_t noisy_subimg;
				unsigned subMinX = tileX*tileWidth;
				unsigned subMaxX = std::min ((tileX+1)*tileWidth - 1, width - 1);
				if ((tileX == maxTileX) && (integLastXTile))
					subMaxX = width - 1;

				unsigned subMinY = tileY*tileHeight;
				unsigned subMaxY = std::min ((tileY+1)*tileHeight - 1, height - 1);
				if ((tileY == maxTileY) && (integLastYTile))
					subMaxY = height - 1;

				if   ((subMinX <= subMaxX) && (subMinY <= subMaxY))
				{

					noisy_subimg.load_subimage(noisy_img.m_img_pixel_arr, subMinX, subMaxX, subMinY, subMaxY);

					tensor_t < image_t::pixel_t, 3 > img_pixel_arr_raw_noisy (noisy_subimg.m_img_pixel_arr);
					const unsigned img_pixel_arr_dct_noisy_extents[5] =
						{
							img_pixel_arr_raw_noisy.m_extents[0],
							img_pixel_arr_raw_noisy.m_extents[1] - bm3d_param_ht.m_patch_size + 2,
							img_pixel_arr_raw_noisy.m_extents[2] - bm3d_param_ht.m_patch_size + 2,
							bm3d_param_ht.m_patch_size,
							bm3d_param_ht.m_patch_size
						};
					tensor_t < image_t::pixel_t, 5 > img_pixel_arr_dct_noisy (img_pixel_arr_dct_noisy_extents);

					{

						auto func_start = high_resolution_clock::now();
						bm3d_precompute_dct_ht
							(
								img_pixel_arr_dct_noisy,
								img_pixel_arr_raw_noisy,
								bm3d_param_ht
							);
						 auto func_end = high_resolution_clock::now();
						 std::chrono::duration<double, std::milli> duration = func_end - func_start;

						time_vec_gpu[0].second += duration.count();
					}
					//**************************************************************************
					const unsigned noisy_dist_vec_extents[3] =
						{
							(img_pixel_arr_raw_noisy.m_extents[1] - bm3d_param_ht.m_patch_size + 1) / bm3d_param_ht.m_reference_step,
							(img_pixel_arr_raw_noisy.m_extents[2] - bm3d_param_ht.m_patch_size + 1) / bm3d_param_ht.m_reference_step,
							bm3d_param_ht.m_max_grp_size
						};
					tensor_t < dist_t, 3 > noisy_dist_vec (noisy_dist_vec_extents);
					tensor_t < dist_t, 3 > noisy_dist_vec_tmp (noisy_dist_vec_extents);

					// if the search window can fit into a thread block
					if ((2 * bm3d_param_ht.m_hf_window_size + 2 - bm3d_param_ht.m_patch_size) *
						(2 * bm3d_param_ht.m_hf_window_size + 2 - bm3d_param_ht.m_patch_size) <= deviceProp.maxThreadsPerBlock)
					{
						std::cout << "SHORT Version of bm3d_compute_and_sort_dct_dist_vec_1st_chnl_ht is chosen." << std::endl;

						auto func_start = high_resolution_clock::now();

						bm3d_compute_and_sort_dct_dist_vec_1st_chnl_short_ht
							(
								noisy_dist_vec,
								img_pixel_arr_dct_noisy,
								img_pixel_arr_raw_noisy,
								bm3d_param_ht
							);

						auto func_end = high_resolution_clock::now();
						std::chrono::duration<double, std::milli> duration = func_end - func_start;
						time_vec_gpu[1].second +=  duration.count();
					}
					else // if the search window is larger than the thread block
					{
						std::cout << "LONG Version of bm3d_compute_and_sort_dct_dist_vec_1st_chnl_ht is chosen." << std::endl;

						auto func_start = high_resolution_clock::now();

						bm3d_compute_and_sort_dct_dist_vec_1st_chnl_long_ht
							(
								noisy_dist_vec,
								img_pixel_arr_dct_noisy,
								img_pixel_arr_raw_noisy,
								bm3d_param_ht
							);

						auto func_end = high_resolution_clock::now();
						std::chrono::duration<double, std::milli> duration = func_end - func_start;
						time_vec_gpu[1].second += duration.count();
					}
					//**************************************************************************
					const unsigned noisy_num_of_matches_vec_extents[2] =
						{
							(img_pixel_arr_raw_noisy.m_extents[1] - bm3d_param_ht.m_patch_size + 1) / bm3d_param_ht.m_reference_step,
							(img_pixel_arr_raw_noisy.m_extents[2] - bm3d_param_ht.m_patch_size + 1) / bm3d_param_ht.m_reference_step
						};
					tensor_t < unsigned, 2 > noisy_num_of_matches_vec (noisy_num_of_matches_vec_extents);
					tensor_t < unsigned, 2 > noisy_num_of_matches_vec_tmp (noisy_num_of_matches_vec_extents);

					{
						// auto func_start = high_resolution_clock::now();

						bm3d_filter_matches_ht
							(
								noisy_num_of_matches_vec,
								noisy_dist_vec,
								img_pixel_arr_dct_noisy,
								bm3d_param_ht
							);

						// clock_t func_end   = clock();
						// time_vec_gpu.push_back(std::make_pair < std::string, float > ("bm3d_filter_matches_ht", (func_end - func_start) / (1.0 * CLOCKS_PER_SEC) * 1000));
					}
					//**************************************************************************
					const unsigned img_pixel_arr_raw_basic_extents[3] =
						{
							img_pixel_arr_raw_noisy.m_extents[0],
							img_pixel_arr_raw_noisy.m_extents[1],
							img_pixel_arr_raw_noisy.m_extents[2]
						};
					tensor_t < image_t::pixel_t, 3 > img_pixel_arr_raw_basic (img_pixel_arr_raw_basic_extents);

					{
						auto func_start = high_resolution_clock::now();

						bm3d_hadtf_hard_inv_hadtf_inv_dct
							(
								img_pixel_arr_raw_basic,
								img_pixel_arr_dct_noisy,
								noisy_dist_vec,
								noisy_num_of_matches_vec,
								img_pixel_arr_raw_noisy,
								sigma_table,
								threshold_ht, bm3d_param_ht
							);

						auto func_end = high_resolution_clock::now();
						std::chrono::duration<double, std::milli> duration = func_end - func_start;
						time_vec_gpu[2].second += duration.count();
					}
					//**************************************************************************
					noisy_dist_vec          .~tensor_t();
					noisy_num_of_matches_vec.~tensor_t();
					img_pixel_arr_raw_noisy .~tensor_t();
					//**************************************************************************

					const unsigned basic_dist_vec_extents[3] =
						{
							(img_pixel_arr_raw_basic.m_extents[1] - bm3d_param_wn.m_patch_size + 1) / bm3d_param_wn.m_reference_step,
							(img_pixel_arr_raw_basic.m_extents[2] - bm3d_param_wn.m_patch_size + 1) / bm3d_param_wn.m_reference_step,
							 bm3d_param_wn.m_max_grp_size
						};
					tensor_t < dist_t, 3 > basic_dist_vec (basic_dist_vec_extents);

					// if the search window can fit into a thread block
					if ((2 * bm3d_param_wn.m_hf_window_size + 2 - bm3d_param_wn.m_patch_size) *
						(2 * bm3d_param_wn.m_hf_window_size + 2 - bm3d_param_wn.m_patch_size) <= deviceProp.maxThreadsPerBlock)
					{
						std::cout << "SHORT Version of bm3d_compute_and_sort_raw_dist_vec_1st_chnl_wn is chosen." << std::endl;

						auto func_start = high_resolution_clock::now();

						bm3d_compute_and_sort_raw_dist_vec_1st_chnl_short_wn
							(
								basic_dist_vec,
								img_pixel_arr_raw_basic,
								bm3d_param_wn
							);

						auto func_end = high_resolution_clock::now();
						std::chrono::duration<double, std::milli> duration = func_end - func_start;
						time_vec_gpu[3].second += duration.count();
					}
					else // if the search window is larger than the thread block
					{
						std::cout << "LONG Version of bm3d_compute_and_sort_raw_dist_vec_1st_chnl_wn is chosen." << std::endl;

						auto func_start = high_resolution_clock::now();

						bm3d_compute_and_sort_raw_dist_vec_1st_chnl_long_wn
							(
								basic_dist_vec,
								img_pixel_arr_raw_basic,
								bm3d_param_wn
							);

						auto func_end = high_resolution_clock::now();
						std::chrono::duration<double, std::milli> duration = func_end - func_start;
						time_vec_gpu[3].second += duration.count();
					}
					//**************************************************************************
					const unsigned img_pixel_arr_dct_basic_extents[5] =
						{
							img_pixel_arr_raw_basic.m_extents[0],
							img_pixel_arr_raw_basic.m_extents[1] - bm3d_param_wn.m_patch_size + 2,
							img_pixel_arr_raw_basic.m_extents[2] - bm3d_param_wn.m_patch_size + 2,
							bm3d_param_wn.m_patch_size,
							bm3d_param_wn.m_patch_size
						};
					tensor_t < image_t::pixel_t, 5 > img_pixel_arr_dct_basic (img_pixel_arr_dct_basic_extents);
					const unsigned basic_num_of_matches_vec_extents[2] =
						{
							(img_pixel_arr_raw_basic.m_extents[1] - bm3d_param_wn.m_patch_size + 1) / bm3d_param_wn.m_reference_step,
							(img_pixel_arr_raw_basic.m_extents[2] - bm3d_param_wn.m_patch_size + 1) / bm3d_param_wn.m_reference_step
						};
					tensor_t < unsigned, 2 > basic_num_of_matches_vec (basic_num_of_matches_vec_extents);

					{
						auto func_start = high_resolution_clock::now();

						bm3d_filter_matches_compute_dct_wn
							(
								img_pixel_arr_dct_basic,
								basic_num_of_matches_vec,
								basic_dist_vec,
								img_pixel_arr_raw_basic,
								bm3d_param_wn
							);

						auto func_end = high_resolution_clock::now();
						std::chrono::duration<double, std::milli> duration = func_end - func_start;
						time_vec_gpu[4].second += duration.count();
					}
					//**************************************************************************
					const unsigned img_pixel_arr_raw_final_extents[3] =
						{
							img_pixel_arr_raw_basic.m_extents[0],
							img_pixel_arr_raw_basic.m_extents[1],
							img_pixel_arr_raw_basic.m_extents[2]
						};
					tensor_t < image_t::pixel_t, 3 > img_pixel_arr_raw_final (img_pixel_arr_raw_final_extents);

					{
						auto func_start = high_resolution_clock::now();

						bm3d_hadtf_wiener_inv_hadtf_inv_dct
							(
								img_pixel_arr_raw_final,
								img_pixel_arr_dct_basic,
								img_pixel_arr_dct_noisy,
								basic_dist_vec,
								basic_num_of_matches_vec,
								img_pixel_arr_raw_basic,
								sigma_table,
								bm3d_param_wn
							);

						auto func_end = high_resolution_clock::now();
						std::chrono::duration<double, std::milli> duration = func_end - func_start;
						time_vec_gpu[5].second += duration.count();
					}
					//**************************************************************************
					boost::multi_array < image_t::pixel_t, 3 > img_pixel_arr_final_ndarr = img_pixel_arr_raw_final.to_ndarray();
					boost::multi_array < image_t::pixel_t, 3 > img_pixel_arr_basic_ndarr = img_pixel_arr_raw_basic.to_ndarray();
					img_pixel_arr_raw_final .~tensor_t();
					img_pixel_arr_dct_basic .~tensor_t();
					img_pixel_arr_dct_noisy .~tensor_t();
					basic_dist_vec          .~tensor_t();
					basic_num_of_matches_vec.~tensor_t();
					img_pixel_arr_raw_basic .~tensor_t();

					//**************************************************************************
					// Dump the Pixel Array to the Output Image File
					//**************************************************************************
					//image_t basic_subimg, final_subimg;

					//basic_subimg.load(img_pixel_arr_basic_ndarr);
					//final_subimg.load(img_pixel_arr_final_ndarr);

					//save to the really big picture image
					final_img.integrate_subimage(img_pixel_arr_final_ndarr, subMinX, subMaxX, subMinY, subMaxY);
					basic_img.integrate_subimage(img_pixel_arr_basic_ndarr, subMinX, subMaxX, subMinY, subMaxY);
				}
			}
		}
		sigma_table             .~tensor_t();
		if (raw_img.m_img_pixel_arr.shape()[0] == 3)
		{
			  raw_img.color_transform(color_tf_mode_inverse);
			basic_img.color_transform(color_tf_mode_inverse);
			final_img.color_transform(color_tf_mode_inverse);
			noisy_img.color_transform(color_tf_mode_inverse);
		}
		// basic_img.save("./png/" + img_fname + "/" + img_fname + "_basic.png");
		// final_img.save("./png/" + img_fname + "/" + img_fname + "_final.png");
		//**************************************************************************
		// Report the RMSE, PSNR, and runtime
		//**************************************************************************
		double rmse_basic = 0.0, psnr_basic, rmse_final = 0.0, psnr_final, rmse_noisy = 0.0, psnr_noisy;

	#pragma omp parallel for collapse (3) reduction (+:rmse_basic, rmse_final,rmse_noisy )
		for (unsigned i = 0; i < raw_img.m_img_pixel_arr.shape()[0]; ++i)
		{
			for (unsigned j = 0; j < raw_img.m_img_pixel_arr.shape()[1]; ++j)
			{
				for (unsigned k = 0; k < raw_img.m_img_pixel_arr.shape()[2]; ++k)
				{
					rmse_basic += (basic_img.m_img_pixel_arr[i][j][k] - raw_img.m_img_pixel_arr[i][j][k]) * (basic_img.m_img_pixel_arr[i][j][k] - raw_img.m_img_pixel_arr[i][j][k]);
					rmse_final += (final_img.m_img_pixel_arr[i][j][k] - raw_img.m_img_pixel_arr[i][j][k]) * (final_img.m_img_pixel_arr[i][j][k] - raw_img.m_img_pixel_arr[i][j][k]);
					rmse_noisy += (noisy_img.m_img_pixel_arr[i][j][k] - raw_img.m_img_pixel_arr[i][j][k]) * (noisy_img.m_img_pixel_arr[i][j][k] - raw_img.m_img_pixel_arr[i][j][k]);
				}
			}
		}
		tensor_t < image_t::pixel_t, 3 > img_pixel_arr_raw_noisy (noisy_img.m_img_pixel_arr);
		rmse_basic = std::sqrt(rmse_basic / img_pixel_arr_raw_noisy.m_size); psnr_basic = 20 * log(255.0 / rmse_basic) / log(10.0);
		rmse_final = std::sqrt(rmse_final / img_pixel_arr_raw_noisy.m_size); psnr_final = 20 * log(255.0 / rmse_final) / log(10.0);
		rmse_noisy = std::sqrt(rmse_noisy / img_pixel_arr_raw_noisy.m_size); psnr_noisy = 20 * log(255.0 / rmse_noisy) / log(10.0);

		std::cout << "RMSE: " << std::endl;
		std::cout << "\tnoisy: " << std::setw(10) << rmse_noisy << std::endl;
		std::cout << "\tbasic: " << std::setw(10) << rmse_basic << std::endl;
		std::cout << "\tfinal: " << std::setw(10) << rmse_final << std::endl;
		std::cout << "PSNR: " << std::endl;
		std::cout << "\tnoisy: " << std::setw(10) << psnr_noisy << std::endl;
		std::cout << "\tbasic: " << std::setw(10) << psnr_basic << std::endl;
		std::cout << "\tfinal: " << std::setw(10) << psnr_final << std::endl;

		//*************SAVE Images************************
		final_img.save(out_dir,img_fname, img_format, "final");
		basic_img.save(out_dir,img_fname, img_format, "basic");
		noisy_img.save(out_dir,img_fname, img_format, "noisy");
		//**************************************************************************
		float total_time = 0.0;


		std::cout << "DCT_HT\tDis_HT\tHT\tDis_W\tDCT_WN\tWeinF" << std::endl;
		for (std::vector < std::pair < std::string, float > > ::iterator iter = time_vec_gpu.begin(); iter != time_vec_gpu.end(); ++iter)
		{
			total_time += iter->second;
			std::cout << iter->second << "\t" ;
		}
		std::cout << std::endl;
		for (std::vector < std::pair < std::string, float > > ::iterator iter = time_vec_gpu.begin(); iter != time_vec_gpu.end(); ++iter)
		{
			std::cout << "Function Name: " << iter->first << std::endl;
			std::cout << "\tRuntime: " << std::setw(10) << iter->second << ", Weight: " << std::setw(10) << iter->second / total_time * 100 << "%" << std::endl;
		}
		std::cout << "Total time: " << std::setw(10) << total_time << std::endl;
		std::cout << "==============================================================" << std::endl;

		std::string outFilePath = out_dir+"/run_summary.txt";
		ifstream summaryFI(outFilePath);
		ofstream summaryF;
		if (!summaryFI)
		{
			summaryF.open (outFilePath.c_str(), ios::out );
			summaryF << "image" << "\t" << "Res. MP" << "\t" << "Runtime" << "\t" << "PSNR"  << "\t" << "PSNR noisy" << std::endl;
		}
		else
		{
			summaryFI.close();
			summaryF.open (outFilePath.c_str(), ios::out | ios::app );
		}

		summaryF << img_fname << "\t" << (double)height*width/1000000 << "\t" << total_time/1000 << "\t" << psnr_final << "\t" << psnr_noisy << std::endl;
		summaryF.close();

		std::string outFilePath2 = out_dir+"/run_summary_breakdown.txt";
		ifstream summaryFI2(outFilePath2);
		ofstream summaryF2;
		if (!summaryFI2)
		{
			summaryF2.open (outFilePath2.c_str(), ios::out );
			summaryF2 << "image" << "\t" << "DCT_HT\tDis_HT\tHT\tDis_W\tDCT_WN\tWeinF" << std::endl;
		}
		else
		{
			summaryFI2.close();
			summaryF2.open (outFilePath2.c_str(), ios::out | ios::app );
		}
		summaryF2 << img_fname << "\t" ;
		for (std::vector < std::pair < std::string, float > > ::iterator iter = time_vec_gpu.begin(); iter != time_vec_gpu.end(); ++iter)
			summaryF2 << iter->second << "\t" ;
		summaryF2 << std::endl;
		summaryF2.close();

	return 0;
}
