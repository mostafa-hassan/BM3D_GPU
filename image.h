#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>
#include <boost/multi_array.hpp>
#include <FreeImagePlus.h>
#define __image_debug__ 1

struct image_t
{
	typedef float pixel_t;
	
	enum class color_transform_t 
	{
		RGB_2_RGB,
		RGB_2_YUV  , YUV_2_RGB,
		RGB_2_YCBCR, YCBCR_2_RGB,
		RGB_2_OPP  , OPP_2_RGB
	};
	
	boost::multi_array < pixel_t, 3 > m_img_pixel_arr;


	// @brief load the pixel array from a PNG image file
	// @param fname: name of the  input image file
	void loadPNG(const std::string & fname);
	// @brief load the pixel array from a generic image file
	// @param fname: name of the  input image file
	void load(const std::string & in_dir, const std::string & out_dir, const std::string & fname, const std::string & format);
	// @brief load the pixel array from an ndarray
	// @param ndarray: multi-dimensional array
	void load(const boost::multi_array < pixel_t, 3 > & ndarray);
	// @brief load the subimage pixel array from an ndarray
	void load_subimage(const boost::multi_array < image_t::pixel_t, 3 > & ndarray, unsigned x_start, unsigned x_end, unsigned y_start, unsigned y_end );
	void integrate_subimage(const boost::multi_array < image_t::pixel_t, 3 > & ndarray, unsigned x_start, unsigned x_end, unsigned y_start, unsigned y_end );
	void setDimensions(unsigned nchnls, unsigned height,unsigned width);
	// @brief save the pixel array into an image file
	// @param fname: name of the output image file
	void save(const std::string & out_dir, const std::string & fname, const std::string & format, const std::string & identifier) const;
	void save(const std::string & fname) const;
	// @brief add a noise to the pixel array
	// @param sigma: standard deviation of the noise
	void add_noise(const float & sigma);
	// @brief get the table that contains the standard deviation of each channel
	// @param sigma: standard deviation of the noise
	std::vector < pixel_t > get_sigma_table(const pixel_t & sigma, const color_transform_t & color_tf_mode_forward);
	// @brief transform the color image
	// @param trans_flag: whether to go from RGB to YUV or the reverse direction
	void color_transform(const color_transform_t & color_tf_mode);
};

#endif // IMAGE_H
