/*
 * File Description and Acknowledgments would be added upon completion
 */
#include <cstdio>	// file pointer
#include <cstring>	// memcpy
#include <ctime>	// time

#include <unistd.h>	// getpid
#include <png.h>	// PNG data structures and API functions
#include <omp.h>	// simple parallel processing

#include <boost/random.hpp> // Gaussian RV generator

#include "../inc/image.h"

enum class cpy_mode_t
{
	png_to_img,
	img_to_png
};

// @brief abort the reading process
// @param fp: file pointer
// @param ptr_ptr_png_read_struct: pointer to PNG read structure pointer
// @param ptr_ptr_png_info_struct: pointer to PNG info structure pointer
static void read_png_abort
(
	FILE * fp,
	png_struct ** ptr_ptr_png_read_struct,
	png_info   ** ptr_ptr_png_info_struct
)
{
	// destroy the read structure regardless of whether it has been constructeds
	png_destroy_read_struct
	(
		ptr_ptr_png_read_struct, 
		ptr_ptr_png_info_struct, 
		nullptr
	);

	// close the file that has been opened
	if (fp != nullptr && fp != stdin)
	{
		fclose(fp);
	}
	
	return;
}

// @brief abort the writing process
// @param fp: file pointer
// @param row_ptr: row pointer
// @param ptr_ptr_png_write_struct: pointer to PNG write structure pointer
// @param ptr_ptr_png_info__struct: pointer to PNG info  structure pointer
static void write_png_abort
(
	FILE * fp, png_bytepp ptr_row,
	png_struct ** ptr_ptr_png_write_struct,
	png_info   ** ptr_ptr_png_info_struct
)
{
	free(ptr_row);
	
	// destroy the write structure regardless of whether it has been constructed
	png_destroy_write_struct
	(
		ptr_ptr_png_write_struct,
		ptr_ptr_png_info_struct
	);

	// close the file that has been opened
	if (fp != nullptr && fp != stdout)
	{
		fclose(fp);
	}
}

// @brief load the pixel array from a generic image file
// @param fname: name of the  input image file
void image_t::load(const std::string & in_dir, const std::string & out_dir, const std::string & fname, const std::string & format)
{
	unsigned nchnls, height, width;
	// initialize the FreeImage library
	FreeImage_Initialise(TRUE);
	FIBITMAP *img = nullptr;
	std::string imageFilePath = in_dir;
	std::string imageFilePathNoExt = out_dir;
	int flags = 0;
	FREE_IMAGE_FORMAT FI_Format;
	if (format == "png")
	{

		imageFilePath += "/png/" + fname + "." + format;
		loadPNG(imageFilePath);
		return;
	}

	imageFilePath += "/raw/" + fname + "." + format;
	imageFilePathNoExt += "/raw/" + fname;
	flags = RAW_DISPLAY;
	FI_Format = FIF_RAW;

	if (!FreeImage_FIFSupportsReading(FI_Format))
		std::cout << "format not supported" << std::endl;
	img = FreeImage_Load(FI_Format, imageFilePath.c_str(), flags);

	if (img == nullptr)
	{
		std::cout << "Can't load file : " << imageFilePath << std::endl;
		exit(-1);
	}

	width = FreeImage_GetWidth(img);
	height = FreeImage_GetHeight(img);
	nchnls = 3;
	m_img_pixel_arr.resize(boost::extents[nchnls][height][width]);


	#pragma omp parallel for collapse(2)
	for (unsigned h = 0; h < height; ++h)
	{
		for (unsigned w = 0; w < width; ++w)
		{
			RGBQUAD color;
			FreeImage_GetPixelColor(img, w, height-1-h, &color);
			m_img_pixel_arr[0][h][w] = color.rgbRed;
			m_img_pixel_arr[1][h][w] = color.rgbGreen;
			m_img_pixel_arr[2][h][w] = color.rgbBlue;

		}
	}

	FreeImage_Save(FIF_BMP, img,(imageFilePathNoExt+"_read.bmp").c_str(), 0);


	FreeImage_Unload(img);
	// release the FreeImage library
	FreeImage_DeInitialise();



}
// @brief load the pixel array from an image file
// @param fname: name of the  input image file
void image_t::loadPNG(const std::string & fname)
{
	FILE * fp = nullptr;
	
#define PNG_SIG_LEN 4 // length of png signature	
	png_byte png_sig[PNG_SIG_LEN];

	png_struct * ptr_png_read_struct = nullptr;
	png_info   * ptr_png_info_struct = nullptr;
	
	// error code received from setjmp
	int err_code;
	
	unsigned nchnls, height, width;
	
	// 2D pixel array obtained from the PNG image file
	png_byte ** png_pixel_arr;
	
	if (fname == "-")
	{
		fp = stdin;
	}
	else // fname != "-"
	{
		// open the file as a binary file for input operations
		if ((fp = fopen(fname.c_str(), "rb")) == nullptr)
		{
			read_png_abort(fp, &ptr_png_read_struct, &ptr_png_info_struct); assert(0);
		}
	}
	
	// read in some of the signature bytes and check those signatures
	if (fread(png_sig, 1, PNG_SIG_LEN, fp) != PNG_SIG_LEN || png_sig_cmp(png_sig, 0, PNG_SIG_LEN) != 0)
	{
		read_png_abort(fp, &ptr_png_read_struct, &ptr_png_info_struct); assert(0);
	}
	
	// create and initialize the PNG read and info structure
	if ((ptr_png_read_struct = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr)) == nullptr)
	{
		read_png_abort(fp, &ptr_png_read_struct, &ptr_png_info_struct); assert(0);
	}
	if ((ptr_png_info_struct = png_create_info_struct(ptr_png_read_struct)) == nullptr)
	{
		read_png_abort(fp, &ptr_png_read_struct, &ptr_png_info_struct); assert(0);
	}
	
	// setup the error handler
	/*
	 * Side Note on setjmp and longjmp:
	 * setjmp and longjmp is similar to the interrupt mechanism in assembly program
	 * The two functions work as follows:
	 *	1.	an error handler is defined using setjmp according to the current environment
	 *	2.	program errors in all the following parts of the code can be issued to the 
	 *		error handler via longjmp(env, some_value) function call
	 *  3.	once the setjmp receives the error, it would return an nonzero value (the same value
	 *		sent from longjmp), which could then be used to check what kinds of errors have occurred
	 */
	
	if ((err_code = setjmp(png_jmpbuf(ptr_png_read_struct))) != 0)
	{
		/*
		 * if function setjmp returns an nonzero value, this means that we encountered 
		 * problems in reading the file, and therefore need to abort the reading process
		 */
		read_png_abort(fp, &ptr_png_read_struct, &ptr_png_info_struct); assert(0);
	}
	
	// setup the input control using the standard C streams
	png_init_io(ptr_png_read_struct, fp);
	
	// inform the libpng that some bytes have already been read as the signature bytes
	png_set_sig_bytes(ptr_png_read_struct, PNG_SIG_LEN);
	
	// read in the entire image at once
	png_read_png
	(
		ptr_png_read_struct,
		ptr_png_info_struct, 
		PNG_TRANSFORM_IDENTITY | PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_PACKING,
		nullptr
	);
	
	// get image information - width, height, channels
	nchnls = png_get_channels    (ptr_png_read_struct, ptr_png_info_struct);
	height = png_get_image_height(ptr_png_read_struct, ptr_png_info_struct);
	width  = png_get_image_width (ptr_png_read_struct, ptr_png_info_struct);
	
	// png_pixel_arr is a 2D array
	png_pixel_arr = png_get_rows(ptr_png_read_struct, ptr_png_info_struct);
	
	// resize the 3D pixel array into (c * h * w)
	m_img_pixel_arr.resize(boost::extents[nchnls][height][width]);
	
#pragma omp parallel for collapse(3)
	for (unsigned c = 0; c < nchnls; ++c)
	{
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				m_img_pixel_arr[c][h][w] = png_pixel_arr[h][w * nchnls + c];
			}
		}
	}

	// close the file and deallocate the read & info structure to complete the reading process
	read_png_abort(fp, &ptr_png_read_struct, &ptr_png_info_struct);
}

// @brief load the pixel array from an ndarray
// @param ndarray: multi-dimensional array
void image_t::load(const boost::multi_array < image_t::pixel_t, 3 > & ndarray)
{
	const unsigned nchnls = ndarray.shape()[0],
		           height = ndarray.shape()[1],
			       width  = ndarray.shape()[2];
				   
	m_img_pixel_arr.resize(boost::extents[nchnls][height][width]);
	
#pragma omp parallel for collapse (3)
	for (unsigned c = 0; c < nchnls; ++c)
	{
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				m_img_pixel_arr[c][h][w] = ndarray[c][h][w];
			}
		}
	}
}
void image_t::load_subimage(const boost::multi_array < image_t::pixel_t, 3 > & ndarray, unsigned x_start, unsigned x_end, unsigned y_start, unsigned y_end )
{
	const unsigned nchnls = ndarray.shape()[0],
		           height = y_end - y_start + 1,
			       width  = x_end - x_start + 1;

	m_img_pixel_arr.resize(boost::extents[nchnls][height][width]);

#pragma omp parallel for collapse (3)
	for (unsigned c = 0; c < nchnls; ++c)
	{
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				m_img_pixel_arr[c][h][w] = ndarray[c][y_start + h][x_start + w];
			}
		}
	}
}
void image_t::integrate_subimage(const boost::multi_array < image_t::pixel_t, 3 > & ndarray, unsigned x_start, unsigned x_end, unsigned y_start, unsigned y_end )
{
	const unsigned nchnls = ndarray.shape()[0],
	           height = y_end - y_start + 1,
		       width  = x_end - x_start + 1;
	#pragma omp parallel for collapse (3)
	for (unsigned c = 0; c < nchnls; ++c)
	{
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				m_img_pixel_arr[c][y_start + h][x_start + w] = ndarray[c][h][w];
			}
		}
	}
}
void image_t::setDimensions(unsigned nchnls, unsigned height,unsigned width)
{
	m_img_pixel_arr.resize(boost::extents[nchnls][height][width]);
	#pragma omp parallel for collapse (3)
	for (unsigned c = 0; c < nchnls; ++c)
	{
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				m_img_pixel_arr[c][h][w] = 0;
			}
		}
	}
}
void image_t::save(const std::string & out_dir, const std::string & fname, const std::string & format, const std::string & identifier) const
{
	std::string imageFilePath;
	if (format == "png")
		imageFilePath = out_dir + "/png/" + fname + "_" + identifier +".png";
	else
		imageFilePath = out_dir + "/raw/" + fname + "_" + identifier +".png";

	save (imageFilePath);
}
// @brief save the pixel array into a image file
// @param fname: name of the output image files
void image_t::save(const std::string & fname) const
{
	FILE * fp;

	png_struct * ptr_png_write_struct = nullptr;
	png_info   * ptr_png_info_struct  = nullptr;
	
	// error code received from setjmp
	int err_code;
	
	// color type determined by the number of chnls
	int png_color_type;
	
	const unsigned nchnls = m_img_pixel_arr.shape()[0],
		           height = m_img_pixel_arr.shape()[1],
			       width  = m_img_pixel_arr.shape()[2];
				   
	// row pointer
	png_bytepp ptr_row  = (png_bytepp)(malloc(m_img_pixel_arr.shape()[1] * sizeof(png_bytep)));
	
	boost::multi_array < png_byte, 2 > png_pixel_arr;
	
	if (fname == "-")
	{
		fp = stdout;
	}
	else
	{
		// open the file as a binary file for output operations
		if ((fp = fopen(fname.c_str(), "wb")) == nullptr)
		{
			write_png_abort(fp, ptr_row, &ptr_png_write_struct, &ptr_png_info_struct); assert(0);
		}
	}
	
	// create and initialize the PNG write and info structure
	if ((ptr_png_write_struct = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr)) == nullptr)
	{
		write_png_abort(fp, ptr_row, &ptr_png_write_struct, &ptr_png_info_struct); assert(0);
	}
	if ((ptr_png_info_struct = png_create_info_struct(ptr_png_write_struct)) == nullptr)
	{
		write_png_abort(fp, ptr_row, &ptr_png_write_struct, &ptr_png_info_struct); assert(0);
	}
	
	// setup the error handler
	/*
	 * For a detailed explanation of the error handler, please refer to the constructor
	 */
	if ((err_code = setjmp(png_jmpbuf(ptr_png_write_struct))) != 0)
	{
		write_png_abort(fp, ptr_row, &ptr_png_write_struct, &ptr_png_info_struct); assert(0);
	}
	
	// setup the input control using the standard C streams
	png_init_io(ptr_png_write_struct, fp);
	
	// determine the color type according to the number of channels
	switch (nchnls)
	{
		case 1:
		{
			png_color_type = PNG_COLOR_TYPE_GRAY;
			break;
		}
		case 2:
		{
			png_color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
			break;
		}
		case 3:
		{
			png_color_type = PNG_COLOR_TYPE_RGB;
			break;
		}
		case 4:
		{
			png_color_type = PNG_COLOR_TYPE_RGB_ALPHA;
			break;
		}
		default:
		{
			write_png_abort(fp, ptr_row, &ptr_png_write_struct, &ptr_png_info_struct); assert(0);
		}
	}
	// set image handler
	png_set_IHDR
	(
		ptr_png_write_struct,
		ptr_png_info_struct,
		width,
		height,
		8 /* bit depth */,
		png_color_type,
		PNG_INTERLACE_ADAM7,
		PNG_COMPRESSION_TYPE_BASE,
		PNG_FILTER_TYPE_BASE
	);
	
	png_write_info(ptr_png_write_struct, ptr_png_info_struct);
	
	// resize the 2D pixel array into (h * (w * c))
	png_pixel_arr.resize(boost::extents[height][width * nchnls]);
	
#pragma omp parallel for collapse(3)
	for (unsigned c = 0; c < nchnls; ++c)
	{
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				pixel_t pixel_val = round(m_img_pixel_arr[c][h][w]);
				
				png_pixel_arr[h][w * nchnls + c] = pixel_val;
				if (pixel_val > 255.0) png_pixel_arr[h][w * nchnls + c] = 255.0;
				if (pixel_val <   0.0) png_pixel_arr[h][w * nchnls + c] =   0.0;
			}
		}
	}
	
#pragma omp parallel for
	for (unsigned h = 0; h < height; ++h)
	{
		ptr_row[h] = &png_pixel_arr[h][0];
	}
	
	// write to the PNG image file
	png_write_image(ptr_png_write_struct, ptr_row);
	// end the writing process
	png_write_end(ptr_png_write_struct, ptr_png_info_struct);
	
	// close the file and deallocate the write & info structure to complete the reading process
	write_png_abort(fp, ptr_row, &ptr_png_write_struct, &ptr_png_info_struct);
}

// @brief add a noise to the pixel array
// @param sigma: standard deviation of the noise
void image_t::add_noise(const image_t::pixel_t & sigma)
{
	// construct a random number generator according to the current time and process ID
	boost::mt19937 gen (100);
	boost::variate_generator< boost::mt19937 &, boost::normal_distribution <> >
		rng (gen, boost::normal_distribution <> (0, sigma));
	
	const unsigned nchnls = m_img_pixel_arr.shape()[0],
			       height = m_img_pixel_arr.shape()[1],
			       width  = m_img_pixel_arr.shape()[2];
	
	for (unsigned c = 0; c < nchnls; ++c)
	{
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				m_img_pixel_arr[c][h][w] += rng() /* noise */;
			}
		}
	}
}

// @brief get the table that contains the standard deviation of each channel
// @param sigma: standard deviation of the noise
std::vector < image_t::pixel_t > image_t::get_sigma_table(const pixel_t & sigma, const color_transform_t & color_tf_mode_forward)
{
	const unsigned nchnls = m_img_pixel_arr.shape()[0];
	
	// make sure that the image is RGB
	if (nchnls != 3){return std::vector < image_t::pixel_t > (nchnls, sigma);}
	
	std::vector < pixel_t > sigma_table (nchnls);
	
	if (color_tf_mode_forward == image_t::color_transform_t::RGB_2_YUV)
	{
		sigma_table[0] = sqrt(0.299   * 0.299   + 0.587   * 0.587   + 0.114   * 0.114  ) * sigma;
		sigma_table[1] = sqrt(0.14713 * 0.14713 + 0.28886 * 0.28886 + 0.436   * 0.436  ) * sigma;
		sigma_table[2] = sqrt(0.615   * 0.615   + 0.51498 * 0.51498 + 0.10001 * 0.10001) * sigma;
	}
	else
	if (color_tf_mode_forward == image_t::color_transform_t::RGB_2_YCBCR)
	{
		sigma_table[0] = sqrt(0.299 * 0.299 + 0.587 * 0.587 + 0.114 * 0.114) * sigma;
		sigma_table[1] = sqrt(0.169 * 0.169 + 0.331 * 0.331 + 0.500 * 0.500) * sigma;
		sigma_table[2] = sqrt(0.500 * 0.500 + 0.419 * 0.419 + 0.081 * 0.081) * sigma;
	}
	else
	if (color_tf_mode_forward == image_t::color_transform_t::RGB_2_OPP)
	{
		sigma_table[0] = sqrt(0.333 * 0.333 + 0.333 * 0.333 + 0.333 * 0.333) * sigma;
		sigma_table[1] = sqrt(0.5   * 0.5                   + 0.5   * 0.5  ) * sigma;
		sigma_table[2] = sqrt(0.25  * 0.25  + 0.5   * 0.5   + 0.25  * 0.25 ) * sigma;
	}
	else
	{
		sigma_table[0] = sigma;
		sigma_table[1] = sigma;
		sigma_table[2] = sigma;
	}
	
	return sigma_table;
}

// @brief transform the color image
// @param trans_flag: whether to go from RGB to YUV or the reverse direction
void image_t::color_transform(const image_t::color_transform_t & color_tf_mode)
{
	const unsigned nchnls = m_img_pixel_arr.shape()[0],
		           height = m_img_pixel_arr.shape()[1],
			       width  = m_img_pixel_arr.shape()[2];
	// make sure that the image is RGB
	assert(nchnls == 3);
	boost::multi_array < pixel_t, 3 > img_pixel_arr (boost::extents[nchnls][height][width]);
	
	if (color_tf_mode == image_t::color_transform_t::RGB_2_RGB)
	{
		return;
	}
	if (color_tf_mode == image_t::color_transform_t::RGB_2_YUV)
	{
#pragma omp parallel for collapse (2)
		for (unsigned h = 0; h < height; ++h)
		{	
			for (unsigned w = 0; w < width; ++w)
			{
				img_pixel_arr[0][h][w] =  0.299   * m_img_pixel_arr[0][h][w] + 0.587   * m_img_pixel_arr[1][h][w] + 0.114   * m_img_pixel_arr[2][h][w];
				img_pixel_arr[1][h][w] = -0.14713 * m_img_pixel_arr[0][h][w] - 0.28886 * m_img_pixel_arr[1][h][w] + 0.436   * m_img_pixel_arr[2][h][w];
				img_pixel_arr[2][h][w] =  0.615   * m_img_pixel_arr[0][h][w] - 0.51498 * m_img_pixel_arr[1][h][w] - 0.10001 * m_img_pixel_arr[2][h][w];
			}
		}
	}
	if (color_tf_mode == image_t::color_transform_t::YUV_2_RGB)
	{
#pragma omp parallel for collapse (2)
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				img_pixel_arr[0][h][w] = m_img_pixel_arr[0][h][w]                                      + 1.13983 * m_img_pixel_arr[2][h][w];
				img_pixel_arr[1][h][w] = m_img_pixel_arr[0][h][w] - 0.39465 * m_img_pixel_arr[1][h][w] - 0.5806  * m_img_pixel_arr[2][h][w];
				img_pixel_arr[2][h][w] = m_img_pixel_arr[0][h][w] + 2.03211 * m_img_pixel_arr[1][h][w]                                     ;
			}
		}
	}
	if (color_tf_mode == image_t::color_transform_t::RGB_2_YCBCR)
	{
#pragma omp parallel for collapse (2)
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				img_pixel_arr[0][h][w] =  0.299 * m_img_pixel_arr[0][h][w] + 0.587 * m_img_pixel_arr[1][h][w] + 0.114 * m_img_pixel_arr[2][h][w];
				img_pixel_arr[1][h][w] = -0.169 * m_img_pixel_arr[0][h][w] - 0.331 * m_img_pixel_arr[1][h][w] + 0.500 * m_img_pixel_arr[2][h][w];
				img_pixel_arr[2][h][w] =  0.500 * m_img_pixel_arr[0][h][w] - 0.419 * m_img_pixel_arr[1][h][w] - 0.081 * m_img_pixel_arr[2][h][w];
			}
		}
	}
	if (color_tf_mode == image_t::color_transform_t::YCBCR_2_RGB)
	{
#pragma omp parallel for collapse (2)
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				img_pixel_arr[0][h][w] = m_img_pixel_arr[0][h][w]                                    + 1.402 * m_img_pixel_arr[2][h][w];
				img_pixel_arr[1][h][w] = m_img_pixel_arr[0][h][w] - 0.344 * m_img_pixel_arr[1][h][w] - 0.714 * m_img_pixel_arr[2][h][w];
				img_pixel_arr[2][h][w] = m_img_pixel_arr[0][h][w] + 1.772 * m_img_pixel_arr[1][h][w]                                   ;
			}
		}
	}
	if (color_tf_mode == image_t::color_transform_t::RGB_2_OPP)
	{
#pragma omp parallel for collapse (2)
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				img_pixel_arr[0][h][w] = 0.333 * m_img_pixel_arr[0][h][w] + 0.333 * m_img_pixel_arr[1][h][w] + 0.333 * m_img_pixel_arr[2][h][w];
				img_pixel_arr[1][h][w] = 0.5   * m_img_pixel_arr[0][h][w]                                    - 0.5   * m_img_pixel_arr[2][h][w];
				img_pixel_arr[2][h][w] = 0.25  * m_img_pixel_arr[0][h][w] - 0.5   * m_img_pixel_arr[1][h][w] + 0.25  * m_img_pixel_arr[2][h][w];
			}
		}
	}
	if (color_tf_mode == image_t::color_transform_t::OPP_2_RGB)
	{
#pragma omp parallel for collapse (2)
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				img_pixel_arr[0][h][w] = m_img_pixel_arr[0][h][w] + m_img_pixel_arr[1][h][w] + 0.666 * m_img_pixel_arr[2][h][w];
				img_pixel_arr[1][h][w] = m_img_pixel_arr[0][h][w] +                          - 1.333 * m_img_pixel_arr[2][h][w];
				img_pixel_arr[2][h][w] = m_img_pixel_arr[0][h][w] - m_img_pixel_arr[1][h][w] + 0.666 * m_img_pixel_arr[2][h][w];
			}
		}
	}
	
#pragma omp parallel for collapse (3)
	for (unsigned c = 0; c < nchnls; ++c)
	{
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				m_img_pixel_arr[c][h][w] = img_pixel_arr[c][h][w];
			}
		}
	}
}
