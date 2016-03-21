///////////////////////////////////////////////////////////////////////////
//
// NAME
//  BlendImages.cpp -- blend together a set of overlapping images
//
// DESCRIPTION
//  This routine takes a collection of images aligned more or less horizontally
//  and stitches together a mosaic.
//
//  The images can be blended together any way you like, but I would recommend
//  using a soft halfway blend of the kind Steve presented in the first lecture.
//
// SEE ALSO
//  BlendImages.h       longer description of parameters
//
// Copyright ?Richard Szeliski, 2001.  See Copyright.h for more details
//
///////////////////////////////////////////////////////////////////////////

#include "ImageLib/ImageLib.h"
#include "BlendImages.h"
#include "WarpSpherical.h"
#include "Pyramid.h"
#include <math.h>


#ifndef M_PI 
#define M_PI    3.1415926536f
#endif // M_PI

#define verbose 1

CImg<float> ConvertToCImg(CByteImage& src)
{
	int width = src.Shape().width;
	int height = src.Shape().height;
	int nBands = src.Shape().nBands;
	CImg<float> dst(width, height, 1, nBands);
	for(int h = 0 ; h < height ; ++h)
	{
		for(int w = 0 ; w < width ; ++w)
		{
			for(int b = 0 ; b < nBands ; ++b)
			{
				dst(w, h, 0, b) = src.Pixel(w, h, b);
			}
		}
	}
	return dst;
}

CByteImage ConvertToImglib(CImg<float>& src)
{
	int width = src.width();
	int height = src.height();
	int nBands = src.spectrum();
	CByteImage dst(width, height, nBands);
	for(int h = 0 ; h < height ; ++h)
	{
		for(int w = 0 ; w < width ; ++w)
		{
			for(int b = 0 ; b < nBands ; ++b)
			{
				dst.Pixel(w, h, b) = src(w, h, 0, b);
			}
		}
	}
	return dst;
}


/******************* TO DO *********************
 * SetImageAlpha:
 *	INPUT:
 *		img: an image to be added to the final panorama
 *		blendRadius: radius of the blending function
 *	OUTPUT:
 *		set the alpha values of img
 *		pixels near the center of img should have higher alpha
 *		use blendRadius to determine how alpha decreases
 */
static void SetImageAlpha(CImg<float>& img, float blendRadius, int type)
{
	// *** BEGIN TODO ***
	// fill in this routine..
	int width = img.width();
	int height = img.height();
	float cw = width / 2.0f;
	float ch = height / 2.0f;

	if (type == 0) {
		//no blending
		for (int x = 0; x < width; ++x)
			for (int y = 0; y < height; ++y) 
				img(x, y, 0, 3) = 255;
	} else if (type == 1) {
		// linear blending
		for (int x = 0; x < width; ++x)
			for (int y = 0; y < height; ++y) 
				img(x, y, 0, 3) = (1 - abs(x - cw) / cw) * (1 - abs(y - ch) / ch);
	} else 
		printf("Unsupported alpha type: %d!\n", type);
	// *** END TODO ***
}

/******************* TO DO *********************
 * AccumulateBlend:
 *	INPUT:
 *		img: a new image to be added to acc
 *		acc: portion of the accumulated image where img is to be added
 *		blendRadius: radius of the blending function
 *	OUTPUT:
 *		add a weighted copy of img to the subimage specified in acc
 *		the first 3 band of acc records the weighted sum of pixel colors
 *		the fourth band of acc records the sum of weight
 */
static void AccumulateBlend(CImg<float>& img, CImg<float>& acc, float blendRadius)
{
	// *** BEGIN TODO ***
	// fill in this routine..
	int width = img.width();
	int height = img.height();
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			float alpha = img(x, y, 0, 3);
			for (int k = 0; k < 3; ++k)
				acc(x, y, 0, k) += img(x, y, 0, k) * alpha;
			acc(x, y, 0, 3) += alpha;
		}

	// *** END TODO ***
}

static void SetOriginalAlpha(CImg<float>& src, CImg<float>& dst)
{
	int width = src.width();
	int height = src.height();
	float cw = width / 2.0f;
	float ch = height / 2.0f;

	for (int x = 0; x < width; ++x)
		for (int y = 0; y < height; ++y) 
			dst(x, y, 0, 3) = src(x, y, 0, 3);
}

static void Accumulate(CImg<float>& img, CImg<float>& acc)
{
	// *** BEGIN TODO ***
	// fill in this routine..
	int width = img.width();
	int height = img.height();
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) 
			for (int k = 0; k < 3; ++k) 
				acc(x, y, 0, k) = acc(x, y, 0, k) + img(x, y, 0, k);
	// *** END TODO ***
}

static void setOpaque(CImg<float>& img)
{
	int width = img.width();
	int height = img.height();
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			// set alpha to opaque for foreground
			if (img(x, y, 0, 0) + img(x, y, 0, 1) + img(x, y, 0, 2) > 0)
				img(x, y, 0, 3) = 255;
			else
				img(x, y, 0, 3) = 0;
		}
}

/******************* TO DO *********************
 * NormalizeBlend:
 *	INPUT
 *		acc: input image whose alpha channel (4th channel) contains
 *		     normalizing weight values
 *		img: where output image will be stored
 *	OUTPUT:
 *		normalize r,g,b values (first 3 channels) of acc and store it into img
 */
static void NormalizeBlend(CImg<float>& acc, CImg<float>& img)
{
	// *** BEGIN TODO ***
	// fill in this routine..
	int width = img.width();
	int height = img.height();
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) 
			for (int k = 0; k < 3; ++k) 
				img(x, y, 0, k) = acc(x, y, 0, k) / acc(x, y, 0, 3);
	
	setOpaque(img);

	// *** END TODO ***
}

void LaplacianBlend(vector<CImg<float>> imgs, CImg<float> &compImage, float blendRadius, int num_levels) {
	vector<CImg<float>> as;
	pyramid py;
	int nimg = imgs.size();
	int w = compImage.width();
	int h = compImage.height();
	int c = compImage.spectrum();
	CImg<float> filter = GaussianFilter(7, 0.5);
	printf("filter done...\n");
	for (int i = 0; i < num_levels; ++i) {
		CImg<float> acc(w, h, 1, c, 0);
		as.push_back(acc);
	}

	for (int i = 0; i < nimg; ++i) {
		py = LaplacianPyramid(imgs[i], num_levels, filter);

		printf("pyramid %d done...\n", i);
		for (int j = 0; j < num_levels; ++j) {
			SetOriginalAlpha(imgs[i], py.levels[j]);
			AccumulateBlend(py.levels[j], as[j], blendRadius);
		}
	}
	
    for (int i = 0; i < num_levels; ++i)
		NormalizeBlend(as[i], as[i]);

	if (verbose) {
		for (int i = 0; i < num_levels; ++i) {
			CImg<float> tmp = as[i];
			setOpaque(tmp);
			char buffer[50];
			sprintf(buffer, "pyramid_level_%d_blending_result.tga", i);
			int x1 = compImage.width() / 3;
			int x2 = compImage.width() - x1 + 100;
			int y1 = compImage.height() / 3;
			int y2 = compImage.height() - y1;
			y1 = y1	/ 2;
			CImg<float> shrink(x2 - x1 + 1, y2 - y1 + 1, 1, 4, 0);
			for (int x = x1; x < x2; ++x)
			for (int y = y1; y < y2; ++y)
				for (int k = 0; k < 4; ++k)
					shrink(x - x1, y - y1, 0, k) = tmp(x, y, 0, k);
			WriteFile(ConvertToImglib(shrink), buffer);
		}
	}

	for (int i = 0; i < num_levels; ++i) 
		Accumulate(as[i], compImage);
	setOpaque(compImage);
	printf("blending done..\n");
}

/******************* TO DO *********************
 * BlendImages:
 *	INPUT:
 *		ipv: list of input images and their global positions in the mosaic
 *		f: focal length
 *		blendRadius: radius of the blending function
 *		type: type of blending
 *	OUTPUT:
 *		create & return final mosaic by blending all images
 */
CImg<float> BlendImages(CImagePositionV& ipv, float f, float blendRadius, int type, int num_levels)
{
    // Assume all the images are of the same shape (for now)
    CImg<float>& img0 = ipv[0].img;
    //CShape sh        = img0.Shape();
    int width        = img0.width();
    int height       = img0.height();
	int nBands       = img0.spectrum();
    int dim[2]       = {width, height};

	int nTheta = (int) (2*M_PI*f / 2 + 0.5);
	int nPhi = (int) (M_PI*f / 2 + 0.5);

    // Create a floating point accumulation image
    //CShape mShape(nTheta, nPhi, nBands);
	
	CImg<float> compImage(nTheta, nPhi, 1, nBands, 0);
	if (type == 0 || type == 1) {
		//0: no blending, 1: linear blending
		CImg<float> accumulator(nTheta, nPhi, 1, nBands, 0);

		//accumulator.ClearPixels();

		// Add in all of the images
		for (unsigned int i = 0; i < ipv.size(); i++)
		{
			// Warp the image into spherical coordinates
			CTransform3x3 M = ipv[i].position;
			CImg<float>& src = ipv[i].img;
			SetImageAlpha(src, blendRadius, type);
			CImg<float> uv = WarpSphericalField(CShape(src.width(), src.height(), src.spectrum()), CShape(nTheta, nPhi, nBands), f, M);
			CImg<float>& dst = src.warp(uv);
			// Perform the accumulation
			AccumulateBlend(dst, accumulator, blendRadius);
			uv.clear();
			dst.clear();
		}
		// Normalize the results
		NormalizeBlend(accumulator, compImage);
	} else if (type == 2) {
		//2: Laplacian Blending
		vector<CImg<float>> ims;
		for (int i = 0; i < ipv.size(); i++) {
			CTransform3x3 M = ipv[i].position;
			CImg<float>& src = ipv[i].img;
			SetImageAlpha(src, blendRadius, 1);
			CImg<float> uv = WarpSphericalField(CShape(src.width(), src.height(), src.spectrum()), CShape(nTheta, nPhi, nBands), f, M);
			CImg<float>& dst = src.warp(uv);
			ims.push_back(dst);
			uv.clear();
		}
		printf("preparing img done...\n");
		LaplacianBlend(ims, compImage, blendRadius, num_levels);
	} else {
		printf("Unsupported blending type: %d\n", type);
	}
	int x1 = compImage.width() / 3;
	int x2 = compImage.width() - x1 + 100;
	int y1 = compImage.height() / 3;
	int y2 = compImage.height() - y1;
	y1 = y1 / 2;
	CImg<float> shrink(x2 - x1 + 1, y2 - y1 + 1, 1, 4, 0);
	for (int x = x1; x < x2; ++x)
		for (int y = y1; y < y2; ++y)
			for (int k = 0; k < 4; ++k)
				shrink(x - x1, y - y1, 0, k) = compImage(x, y, 0, k);
    return shrink;
}

