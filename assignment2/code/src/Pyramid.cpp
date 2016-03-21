#include "Pyramid.h"
#include <math.h>

#ifndef inRange
#define inRange(x,y,width,height) x >= 0 && x < width && y >= 0 && y < height
#endif

void Conv(CImg<float> src, CImg<float> filter, CImg<float>& dst) {
	if (src.width() != dst.width() || src.height() != dst.height()) {
		printf("dimension not match!\n");
		return;
	}

	int iw = src.width();
	int ih = src.height();
	int ic = src.spectrum();
	int fw = filter.width();
	int fh = filter.height();
	int cx = fw / 2;
	int cy = fh / 2;

	//conv for each channel
	for (int c = 0; c < ic; ++c) 
		for (int x = 0; x < iw; ++x)
			for (int y = 0; y < ih; ++y) {
				float sum = 0;
				for (int fx = 0; fx < fw; ++fx)
					for (int fy = 0; fy < fh; ++fy) {
						int xx = x - fx + cx;
						int yy = y - fy + cy;
						if (inRange(xx, yy, iw, ih))
							sum += filter(fx, fy, 0, 0) * src(xx, yy, 0, c);
					}
				dst(x, y, 0, c) = sum;
			}
}

CImg<float> GaussianFilter(int size, float delta) {
	CImg<float> filter(size, size, 1, 1, 0);
	float c = size / 2.f;
	float sum = 0;
	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j) {
			filter(i, j, 0, 0) = exp((- (i + 1 - c) * (i + 1 - c) - (j + 1 - c) * (j + 1 - c)) / (2 * delta));
			sum += filter(i, j, 0, 0);
		}
    
	//normalize 
	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j) 
			filter(i, j, 0, 0) /= sum;
	return filter;
}

CImg<float> Difference(CImg<float> op1, CImg<float> op2) {
	int w = op1.width();
	int h = op1.height();
	int c = op1.spectrum();
	CImg<float> diff(w, h, 1, c, 0);

	for (int x = 0; x < w; ++x)
		for (int y = 0; y < h; ++y)
			for (int s = 0; s < c; ++s)
				diff(x, y, 0, s) = op1(x, y, 0, s) - op2(x, y, 0, s);

	return diff;
}

pyramid GaussianPyramid(CImg<float> src, int num_levels, CImg<float> filter) {
	pyramid gp;
	CImg<float> dst(src.width(), src.height(), 1, src.spectrum(), 0);
	gp.levels.push_back(src);
	for (int i = 0; i < num_levels; ++i) {
		Conv(src, filter, dst);
		gp.levels.push_back(dst);
		src = dst;
	}
	return gp;
}

pyramid LaplacianPyramid(CImg<float> src, int num_levels, CImg<float> filter) {
	pyramid lp;
	lp = GaussianPyramid(src, num_levels, filter);
	for (int i = 0; i < num_levels - 1; ++i) 
		lp.levels[i] = Difference(lp.levels[i], lp.levels[i + 1]);
	return lp;
}