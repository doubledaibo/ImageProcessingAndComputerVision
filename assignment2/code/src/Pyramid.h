#include "CImg.h"
#include <vector>

using namespace cimg_library;
using namespace std;

void Conv(CImg<float> src, CImg<float> filter, CImg<float>& dst);

CImg<float> GaussianFilter(int size, float delta);

struct pyramid {
	vector<CImg<float>> levels;
};

CImg<float> Difference(CImg<float> op1, CImg<float> op2);

pyramid GaussianPyramid(CImg<float> src, int num_levels, CImg<float> filter);

pyramid LaplacianPyramid(CImg<float> src, int num_levels, CImg<float> filter);