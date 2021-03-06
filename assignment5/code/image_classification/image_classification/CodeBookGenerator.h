#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

#pragma once
class CodeBookGenerator
{
public:
	CodeBookGenerator(const int clusterNum);
	~CodeBookGenerator(void);
	Mat GenerateCodebook(const Mat& samples);
private:
	int codebookSize;
};

