#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

#pragma once
class ImageRepresentator
{
public:
	ImageRepresentator(const vector<Point2f>& spmGrid, const int knn);
	~ImageRepresentator(void);
	Mat GetRepresentation(const Mat& im, const vector<KeyPoint>& kps, 
		const Mat& featureDescriptor, const Mat& codebook);

	Mat LLCCoding(const Mat& codebook, const Mat& featureDescriptor, const int knn, const float lambda);
	Mat VQCoding(const Mat& codebook, const Mat& featureDescriptor);

	Mat sqrtPooling(const Mat& coding, vector<int> localIds, int numCell);
	Mat maxPooling(const Mat& coding, vector<int> localIds, int numCell);
	Mat absolutePooling(const Mat& coding, vector<int> localIds, int numCell);
	Mat sumPooling(const Mat& coding, vector<int> localIds, int numCell);
	Mat gridMaxPooling(const Mat& coding, vector<int> localIds, int numCell);

	int poolType;
	int codingType;
	
private:
	Mat ComputeCode(const Mat& descriptor, const Mat& localBase);
	void updateNNIds(const int knn, int id, float dist, vector<int>& nnIds, vector<float>& nnDists); 

	vector<Point2f> spmGrid;
	int knn;
};

