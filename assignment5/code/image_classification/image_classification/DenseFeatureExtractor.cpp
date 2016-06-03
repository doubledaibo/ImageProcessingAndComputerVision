#include "DenseFeatureExtractor.h"


DenseFeatureExtractor::DenseFeatureExtractor(void) {
}


DenseFeatureExtractor::~DenseFeatureExtractor(void) {
}

/* ExtractFeature is for extracting feature descriptor of an image
   Input:
           im: A grayscale image in height x width.
		   keypoints: The vector for storing the key points.
   Output:
           siftMat: The SIFT descriptors of the input image.
*/
Mat DenseFeatureExtractor::ExtractFeature(const Mat& im, vector<KeyPoint>& keypoints) {
	// Initialize the feature detector and the SIFT extractor. 
	// You may choose to combine other feature descriptor.
	DenseFeatureDetector detector = DenseFeatureDetector(1.0, 1.0, 0.1, 6, 8, true, false);
	SiftDescriptorExtractor siftExtractor;
	SurfDescriptorExtractor surfExtractor;
	//OrbDescriptorExtractor orbExtractor;
	//BriefDescriptorExtractor briefExtractor;
	//BriefDescriptorExtractor briskExtractor;
	Mat siftMat;
	Mat surfMat;
	//Mat orbMat;
	//Mat briefMat;
	//Mat briskMat;
	Mat combineMat;
	/* TODO 1: 
       Detect the feature descriptors using "detector". Place the key points in 
	   the vector "keypoints". Then use the input image to compute the descriptor
	   of each key points.
	*/
	detector.detect(im, keypoints);
	siftExtractor.compute(im, keypoints, siftMat);
	surfExtractor.compute(im, keypoints, surfMat);
	//orbExtractor.compute(im, keypoints, orbMat);
	//briefExtractor.compute(im, keypoints, briefMat);
	//briskExtractor.compute(im, keypoints, briskMat);
	vector<Mat> features;
	features.push_back(siftMat);
	features.push_back(surfMat);
	//features.push_back(orbMat);
	//features.push_back(briefMat);
	//features.push_back(briskMat);
	hconcat(features, combineMat);
	// End TODO 1

	return combineMat;
}