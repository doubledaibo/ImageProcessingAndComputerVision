#include "Detector.h"

Detector::Detector() {
}

Detector::~Detector(void) {
}

void Detector::TrainDetector(const Mat& trainSample, const Mat& trainLabel, 
	const int* featureSizes, const CvSVMParams& params) {
	this->train(trainSample, trainLabel, Mat(), Mat(), params);
	Mat tem1 = TransSV2Detector();
	Mat tmp2(featureSizes[0], featureSizes[1], CV_32FC(featureSizes[2]), tem1.data);
	tmp2.copyTo(detector);
}

Mat Detector::TransSV2Detector() {
	int numOfFeatures = this->var_all;
	int numSupportVectors = this->get_support_vector_count();
	const CvSVMDecisionFunc *dec = this->decision_func;
	Mat detector(1, numOfFeatures, CV_32FC1, Scalar(0));
	for (int i = 0; i < numSupportVectors; ++i) {
		float alpha = *(dec->alpha+i);
		const float* supportVector = this->get_support_vector(i);
		for(int j = 0; j < numOfFeatures; j++) {
			detector.at<float>(0, j) += alpha * *(supportVector+j);
		}
	}
	return -detector;
}

void Detector::DisplayDetection(Mat im, Mat top) {
	for (int i = 0 ; i < top.size[0] ; ++i) {
		rectangle(im, Point(top.at<float>(i, 1), top.at<float>(i, 0)), 
			Point(top.at<float>(i, 3), top.at<float>(i, 2)), 
			Scalar(0, 0, 255), 2);
	}
	namedWindow("Display detection", WINDOW_AUTOSIZE);
	imshow("Display detection", im); 
			
	waitKey(0);
	destroyWindow("Display detection");
}

void Detector::SetDetector(const Mat det) {
	det.copyTo(this->detector);
}

Mat Detector::GetDetector() {
	return detector;
}

/* Detection is for single scale detection
   Input:
           1. im: A grayscale image in height x width.
           2. ext: The pre-defined HOG extractor.
           3. threshold: The constant threshold to control the prediction.
   Output:
           1. bbox: The predicted bounding boxes in this format (n x 5 matrix):
                                    x11 y11 x12 y12 s1
                                    ... ... ... ... ...
                                    xi1 yi1 xi2 yi2 si
                                    ... ... ... ... ...
                                    xn1 yn1 xn2 yn2 sn
                    where n is the number of bounding boxes. For the ith 
                    bounding box, (xi1,yi1) and (xi2, yi2) correspond to its
                    top-left and bottom-right coordinates, and si is the score
                    of convolution. Please note that these coordinates are
                    in the input image im.
*/
Mat Detector::Detection(const Mat& im, HOGExtractor& ext, const float threshold) {

	/* TODO 4: 
       Obtain the HOG descriptor of the input im. And then convolute the linear
	   detector on the HOG descriptor. You may put the score of convolution in 
	   the structure "convolutionScore".
	*/
	Mat HOGDes = ext.ExtractHOG(im);
	Mat convolutionScore(HOGDes.size[0], HOGDes.size[1], CV_32FC1, Scalar(0));
	Mat channelScore(HOGDes.size[0], HOGDes.size[1], CV_32FC1, Scalar(0));
	// Begin TODO 4
	
	int numChannels = HOGDes.channels();
	Mat hogChannels[36];
	split(HOGDes, hogChannels);
	Mat detChannels[36];
	split(detector, detChannels);
	for (int i = 0; i < numChannels; ++i) {
		filter2D(hogChannels[i], channelScore, -1, detChannels[i]);
		convolutionScore += channelScore;
	}
	// End TODO 4

	/* TODO 5: 
       Select out those positions where the score is above the threshold. Here,
	   the positions are in ConvolutionScore. To output the coordinates of the
	   bounding boxes, please remember to recover the positions to those in the
	   input image. Please put the predicted bounding boxes and their scores in
	   the below structure "bbox".
	*/
	// Begin TODO 5
	int numDetection = 0;
	for (int i = 0; i < HOGDes.size[0]; ++i)
		for (int j = 0; j < HOGDes.size[1]; ++j)
			if (convolutionScore.at<float>(i, j) > threshold)
				++numDetection;
	Mat bbox(numDetection, 5, CV_32FC1, Scalar(0));
	numDetection = 0;
	int cy = detector.size[0] / 2, cx = detector.size[1] / 2;
	for (int i = 0; i < HOGDes.size[0]; ++i)
		for (int j = 0; j < HOGDes.size[1]; ++j)
			if (convolutionScore.at<float>(i, j) > threshold) {
				//cout<<i<<" "<<j<<endl;
				bbox.at<float>(numDetection, 0) = i * ext.GetCells() - cy * ext.GetCells();
				bbox.at<float>(numDetection, 1) = j * ext.GetCells() - cx * ext.GetCells();
				bbox.at<float>(numDetection, 2) = i * ext.GetCells() + (cy + 2) * ext.GetCells() - 1; 
				bbox.at<float>(numDetection, 3) = j * ext.GetCells() + (cx + 2) * ext.GetCells() - 1;
				bbox.at<float>(numDetection++, 4) = convolutionScore.at<float>(i, j);
			}
	
	
	// End TODO 5

	return bbox;
}


/* MultiscaleDetection is for multiscale detection
   Input:
           1. im: A grayscale image in height x width.
           2. ext: The pre-defined HOG extractor.
		   3. scales: The scales for resizeing the image.
		   4. numberOfScale: The number of different scales.
           5. threshold: The constant threshold to control the prediction.
   Output:
           1. bbox: The predicted bounding boxes in this format (n x 5 matrix):
                                    x11 y11 x12 y12 s1
                                    ... ... ... ... ...
                                    xi1 yi1 xi2 yi2 si
                                    ... ... ... ... ...
                                    xn1 yn1 xn2 yn2 sn
                    where n is the number of bounding boxes. For the ith 
                    bounding box, (xi1,yi1) and (xi2, yi2) correspond to its
                    top-left and bottom-right coordinates, and si is the score
                    of convolution. Please note that these coordinates are
                    in the input image im.
*/
Mat Detector::MultiscaleDetection(const Mat& im, HOGExtractor& ext, 
	const float* scales, const int numberOfScale, const float* threshold) {

	/* TODO 6: 
       You should firstly resize the input image by scales 
	   and store them in the structure pyra. 
	*/
	vector<Mat> pyra(numberOfScale);

	// Begin TODO 6
	for (int i = 0; i < numberOfScale; ++i) {
		int dh = round(scales[i] * im.size[0]), dw = round(scales[i] * im.size[1]);
		Mat dst(dh, dw, CV_32FC1, Scalar(0));
		resize(im, dst, dst.size(), 0, 0, cv::INTER_CUBIC);
		pyra.push_back(dst);
	}

	// End TODO 6
	

	/* TODO 7: 
       Perform detection with different scales. Please remember 
	   to transfer the coordinates of bounding box according to 
	   their scales. 
	   You should complete the helper-function  "Detection" and 
	   call it here. All the detected bounding boxes should be 
	   stored in the below structure "bbox".
	*/
	vector<Mat> bboxes;
	int totalDet = 0;
	// Begin TODO 7
	for (int i = 0; i < numberOfScale; ++i) {
		Mat scaleBBox = Detection(im, ext, threshold[i]);
		for (int j = 0; j < scaleBBox.size[0]; ++j) 
			for (int k = 0; k < 4; ++k)
				scaleBBox.at<float>(j, k) /= scales[i];
		bboxes.push_back(scaleBBox);
		totalDet += scaleBBox.size[0];
	}
	// End TODO 7
	Mat bbox(totalDet, 5, CV_32FC1, Scalar(0));
	totalDet = 0;
	for (int i = 0; i < numberOfScale; ++i) {
		for (int j = 0; j < bboxes[i].size[0]; ++j) {
			for (int k = 0; k < 5; ++k)
				bbox.at<float>(totalDet, k) = bboxes[i].at<float>(j, k);
			++totalDet;
		}
	}
	return bbox;
}

