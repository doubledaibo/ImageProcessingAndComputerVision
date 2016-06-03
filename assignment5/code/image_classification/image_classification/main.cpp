#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "DenseFeatureExtractor.h"
#include "CodeBookGenerator.h"
#include "ImageRepresentor.h"

using namespace std;
using namespace cv;

int main() {
	/* Step 0: Configuration of parameters
	*/
	const int numCls = 5; // The number of classes and training/testing images per class
	const int trainImPerCls = 60;
	const int testImPerCls = 40;
	
	vector<Point2f> spmGrid; // The parameter for spatial pyramid matching
	spmGrid.push_back(Point2f(1, 1));
	spmGrid.push_back(Point2f(2, 2));
	spmGrid.push_back(Point2f(4, 4));

	const int codebookSize = 1024; // The size of the codebook
	const int knn = 5; // The k-nearest neighbors (for LLC only)

	CvSVMParams params; // The parameters of SVM
    params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;


	/* Step 1: Extract feature descriptors of all training samples
	*/
	cout << "Step 1: Extract feature descriptors of all training samples..." << endl;
	DenseFeatureExtractor denseExt;
	Mat samples;
	vector<Mat> trainSample;
	vector<vector<KeyPoint>> keypoints;
	Mat trainLabel;
	char clsname[5];
	char number[5];
	if (!fopen("..\\Representation\\trainRepresentation.yml", "rb")) {
		for (int cls = 0 ; cls < numCls ; ++cls) {
			for (int i = 0 ; i < trainImPerCls ; ++i) {
				cout << "SIFT descriptor: class " << cls+1 << " number " << i+1 << endl;
				sprintf(clsname, "%d", cls+1);
				sprintf(number, "%.4d", i+1);
				Mat im = imread("..\\Dataset\\Train\\"+string(clsname)+"\\image"+string(number)+".jpg", 
					CV_LOAD_IMAGE_GRAYSCALE);
				vector<KeyPoint> kps;
				Mat des = denseExt.ExtractFeature(im, kps);
				keypoints.push_back(kps);
				samples.push_back(des);
				trainSample.push_back(des);
				trainLabel.push_back(float(cls + 1));
			}
		}
		FileStorage fs("..\\Representation\\trainLabel.yml", FileStorage::WRITE);
		fs << "trainLabel" << trainLabel;
		fs.release();
	} else {
		FileStorage fs("..\\Representation\\trainLabel.yml", FileStorage::READ);
		fs["trainLabel"] >> trainLabel;
		fs.release();
	}
	cout << "done!" << endl;
	
	/* Step 2: Generate codebook
	*/
	cout << "Step 2: Generate codebook..." << endl;
	CodeBookGenerator codebookGet(codebookSize);
	Mat codebook;
	if (!fopen("..\\Codebook\\codebook.yml", "rb")) {
		codebook = codebookGet.GenerateCodebook(samples);
		FileStorage fs("..\\Codebook\\codebook.yml", FileStorage::WRITE);
		fs << "codebook" << codebook;
		fs.release();
	} else {
		FileStorage fs("..\\Codebook\\codebook.yml", FileStorage::READ);
		fs["codebook"] >> codebook;
		fs.release();
	}
	cout << "done!" << endl;

	/* Step 3: Apply SPM and encode the feature descriptors, and then exrtact the image representation
	*/
	cout << "Step 3: Apply SPM and encode the feature descriptors, and then exrtact the image representation..." << endl;
	ImageRepresentator imageRep(spmGrid, knn);
	imageRep.codingType = 1;
	/* coding method:
		0: VQ coding
		1: LLC coding
	*/
	imageRep.poolType = 4; 
	/* pooling method: (c: length of the coding vector of one descriptor; g: number of grids in a configuration)
		0: sqrt-pooling (c x 1 vector)
		1: max-pooling (local grids collapsed) (c x 1 vector)
		2: absolute-pooling (c x 1 vector)
		3: sum-pooling (cg x 1 vector) 
		4: max-pooling (local grids kept) (cg x 1 vector)
	*/
	Mat trainRepresentation;
	if (!fopen("..\\Representation\\trainRepresentation.yml", "rb")) {
		for (int cls = 0 ; cls < numCls ; ++cls) {
			for (int i = 0 ; i < trainImPerCls ; ++i) {
				Mat des = trainSample[cls*60+i];
				vector<KeyPoint> kps = keypoints[cls*60+i];
				cout << "Image representation: class " << cls+1 << " number " << i+1 << endl;
				sprintf(clsname, "%d", cls+1);
				sprintf(number, "%.4d", i+1);
				Mat im = imread("..\\Dataset\\Train\\"+string(clsname)+"\\image"+string(number)+".jpg", 
					CV_LOAD_IMAGE_GRAYSCALE);
				Mat imRep = imageRep.GetRepresentation(im, kps, des, codebook);
				trainRepresentation.push_back(imRep);
			}
		}
		FileStorage fs("..\\Representation\\trainRepresentation.yml", FileStorage::WRITE);
		fs << "trainRepresentation" << trainRepresentation;
		fs.release();
	} else {
		FileStorage fs("..\\Representation\\trainRepresentation.yml", FileStorage::READ);
		fs["trainRepresentation"] >> trainRepresentation;
		fs.release();
	}
	cout << "done!" << endl;

	/* Step 4: Train linear classifier with SVM
	*/
	cout << "Step 4: Train linear classifier with SVM..." << endl;
	CvSVM classifier;
	if (!fopen("..\\Classifier\\classifier.yml", "rb")) {
		classifier.train_auto(trainRepresentation, trainLabel, Mat(), Mat(), params);
		classifier.save("..\\Classifier\\classifier.yml");
	} else {
		classifier.load("..\\Classifier\\classifier.yml");
	}
	cout << "done!" << endl;


	/* Step 5: Evaluate the classification accuracy
	*/
	cout << "Step 5: Evaluate the classification accuracy..." << endl;
	float hit = 0;
	for (int cls = 0 ; cls < numCls ; ++cls) {
		for (int i = 0 ; i < testImPerCls ; ++i) {
			cout << "Evaluate image: class " << cls+1 << " number " << i+61 << "...";
			sprintf(clsname, "%d", cls+1);
			sprintf(number, "%.4d", i+61);
			Mat im = imread("..\\Dataset\\Test\\"+string(clsname)+"\\image"+string(number)+".jpg", 
				CV_LOAD_IMAGE_GRAYSCALE);
			vector<KeyPoint> kps;
			Mat des = denseExt.ExtractFeature(im, kps);
			Mat imRep = imageRep.GetRepresentation(im, kps, des, codebook);
			int pre = classifier.predict(imRep);
			if (pre == cls+1) {
				++hit;
				cout << "hit";
			} else {
				cout << "miss";
			}
			cout << endl;
		}
	}
	cout << "Classification accuracy: " << hit/(float)(numCls*testImPerCls)<< endl;
	system("pause");
	return 0;
}
