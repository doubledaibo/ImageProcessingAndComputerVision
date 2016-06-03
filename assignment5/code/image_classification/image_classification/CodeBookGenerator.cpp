#include "CodeBookGenerator.h"


CodeBookGenerator::CodeBookGenerator(const int codebookSize) {
	this->codebookSize = codebookSize;
}


CodeBookGenerator::~CodeBookGenerator(void) {
}


/* GenerateCodebook is for generating the codebook with the feature descriptors
   Input:
           samples: The feature descriptors.
   Output:
           codebook: The codebook generated. It is used for encoding feature 
					 descriptors.
*/
Mat CodeBookGenerator::GenerateCodebook(const Mat& descriptors) {
	int sampleNum = descriptors.rows;
	int dims = descriptors.cols;
	Mat labels(sampleNum, 1, CV_32FC1);
	Mat codebook(this->codebookSize, dims, CV_32FC1);

	/* TODO 2: 
       Use kmeans or fisher vector encoding to generate the codebook
	*/
	kmeans(descriptors, this->codebookSize, labels, TermCriteria(TermCriteria::MAX_ITER, 100, 1e-5), 3, KMEANS_RANDOM_CENTERS, codebook);

	// End TODO 2

	return codebook;
}