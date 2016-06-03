




#include "ImageRepresentor.h"


ImageRepresentator::ImageRepresentator(const vector<Point2f>& spmGrid, const int knn) {
	this->spmGrid = spmGrid;
	this->knn = knn;
	this->poolType = 0;
	this->codingType = 0;
}


ImageRepresentator::~ImageRepresentator(void) {
}


void ImageRepresentator::updateNNIds(const int knn, int id, float dist, vector<int>& nnIds, vector<float>& nnDists) {
	int curIdx = 0;
	while (curIdx < nnIds.size() && curIdx < knn) {
		if (nnDists[curIdx] > dist) {
			float tmp = nnDists[curIdx];
			nnDists[curIdx] = dist;
			dist = tmp;
			int t = nnIds[curIdx];
			nnIds[curIdx] = id;
			id = t;
		}
		++curIdx;
	}
	if (curIdx < knn) {
		nnIds.push_back(id);
		nnDists.push_back(dist);
	}
}

Mat ImageRepresentator::LLCCoding(const Mat& codebook, const Mat& featureDescriptor, const int knn, const float lambda) {
	int numC = codebook.rows;
	int lenC = codebook.cols;
	int numDescriptor = featureDescriptor.rows;
	Mat ll = Mat::eye(knn, knn, CV_32FC1);
	Mat coding(numDescriptor, numC, CV_32FC1, Scalar(0));
	vector<int> nnIds;
	vector<float> nnDists;
	Mat zT(knn, lenC, CV_32FC1, Scalar(0));
	Mat one = Mat::ones(knn, 1, CV_32FC1);
	Mat w(1, knn, CV_32FC1, Scalar(0));
	for (int i = 0; i < numDescriptor; ++i) {
		nnIds.clear();
		nnDists.clear();
		for (int j = 0; j < numC; ++j) {
			float dist = norm(featureDescriptor.row(i), codebook.row(j));
			updateNNIds(knn, j, dist, nnIds, nnDists);
		}
		Mat z(knn, lenC, CV_32FC1, Scalar(0));
		for (int j = 0; j < knn; ++j) 
			subtract(codebook.row(nnIds[j]), featureDescriptor.row(i), z.row(j));
		transpose(z, zT);
		Mat C = z * zT;
		float trC = trace(C)[0];
		C += ll * lambda * trC;
		solve(one, C, w, DECOMP_SVD);
		float sw = 0;
		for (int j = 0; j < knn; ++j)
			sw += w.at<float>(0, j);
		for (int j = 0; j < knn; ++j) 
			coding.at<float>(i, nnIds[j]) = w.at<float>(0, j) / sw;
	}
	return coding;
}

Mat ImageRepresentator::VQCoding(const Mat& codebook, const Mat& featureDescriptor) {
	int numC = codebook.rows;
	int numDescriptor = featureDescriptor.rows;
	Mat coding(numDescriptor, numC, CV_32FC1, Scalar(0));
	for (int i = 0; i < numDescriptor; ++i) {
		int nnId = 0;
		float nnDist = norm(featureDescriptor.row(i), codebook.row(0));
		for (int j = 1; j < numC; ++j) {
			float dist = norm(featureDescriptor.row(i), codebook.row(j));
			if (dist < nnDist) {
				nnId = j;
				nnDist = dist;
			}
		}
		coding.at<float>(i, nnId) = 1;
	}
	return coding;
}

Mat ImageRepresentator::sumPooling(const Mat& coding, vector<int> localIds, int numCell) {
	int lenP = coding.cols;
	Mat pooled(numCell, lenP, CV_32FC1, Scalar(0));
	for (int i = 0; i < coding.rows; ++i) 
		add(pooled.row(localIds[i]), coding.row(i), pooled.row(localIds[i]));
	// normalize
	for (int i = 0; i < numCell; ++i)
		normalize(pooled.row(i), pooled.row(i), 1, NORM_L1);
	return pooled;
}

Mat ImageRepresentator::gridMaxPooling(const Mat& coding, vector<int> localIds, int numCell) {
	int lenP = coding.cols;
	Mat pooled(numCell, lenP, CV_32FC1, Scalar(0));
	Mat row(1, lenP, CV_32FC1, Scalar(0));
	for (int i = 0; i < coding.rows; ++i) {
		max(pooled.row(localIds[i]), coding.row(i), row);
		row.copyTo(pooled.row(localIds[i]));
	}
	return pooled;
}

Mat ImageRepresentator::sqrtPooling(const Mat& coding, vector<int> localIds, int numCell) {
	Mat sp = sumPooling(coding, localIds, numCell);
	int lenP = coding.cols;
	Mat pooled(1, lenP, CV_32FC1, Scalar(0));
	for (int i = 0; i < numCell; ++i) 
		add(pooled, sp.row(i), pooled);
	pooled /= numCell;
	sqrt(pooled, pooled);
	return pooled;
}

Mat ImageRepresentator::maxPooling(const Mat& coding, vector<int> localIds, int numCell) {
	Mat sp = sumPooling(coding, localIds, numCell);
	int lenP = coding.cols;
	Mat pooled(1, lenP, CV_32FC1, Scalar(0));
	for (int i = 0; i < numCell; ++i) 
		max(pooled, sp.row(i), pooled);
	return pooled;
}

Mat ImageRepresentator::absolutePooling(const Mat& coding, vector<int> localIds, int numCell) {
	Mat sp = sumPooling(coding, localIds, numCell);
	int lenP = coding.cols;
	Mat pooled(1, lenP, CV_32FC1, Scalar(0));
	for (int i = 0; i < numCell; ++i) 
		add(pooled, abs(sp.row(i)), pooled);
	pooled /= numCell;
	return pooled;
}		

/* ImageRepresentator is for extracting image representation of an image
   Input:
           im: A grayscale image in height x width.
		   kps: The vector which has stored the key points.
		   featureDescriptor: The feature descriptors of the input image.
		   codebook: The codebook for encoding the feature descriptors.
   Output:
           spmFeat: The final representation of the input image.
*/
Mat ImageRepresentator::GetRepresentation(const Mat& im, const vector<KeyPoint>& kps,
	const Mat& featureDescriptor, const Mat& codebook) {


	/* TODO 3: 
       Apply spatial pyramid matching to group the descriptors according to their 
	   coordinates. The configuration of the local grids is stored in the member
	   variable "spmGrid" of this class.
	*/
	
	// coding
	Mat coding;
	
	switch (this->codingType) {
		case 0:
			coding = VQCoding(codebook, featureDescriptor);
			break;
		case 1:
			coding = LLCCoding(codebook, featureDescriptor, 5, 0.0001);
			break;
		default:
			coding = VQCoding(codebook, featureDescriptor);
	}

	float imgW = im.cols;
	float imgH = im.rows;
	int numDescriptor = featureDescriptor.rows;
	int numGrid = this->spmGrid.size();
	int totalGrid = 0;
	for (int i = 0; i < numGrid; ++i)
		totalGrid += this->spmGrid[i].x * this->spmGrid[i].y;
	Mat spmFeat;
	switch (this->poolType) {
		case 0:
		case 1:
		case 2:
			spmFeat = Mat(1, numGrid * coding.cols, CV_32FC1, Scalar(0));
			break;
		case 3:
		case 4:
			spmFeat = Mat(1, totalGrid * coding.cols, CV_32FC1, Scalar(0));
			break;
		default:
			spmFeat = Mat(1, numGrid * coding.cols, CV_32FC1, Scalar(0));
	}
	int offset = 0;
	for (int pId = 0; pId < numGrid; ++pId) {
		Point2f p = this->spmGrid[pId];
		float intervalX = imgW / p.x + 0.0001, intervalY = imgH / p.y + 0.0001;
		vector<int> localIds;
		for (int i = 0; i < numDescriptor; ++i) {
			int idX = (int)floor(kps[i].pt.x / intervalX), idY = (int)floor(kps[i].pt.y / intervalY);
			localIds.push_back(idX * p.y + idY);
		}
		// pooling
		Mat pooled;
		switch (this->poolType) {
			case 0:
				pooled = sqrtPooling(coding, localIds, p.x * p.y);
				break;
			case 1:
				pooled = maxPooling(coding, localIds, p.x * p.y);
				break;
			case 2:
				pooled = absolutePooling(coding, localIds, p.x * p.y);
				break;
			case 3:
				pooled = sumPooling(coding, localIds, p.x * p.y);
				break;
			case 4:
				pooled = gridMaxPooling(coding, localIds, p.x * p.y);
				break;
			default:
				pooled = sqrtPooling(coding, localIds, p.x * p.y);
		}
		for (int i = 0; i < pooled.rows; ++i) {
			for (int j = 0; j < pooled.cols; ++j) 
				spmFeat.at<float>(0, offset + j) = pooled.at<float>(i, j);
			offset += pooled.cols;
		}
	}	
	return spmFeat;
}


Mat ImageRepresentator::ComputeCode(const Mat& descriptor, const Mat& localBase) {
	Mat localBaseTrans = localBase.t();
	Mat code = ((localBase * localBaseTrans).inv()*localBase*(descriptor.t())).t();
	float sum = 1e-10;
	for (int i = 0 ; i < code.cols ; ++i) {
		sum += code.at<float>(0, i);	
	}
	for (int i = 0 ; i < code.cols ; ++i) {
		code.at<float>(0, i) = code.at<float>(0, i) / sum;
	}
	return code;
}