///////////////////////////////////////////////////////////////////////////
//
// NAME
//  FeatureAlign.h -- image registration using feature matching
//
// SEE ALSO
//  FeatureAlign.h      longer description
//
// Copyright ?Richard Szeliski, 2001.  See Copyright.h for more details
//
///////////////////////////////////////////////////////////////////////////

#include "ImageLib/ImageLib.h"
#include "FeatureAlign.h"
#include "P3Math.h"
#include <math.h>
#include <time.h>

/******************* TO DO *********************
 * alignImagePair:
 *	INPUT:
 *		f1, f2: source feature sets
 *		matches: correspondences between f1 and f2
 *		m: motion model
 *		f: focal length
 *		width: image width
 *		height: image height
 *		nRANSAC: number of RANSAC iterations
 *		RANSACthresh: RANSAC distance threshold
 *		M: transformation matrix (output)
 *	OUTPUT:
 *		repeat for nRANSAC iterations:
 *			choose a minimal set of feature matches
 *			estimate the transformation implied by these matches
 *			count the number of inliers
 *		for the transformation with the maximum number of inliers,
 *		compute the least squares motion estimate using the inliers,
 *		and store it in M
 */
int alignImagePair(const FeatureSet &f1, const FeatureSet &f2,
			  const vector<FeatureMatch> &matches, MotionModel m,
			  float f, int width, int height,
			  int nRANSAC, double RANSACthresh, CTransform3x3& M, vector<int> &inliers)
{
	// BEGIN TODO
	// write this entire method
	CTransform3x3 iterM;
	vector<int> iterInliers;
	int maxCount = 0, pair1 = 0, pair2 = 0;
	int iterCount = 0;
	inliers.clear();
	for (int i = 0; i < nRANSAC; ++i) {
			pair1 = rand() % f1.size();
			while (matches[pair1].id == 0) {
				pair1 = rand() % f1.size(); 
			}
			pair2 = rand() % f1.size();
			while (pair2 == pair1 || matches[pair2].id == 0) {
				pair2 = rand() % f1.size();
			}
			iterInliers.clear();
			iterInliers.push_back(pair1);
			iterInliers.push_back(pair2);
			leastSquaresFit(f1, f2, matches, m, f, width, height, iterInliers, iterM);
			iterCount = countInliers(f1, f2, matches, m, f, width, height, iterM, RANSACthresh, iterInliers);
			if (maxCount < iterCount) {
				maxCount = iterCount;
				inliers = iterInliers;
			}
	}

	leastSquaresFit(f1, f2, matches, m, f, width, height, inliers, M);

	// END TODO

	return 0;
}


float euDist(CVector3 p1, CVector3 p2) {
	float sum = 0;
	for (int i = 0; i < 3; ++i) 
		sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
	return sqrt(sum);
}

CVector3 getPoint(FeatureSet f1, int idx, float f, int width, int height) {
	CVector3 p;
	p[0] = f1[idx].x - 0.5f * width;
	p[1] = f1[idx].y - 0.5f * height;
	p[2] = f;
	return p;
}

/******************* TO DO *********************
 * countInliers:
 *	INPUT:
 *		f1, f2: source feature sets
 *		matches: correspondences between f1 and f2
 *		m: motion model
 *		f: focal length
 *		width: image width
 *		height: image height
 *		M: transformation matrix
 *		RANSACthresh: RANSAC distance threshold
 *		inliers: inlier feature IDs
 *	OUTPUT:
 *		transform the features in f1 by M
 *
 *		count the number of features in f1 for which the transformed
 *		feature is within Euclidean distance RANSACthresh of its match
 *		in f2
 *
 *		store these features IDs in inliers
 *
 *		this method should be similar to evaluateMatch from project 1,
 *		except you are comparing each distance to a threshold instead
 *		of averaging them
 */
int countInliers(const FeatureSet &f1, const FeatureSet &f2,
				 const vector<FeatureMatch> &matches, MotionModel m,
				 float f, int width, int height,
				 CTransform3x3 M, double RANSACthresh, vector<int> &inliers)
{
	inliers.clear();
	int count = 0;
	CVector3 p1;
	CVector3 p2;
	for (unsigned int i=0; i<f1.size(); i++) {
		// BEGIN TODO
		// determine if the ith feature in f1, when transformed by M,
		// is within RANSACthresh of its match in f2 (if one exists)
		//
		// if so, increment count and append i to inliers
		if (matches[i].id == 0) continue; 
		p1 = getPoint(f1, i, f, width, height);
		p2 = getPoint(f2, matches[i].id - 1, f, width, height);
		p1 = M * p1;
		if (euDist(p1, p2) <= RANSACthresh) {
			++count;
			inliers.push_back(i);
		}
		// END TODO
	}

	return count;
}

/******************* TO DO *********************
 * leastSquaresFit:
 *	INPUT:
 *		f1, f2: source feature sets
 *		matches: correspondences between f1 and f2
 *		m: motion model
 *		f: focal length
 *		width: image width
 *		height: image height
 *		inliers: inlier feature IDs
 *		M: transformation matrix (output)
 *	OUTPUT:
 *		compute the transformation from f1 to f2 using only the inliers
 *		and return it in M
 */
int leastSquaresFit(const FeatureSet &f1, const FeatureSet &f2,
					const vector<FeatureMatch> &matches, MotionModel m,
					float f, int width, int height,
					const vector<int> &inliers, CTransform3x3& M)
{
	// BEGIN TODO
	// write this entire method
	CTransform3x3 A;
	A = A * 0;
	CTransform3x3 U;
	CTransform3x3 V; 
	CTransform3x3 S;
	CVector3 p1;
	CVector3 p2;
	for (int i = 0; i < inliers.size(); ++i) {
		p1 = getPoint(f1, inliers[i], f, width, height);
		p2 = getPoint(f2, matches[inliers[i]].id - 1, f, width, height);
		A = A + p1 * p2;
	}


	svd(A, U, S, V);
	M = V * U.Transpose();

	// END TODO

	return 0;
}
