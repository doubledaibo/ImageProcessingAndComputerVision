#include "HOGExtractor.h"

HOGExtractor::HOGExtractor(const int bins, const int cells, const int blocks) {
	this->bins = bins;
	this->cells = cells;
	this->blocks = blocks;
}


HOGExtractor::~HOGExtractor(void) {
}

int HOGExtractor::GetBins() {
	return this->bins;
}
int HOGExtractor::GetCells() {
	return this->cells;
}
int HOGExtractor::GetBlocks() {
	return this->blocks;
}


/* ExtractHOG is for extracting HOG descriptor of an image
   Input:
           im: A grayscale image in height x width.
   Output:
           HOGBlock: The HOG descriptor of the input image.
*/
Mat HOGExtractor::ExtractHOG(const Mat& im) {
	// Pad the im in order to make the height and width the multiplication of
    // the size of cells.
	int height = im.rows;
	int width = im.cols;
	int padHeight = height % cells == 0 ? 0 : (cells - height % cells);
    int padWidth = width % cells == 0 ? 0 : (cells - width % cells);
	Mat paddedIm(height+padHeight, width+padWidth, CV_32FC1, Scalar(0));
	Range imRanges[2];
	imRanges[0] = Range(0, height);
	imRanges[1] = Range(0, width);
	im.copyTo(paddedIm(imRanges));
	height = paddedIm.rows;
	width = paddedIm.cols;

	/* TODO 1: 
       Compute the horizontal and vertical gradients for each pixel. Put them 
	   in gradX and gradY respectively. In addition, compute the angles (using
	   atan2) and magnitudes by gradX and gradY, and put them in angle and 
	   magnitude. 
	*/
	Mat hx(1, 3, CV_32FC1, Scalar(0));
	hx.at<float>(0, 0) = -1;
	hx.at<float>(0, 1) = 0;
	hx.at<float>(0, 2) = 1;
	Mat hy = -hx.t();

	Mat gradX(height, width, CV_32FC1, Scalar(0));
	Mat gradY(height, width, CV_32FC1, Scalar(0));
	Mat angle(height, width, CV_32FC1, Scalar(0));
	Mat magnit(height, width, CV_32FC1, Scalar(0));
	float pi = 3.1416;
	
	// Begin TODO 1
	filter2D(paddedIm, gradX, -1, hx);
	filter2D(paddedIm, gradY, -1, hy);
	magnitude(gradX, gradY, magnit);
	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j)
			angle.at<float>(i, j) = atan2(gradY.at<float>(i, j), gradX.at<float>(i, j));

    // End TODO 1

	/* TODO 2:
	   Construct HOG for each cells, and put them in HOGCell. numberOfVerticalCell
	   and numberOfHorizontalCell are the numbers of cells in vertical and 
	   horizontal directions.
	   You should construct the histogram according to the bins. The bins range
	   from -pi to pi in this project, and the interval is given by
	   (2*pi)/bins.
	*/
	int numberOfVerticalCell = height / cells;
	int numberOfHorizontalCell = width / cells;
	Mat HOGCell(numberOfVerticalCell, numberOfHorizontalCell, 
		CV_32FC(bins), Scalar(0));
	float piInterval = 2 * pi / bins;
	// Begin TODO 2
	for (int i = 0; i < numberOfVerticalCell; ++i)
		for (int j = 0; j < numberOfHorizontalCell; ++j) 
			for (int p = i * cells; p < i * cells + cells; ++p)
				for (int q = j * cells; q < j * cells + cells; ++q) 
					HOGCell.at<Vec<float, 9>>(i, j)[min(bins - 1, int(floor((angle.at<float>(p, q) + pi) / piInterval)))] += magnit.at<float>(i, j);
	// End TODO 2

	/* TODO 3:
	   Concatenate HOGs of the cells within each blocks and normalize them.
	   The result should be stored in HOGBlock, where numberOfVerticalBlock and
	   numberOfHorizontalBlock are the number of blocks in vertical and
	   horizontal directions.
	*/
	int numberOfVerticalBlock = numberOfVerticalCell - 1;
	int numberOfHorizontalBlock = numberOfHorizontalCell - 1;
	Mat HOGBlock(numberOfVerticalBlock, numberOfHorizontalBlock, 
		CV_32FC(blocks*blocks*bins), Scalar(0));
	Mat Block(1, blocks * blocks * bins, CV_32FC1, Scalar(0));

	// Begin TODO 3
	for (int i = 0; i < numberOfVerticalBlock; ++i)
		for (int j = 0; j < numberOfHorizontalBlock; ++j) { 
			for (int k = 0; k < bins; ++k) {
				Block.at<float>(0, k) = HOGCell.at<Vec<float, 9>>(i, j)[k];
				Block.at<float>(0, k + bins) = HOGCell.at<Vec<float, 9>>(i, j + 1)[k];
				Block.at<float>(0, k + 2 * bins) = HOGCell.at<Vec<float, 9>>(i + 1, j)[k];
				Block.at<float>(0, k + 3 * bins) = HOGCell.at<Vec<float, 9>>(i + 1, j + 1)[k];
			}
			float sum = 0;
			for (int k = 0; k < blocks * blocks * bins; ++k) 
				sum += Block.at<float>(0,k);
			for (int k = 0; k < blocks * blocks * bins; ++k) 
				HOGBlock.at<Vec<float, 36>>(i, j)[k] = sum == 0 ? 0 : Block.at<float>(0, k) / sum;
		}
	// End TODO 3
	return HOGBlock;
}