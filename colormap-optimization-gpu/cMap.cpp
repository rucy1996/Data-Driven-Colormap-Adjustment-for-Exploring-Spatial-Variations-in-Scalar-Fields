#include "cMap.h"

cMap::cMap(const  cv::Mat& anchorPos, const cv::Mat& anchorColor) {

	//for (int i = 0; i < anchorPos.cols - 1; i++)//如果有n个数，就要连续排序n - 1 次
	//{
	//	for (int j = 0; j < anchorPos.cols - 1 - i; j++)//执行每一次比较的次数
	//	{
	//		if (anchorPos.at<float>(0,j) > anchorPos.at<float>( 0,j + 1))//如果这个数比前面的大
	//		{
	//			float temp = anchorPos.at<float>(0,j);
	//			anchorPos.at<float>(0,j) = anchorPos.at<float>(0,j+1);//把pos 进行交换
	//			anchorPos.at<float>(0,j+1) = temp;

	//			cv::Vec3f temp_color = anchorColor.at<cv::Vec3f>(0,j);
	//			anchorColor.at<cv::Vec3f>(0,j) = anchorColor.at<cv::Vec3f>(0,j+1);
	//			anchorColor.at<cv::Vec3f>(0,j+1) = temp_color;
	//		}
	//	}
	//}


	assert(anchorPos.size() == anchorColor.size());
	
	anchorPos.copyTo(mAnchorPos);
	anchorColor.copyTo(mAnchorColor);


	// Reshape anchorColor to 256 rows. 
	// This looks ugly, but I can't come up with a better way.
	// For CUDA version this is unnecessary, only to make a comparision with CPU version
	cv::Mat tmpPos, tmpColor;
	tmpPos.create(1, BIN_NUM + 1, CV_32FC1);
	tmpColor.create(1, BIN_NUM + 1, CV_32FC3);
	for (int i = 0; i <= BIN_NUM; i++) {
		float pos = (float)i / BIN_NUM;
		tmpPos.at<float>(i) = pos;
		tmpColor.at<cv::Vec3f>(i) = getColor(pos);
	}

	// Set cpu buffer
	mAnchorPos = tmpPos;
	mAnchorColor = tmpColor;
	computeAnchorLength();

	// Set gpu buffer
	mAnchorPosGpu.upload(tmpPos);
	mAnchorColorGpu.upload(tmpColor);
}


cMap::cMap() {

}
//cpu版本的获取颜色
cv::Mat cMap::getColor(const cv::Mat& pos) {

	cv::Mat colors(pos.size(), CV_32FC3);
	for (int i = 0; i < pos.rows; i++) {
		cv::Vec3f* pcolor = colors.ptr<cv::Vec3f>(i);
		const float* pi = pos.ptr<float>(i);
		for (int j = 0; j < pos.cols; j++) {
			pcolor[j] = getColor(pi[j]);
		}
	}

	return colors;

}

// Return binned color
cv::Mat cMap::getColor() {
	cv::Mat colors;
	for (int i = 0; i < BIN_NUM; i++) {
		colors.push_back(getColor((float)i / (BIN_NUM - 1)));
	}

	return colors;
}

cv::Vec3f cMap::getColor(float pos) {
	if (isnan(pos)) {
		return cMapBuffer::bgColor;
	}

	// Binary search in (m - 1) intervals
	int low = 0, high = mAnchorPos.cols - 2;
	while (low <= high) {
		int mid = low + ((high - low) >> 1); // /2

		float anchorPos0 = mAnchorPos.at<float>(mid);
		float anchorPos1 = mAnchorPos.at<float>(mid + 1);

		if (pos < anchorPos0) {
			high = mid - 1;
		}
		else if (pos > anchorPos1) {
			low = mid + 1;
		}
		else {
			float alpha = (pos - anchorPos0) / (anchorPos1 - anchorPos0);
			return mAnchorColor.at<cv::Vec3f>(mid) * (1 - alpha) + mAnchorColor.at<cv::Vec3f>(mid + 1) * alpha;
		}
	}

	// Could never get here
	/*assert(0);
	exit(-1);*/
	return cMapBuffer::bgColor;
}


cv::cuda::GpuMat cMap::getColor(const cv::cuda::GpuMat& posGpu) {
	return getColorGpu(mAnchorPosGpu, mAnchorColorGpu, posGpu);
	//return getColorGpu2(mAnchorColorGpu, posGpu);
}




cv::Mat cMap::arcLength(const cv::Mat& pos) const {
	cv::Mat lengths(pos.size(), CV_32FC1);
	for (int i = 0; i < pos.rows; i++) {
		float* plength = lengths.ptr<float>(i);
		const float* pi = pos.ptr<float>(i);
		for (int j = 0; j < pos.cols; j++) {
			plength[j] = arcLength(pi[j]);
		}
	}
	
	return lengths;
}

// Given position, compute its arc length to origin color
float cMap::arcLength(float pos) const {
	assert(pos >= 0 && pos <= 1);

	int i;
	for (i = 0; i < mAnchorPos.cols - 1; i++) {
		float anchorPos0 = mAnchorPos.at<float>(i);
		float anchorPos1 = mAnchorPos.at<float>(i + 1);

		if (pos >= anchorPos0 && pos < anchorPos1) {
			float alpha = (pos - anchorPos0) / (anchorPos1 - anchorPos0);
			cv::Vec3f color = mAnchorColor.at<cv::Vec3f>(i) * (1 - alpha) + mAnchorColor.at<cv::Vec3f>(i + 1) * alpha;

			return mAnchorLength.at<float>(i) + MyColor::CIE76(mAnchorColor.at<cv::Vec3f>(i), color);
		}
	}

	// pos == 1.f
	return mAnchorLength.at<float>(i);
}

// Returns arc length which use 1 / binNum as interval.
cv::Mat cMap::arcLength(const int binNum) const {

	cv::Mat binPos;
	for (int i = 0; i < binNum; i++) {
		binPos.push_back((float)i / binNum);
	}

	return arcLength(binPos);
}



// Compute arc length for each anchor
void cMap::computeAnchorLength() {//所以这里实际上取得就是256个点，来计算的累计弧长
	mAnchorLength = cv::Mat(mAnchorPos.size(), CV_32FC1, 0.f);
	//std::cout <<"mAnchorPos.size():"<< mAnchorPos.size() << std::endl;//这里是1x256的数组
	// For subsequent data points, calculate interval distances	
	for (int i = 1; i < mAnchorLength.cols; i++) {
		mAnchorLength.at<float>(i) = MyColor::CIE76(mAnchorColor.at<cv::Vec3f>(i - 1), mAnchorColor.at<cv::Vec3f>(i));
	}

	for (int i = 1; i < mAnchorLength.cols; i++) {
		mAnchorLength.at<float>(i) += mAnchorLength.at<float>(i - 1);
	}
	
}