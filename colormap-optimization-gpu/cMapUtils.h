#ifndef UTILS_H
#define UTILS_H

#include "stdafx.h"
#include "mycolor.h"

#include <time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <io.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/softfloat.hpp>

#include <knitro.h>

namespace cMapUtils {
	//-------------------------------------------------------------------
	// API for loading file
	//-------------------------------------------------------------------

	/** @brief		Load mat from txt file
	* @param path	File path
	* @param MA_Out	Matrix to be load
	* @param chns	Channels specifed
	* @return		0 if succeed. Else -1
	*/
	int Load_Txt_File(const std::string& path,
		cv::Mat& MA_Out, int chns = 1);

	/** @brief			Load video key frames from file folder
	* @param path		File folder path
	* @param MA_Outs	vector of Matrix to be load
	* @param chns		Channels specifed
	* @return			0 if succeed. Else -1
	*/
	 int Load_Txt_Folder(const std::string& path,
		std::vector<cv::Mat>& MA_Outs, int chns = 1);

	 void saveVecToFIle(const std::string& path, const float* vec, int len);

	 void saveMatTotxt(const std::string& path, cv::Mat& data);
	//-------------------------------------------------------------------
	// API for computation
	//-------------------------------------------------------------------

	// Normalize according to negative half
	 void normalizeSymmetric(cv::Mat& data, 
		const cv::Mat& mask);

	// Calculate boundary likehood for data.
	 cv::Mat getDiffWeight(const cv::Mat& data, 
		const cv::Mat& dataGrad, 
		const cv::Mat& dataLap, 
		const cv::Mat& mask, 
		const std::vector<cv::Point>& roi = {});

	// For each bin, calculate mean value of targets which falls into it.
	 cv::Mat valueAverage(const cv::Mat& values, 
		const std::vector<cv::Mat>& binIdx);

	 //求比例
	 cv::Mat Ratio(const std::vector<cv::Mat>& binIdx, int bins, float threshould);
	 //求每一个bins的均值梯度
	 cv::Mat Ratio_average(const cv::Mat& values, const std::vector<cv::Mat>& binIdx, int bins);


	 std::vector<cv::Mat> getHist(const cv::Mat& data, 
		const cv::Mat& mask);


	 cv::Mat getContrastBinWeight(const std::vector<cv::Mat>& hist, 
		const cv::Mat& mask);
	 float nearestDistance2nan(const cv::Point& targetPosition, 
		const cv::Mat& mask);

	 

	 void getContrastWeight(const std::vector<cv::Mat>& hist, 
		const cv::Mat& mask, 
		cv::Mat& contrastWeight, 
		cv::Mat& contrastPosition);
	 void nearestDistance2nan(const cv::Point& targetPosition, 
		const cv::Mat& mask, 
		float* distance, 
		cv::Point* position);





	// Initialze anchor position, end points included.
	 cv::Mat initAnchorPos(float anchor0, 
		float anchor1, 
		int num);


	 void insertRow(cv::Mat& matrix, 
		const cv::Mat& row, 
		int nrow);


	// Get image gradient and laplacian. This function is extremely slow because of abuse of at().
	// Please don't use it in your objective function.
	 void getDvts(const cv::Mat& data, 
		const cv::Mat& mask, 
		cv::Mat& dataGrad, 
		cv::Mat& dataLap);


	//-------------------------------------------------------------------
	// API for conversion between opencv and array.
	//-------------------------------------------------------------------

	/* Convert an opencv mat to array. arr must be pre-allocated */
	 void matrix2array1d(const cv::Mat& cvMat, 
		double* arr, 
		int offset = 0);
	/* Convert an array to opencv mat. Mat must be pre-allocated. */
	 void array1d2matrix(const double* arr, 
		cv::Mat& cvMat, 
		int len, 
		int offset = 0);


	//-------------------------------------------------------------------
	// API for display
	//-------------------------------------------------------------------

	 void dispImage(const std::string& windowName, 
		const cv::cuda::GpuMat& image, 
		bool isGray);
	 void dispImage(const std::string& windowName, 
		const cv::Mat& image, 
		bool isGray);


	 void saveImage(const std::string& imagePath, 
		const cv::cuda::GpuMat& image, 
		bool isGray);
	 void saveImage(const std::string& imagePath, 
		const cv::Mat& image, 
		bool isGray);


	 void dispColormap(const std::string& windowName, 
		const cv::Mat& anchorColor);

	// Convert a gray image to rgb
	 cv::Mat gray2bgr(const cv::Mat& grayImage, 
		const cv::Mat& mask);

	// Overlay one image on the other with the same size.
	 cv::Mat overlay(const cv::Mat& srcTop, 
		const cv::Mat& srcBottom, 
		const cv::Mat& mask);


	 //cpu版本计算localdiff
	  cv::Mat getLocalDiff( cv::Mat& labData, const cv::Mat& mask);



	 //对原始数据进行直方图统计
	 static cv::Mat getHist(const cv::Mat& data, float bin_num) {
		 cv::Mat binData;
		 data.convertTo(binData, CV_32SC1, bin_num - 1);

		 float histSize = 0;
		 cv::Mat hist(bin_num, 1, CV_32FC1, 0.f);

		 for (int i = 0; i < bin_num; i++) {
			 hist.at<float>(i, 0) = cv::countNonZero(binData == i);
			 histSize += hist.at<float>(i, 0);
		 }
		 for (int i = 0; i < bin_num; i++) {
			 hist.at<float>(i, 0) /= histSize;
		 }
		 // Accum
		 for (int i = 1; i < bin_num; i++) {
			 hist.at<float>(i, 0) += hist.at<float>(i - 1, 0);
		 }
		 return hist;
	 }
	 //find pos
	 static	 cv::Mat get_histpos(const cv::Mat& AnchorPos, const cv::Mat& stored_hist) {
		 cv::Mat Hist_pos(AnchorPos.rows, 1, CV_32FC1, 0.f);//用原始直方图计算出来的新的位置
		 Hist_pos.at<float>(0, 0) = 0;
		 Hist_pos.at<float>(AnchorPos.rows - 1, 0) = 1;
		 for (int i = 1; i < AnchorPos.rows - 1; i++)
		 {
			 float value = AnchorPos.at<float>(i, 0);
			 // Binary search in (m - 1) intervals
			 //折半查找在（m-1)个区间
			 int low = 0, high = AnchorPos.rows - 2;
			 while (low <= high) {
				 int mid = (high + low) / 2; // 折半
				 float anchorPos0 = stored_hist.at<float>(mid);
				 float anchorPos1 = stored_hist.at<float>(mid + 1);
				 if (value < anchorPos0) {
					 high = mid - 1;
				 }
				 else if (value > anchorPos1) {
					 low = mid + 1;
				 }
				 else {//进行一维线性插值
					 float pos = (mid + (value - anchorPos0) / (anchorPos1 - anchorPos0)) / (AnchorPos.rows - 1);
					 Hist_pos.at<float>(i, 0) = pos;
					 break;
				 }
			 }
		 }
		 return Hist_pos;
	 }

};


#endif // !UTILS_H