#ifndef COLORMAP_GPU_H
#define COLORMAP_GPU_H

#include "cMapUtils.h"
#include "cuda_common.cuh"


namespace cMapBuffer {
	extern cv::Vec3f bgColor;
}

// A piecewise function to represent colormap
class cMap {

public:
	// Cpu data
	cv::Mat mAnchorPos;					
	cv::Mat mAnchorColor;
	cv::Mat mAnchorLength;

	// Gpu data
	cv::cuda::GpuMat mAnchorPosGpu;
	cv::cuda::GpuMat mAnchorColorGpu;



	cMap(const  cv::Mat& anchorPos, const cv::Mat& anchorColor);
	cMap();


	// Return binned color
	cv::Mat getColor();
	cv::Vec3f getColor(float pos);
	cv::cuda::GpuMat getColor(const cv::cuda::GpuMat& posGpu);
	cv::Mat getColor(const cv::Mat& pos);



	cv::Mat arcLength(const cv::Mat& pos) const;
	// Given position, compute its arc length to origin color
	float arcLength(float pos) const;
	// Returns arc length which use 1 / binNum as interval.
	cv::Mat arcLength(const int binNum) const;



private:
	// Compute arc length for each anchor
	void computeAnchorLength();
};


#endif
