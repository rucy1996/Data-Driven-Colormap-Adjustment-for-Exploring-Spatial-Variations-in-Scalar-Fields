#ifndef CONSTRAINT_H
#define CONSTRAINT_H

#include "cMap.h"
#include "cMapUtils.h"


// Parameters for optimzer
namespace cMapParam {
	// Parameters of objective function
	extern float alpha;
	extern float beta;
	extern float gamma;
	extern float eta;

	// Increasing rate of arc lengths
	extern float t1;
	extern float t2;

	// Weight magnifier
	extern int roiMag;


	// Size of input scalar field data
	extern int rows;
	extern int cols;

	// Number of variables
	extern int varNum;
	extern int anchorNum;
}


//Buffers for optimizer
namespace cMapBuffer {
	// Original information of colormap
	extern cMap oriColormap;
	extern float maxAnchorLength;

	extern cv::Mat oriAnchorColor;
	extern cv::Mat oriAnchorLengths;


	extern cv::cuda::GpuMat data;
	extern cv::cuda::GpuMat mask;

	// Weight for local difference
	extern cv::cuda::GpuMat diffWeight;
	// Weight for contrast
	extern cv::cuda::GpuMat contrastWeight;
	// luminance for background
	extern cv::cuda::GpuMat bgImageL;

	// Region of interests
	extern std::vector<cv::Point> roi;
	// Constant anchors
	extern std::vector<int> anchorFixed;
	// Customize background
	extern cv::Vec3f bgColorRgb;
	extern cv::Vec3f bgColor;
	extern cv::Mat bgImg;

	// Calculate gradient, laplacian and nearest background. etc.
	extern void initOptBuffer(const cMap& oriColormap,
		const cv::Mat& oriAnchorColor,
		const cv::Mat& oriAnchorLengths,
		const cv::Mat& data,
		const cv::Mat& mask,
		const cv::Mat& bgImageL);

}

// Colormap Optimzer
namespace cMapOpt {

	// Objective function: Boundary term
	float boundFunc(const cv::cuda::GpuMat& localDiff, 
		const cv::cuda::GpuMat& diffWeight, 
		const cv::cuda::GpuMat& mask);


	// Objective function: Boundary term_1
	float boundFunc_1(const cv::cuda::GpuMat& localDiff,
		const cv::cuda::GpuMat& diffWeight,
		const cv::cuda::GpuMat& mask, const cv::cuda::GpuMat& data);

	// Objective function: Contrast term
	float contrastFunc(const cv::cuda::GpuMat& labData, 
		const cv::cuda::GpuMat& contrastWeight, 
		const cv::cuda::GpuMat& mask);


	// Objective function: Fidelity term
	float colormapFunc(const cv::Mat& newAnchorLengths);


	// Evaluation function
	int KNEvalFC(KN_context_ptr             kc,
		CB_context_ptr             cb,
		KN_eval_request_ptr const  evalRequest,
		KN_eval_result_ptr  const  evalResult,
		void              * const  userParams);



	// Optimization with hard contraints
	void KNOpt(cv::Mat& anchorPos);

	// Optimization without hard contraints
	void KNOpt2(cv::Mat& anchorPos);


};



#endif