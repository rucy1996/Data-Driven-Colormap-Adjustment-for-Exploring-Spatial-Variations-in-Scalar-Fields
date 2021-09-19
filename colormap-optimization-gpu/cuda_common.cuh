// This file contains functions which uses cuda acceleration
// You can invoke them from inside c/c++ file.

#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <cvconfig.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_functions.h>
#include <sm_20_atomic_functions.h>
#include <device_functions.h>
#include <device_atomic_functions.h>

#include <stdio.h>
#include <time.h>
#include <sys/timeb.h>
#include <algorithm>

#include "stdafx.h"


// Error handler
#define CHECK(call)															\
{																			\
	const cudaError_t error = call;											\
	if (error != cudaSuccess)												\
	{																		\
		printf("Error: %s:%d ", __FILE__, __LINE__);							\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
		exit(1);															\
	}																		\
}


//// Error handler on device
//#define CHECK_DEVICE(call)															\
//{																			\
//	const cudaError_t error = call;											\
//	if (error != cudaSuccess)												\
//	{																		\
//		printf("Error: %s:%d ", __FILE__, __LINE__);							\
//		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
//	}																		\
//}

// CPU timer
inline double seconds() {
	struct timeb t1;
	ftime(&t1);

	return (t1.time*1.e+3 + t1.millitm) / 1.e+3;
}


inline __device__ float3 operator*(const float3& a, const float& b) {
	return make_float3(a.x*b, a.y*b, a.z*b);
}

inline __device__ float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ float CIE76(const float3& a, const float3& b) {
	return (a.x - b.x) * (a.x - b.x)
		+ (a.y - b.y) * (a.y - b.y)
		+ (a.z - b.z) * (a.z - b.z);
}


#ifndef M_PI
#define M_PI        3.14159265358979323846264338327950288   /* pi */
#endif

inline __device__ double deg2Rad(const double deg) {
	return (deg * (M_PI / 180.0));
}

inline __device__ double rad2Deg(const double rad) {
	return ((180.0 / M_PI) * rad);
}

inline __device__ double CIEDE2000(const float3&lab1, const float3&lab2) {
	/*
	* "For these and all other numerical/graphical 􏰀delta E00 values
	* reported in this article, we set the parametric weighting factors
	* to unity(i.e., k_L = k_C = k_H = 1.0)." (Page 27).
	*/
	const double k_L = 1.0, k_C = 1.0, k_H = 1.0;
	const double deg360InRad = deg2Rad(360.0);
	const double deg180InRad = deg2Rad(180.0);
	const double pow25To7 = 6103515625.0; /* pow(25, 7) */

										  /*
										  * Step 1
										  */
										  /* Equation 2 */
	double C1 = sqrt((lab1.y * lab1.y) + (lab1.z * lab1.z));
	double C2 = sqrt((lab2.y * lab2.y) + (lab2.z * lab2.z));
	/* Equation 3 */
	double barC = (C1 + C2) / 2.0;
	/* Equation 4 */
	double G = 0.5 * (1 - sqrt(pow(barC, 7) / (pow(barC, 7) + pow25To7)));
	/* Equation 5 */
	double a1Prime = (1.0 + G) * lab1.y;
	double a2Prime = (1.0 + G) * lab2.y;
	/* Equation 6 */
	double CPrime1 = sqrt((a1Prime * a1Prime) + (lab1.z * lab1.z));
	double CPrime2 = sqrt((a2Prime * a2Prime) + (lab2.z * lab2.z));
	/* Equation 7 */
	double hPrime1;
	if (lab1.z == 0 && a1Prime == 0)
		hPrime1 = 0.0;
	else {
		hPrime1 = atan2((double)lab1.z, a1Prime);
		/*
		* This must be converted to a hue angle in degrees between 0
		* and 360 by addition of 2􏰏 to negative hue angles.
		*/
		if (hPrime1 < 0)
			hPrime1 += deg360InRad;
	}
	double hPrime2;
	if (lab2.z == 0 && a2Prime == 0)
		hPrime2 = 0.0;
	else {
		hPrime2 = atan2((double)lab2.z, a2Prime);
		/*
		* This must be converted to a hue angle in degrees between 0
		* and 360 by addition of 2􏰏 to negative hue angles.
		*/
		if (hPrime2 < 0)
			hPrime2 += deg360InRad;
	}

	/*
	* Step 2
	*/
	/* Equation 8 */
	double deltaLPrime = lab2.x - lab1.x;
	/* Equation 9 */
	double deltaCPrime = CPrime2 - CPrime1;
	/* Equation 10 */
	double deltahPrime;
	double CPrimeProduct = CPrime1 * CPrime2;
	if (CPrimeProduct == 0)
		deltahPrime = 0;
	else {
		/* Avoid the fabs() call */
		deltahPrime = hPrime2 - hPrime1;
		if (deltahPrime < -deg180InRad)
			deltahPrime += deg360InRad;
		else if (deltahPrime > deg180InRad)
			deltahPrime -= deg360InRad;
	}
	/* Equation 11 */
	double deltaHPrime = 2.0 * sqrt(CPrimeProduct) *
		sin(deltahPrime / 2.0);

	/*
	* Step 3
	*/
	/* Equation 12 */
	double barLPrime = (lab1.x + lab2.x) / 2.0;
	/* Equation 13 */
	double barCPrime = (CPrime1 + CPrime2) / 2.0;
	/* Equation 14 */
	double barhPrime, hPrimeSum = hPrime1 + hPrime2;
	if (CPrime1 * CPrime2 == 0) {
		barhPrime = hPrimeSum;
	}
	else {
		if (fabs(hPrime1 - hPrime2) <= deg180InRad)
			barhPrime = hPrimeSum / 2.0;
		else {
			if (hPrimeSum < deg360InRad)
				barhPrime = (hPrimeSum + deg360InRad) / 2.0;
			else
				barhPrime = (hPrimeSum - deg360InRad) / 2.0;
		}
	}
	/* Equation 15 */
	double T = 1.0 - (0.17 * cos(barhPrime - deg2Rad(30.0))) +
		(0.24 * cos(2.0 * barhPrime)) +
		(0.32 * cos((3.0 * barhPrime) + deg2Rad(6.0))) -
		(0.20 * cos((4.0 * barhPrime) - deg2Rad(63.0)));
	/* Equation 16 */
	double deltaTheta = deg2Rad(30.0) *
		exp(-pow((barhPrime - deg2Rad(275.0)) / deg2Rad(25.0), 2.0));
	/* Equation 17 */
	double R_C = 2.0 * sqrt(pow(barCPrime, 7.0) /
		(pow(barCPrime, 7.0) + pow25To7));
	/* Equation 18 */
	double S_L = 1 + ((0.015 * pow(barLPrime - 50.0, 2.0)) /
		sqrt(20 + pow(barLPrime - 50.0, 2.0)));
	/* Equation 19 */
	double S_C = 1 + (0.045 * barCPrime);
	/* Equation 20 */
	double S_H = 1 + (0.015 * barCPrime * T);
	/* Equation 21 */
	double R_T = (-sin(2.0 * deltaTheta)) * R_C;

	/* Equation 22 */
	double deltaE = sqrt(
		pow(deltaLPrime / (k_L * S_L), 2.0) +
		pow(deltaCPrime / (k_C * S_C), 2.0) +
		pow(deltaHPrime / (k_H * S_H), 2.0) +
		(R_T * (deltaCPrime / (k_C * S_C)) * (deltaHPrime / (k_H * S_H))));

	return (deltaE);
}

extern "C"
void initCudaArgs(int rows, int cols, const cv::Vec3f& bgColor);

extern "C"
cv::cuda::GpuMat getLocalDiffGpu(const cv::cuda::GpuMat& labData, const cv::cuda::GpuMat& mask);

extern "C"
cv::cuda::GpuMat getLocalDiffGpu76(const cv::cuda::GpuMat& labData, const cv::cuda::GpuMat& mask);

extern "C"
cv::cuda::GpuMat getLocalDiffGpu2000(const cv::cuda::GpuMat& labData, const cv::cuda::GpuMat& mask);

extern "C"
cv::cuda::GpuMat getColorGpu(const cv::cuda::GpuMat& mAnchorPos, const cv::cuda::GpuMat& mAnchorColor, const cv::cuda::GpuMat& pos);

extern "C"
cv::cuda::GpuMat getColorGpu2(const cv::cuda::GpuMat& mAnchorColor, const cv::cuda::GpuMat& pos);

extern "C"
float contrastFuncGpu(const cv::cuda::GpuMat& labData,
	const cv::cuda::GpuMat& bgImageL,
	const cv::cuda::GpuMat& contrastWeight,
	const cv::cuda::GpuMat& mask);


