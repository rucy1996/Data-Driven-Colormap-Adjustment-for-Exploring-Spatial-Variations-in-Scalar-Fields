#include "cuda_common.cuh"

// Constant buffers
__constant__ int __rows__;
__constant__ int __cols__;
__constant__ float3 __bgcolor__;



// __constant__ cannot be written in kernels, but OKAY in __host__
// Access efficiency：register > shared > constant >local > device
void initCudaArgs(const int rows, const int cols, const cv::Vec3f& bgColor) {
	CHECK(cudaMemcpyToSymbol(__rows__, &rows, sizeof(int)));
	CHECK(cudaMemcpyToSymbol(__cols__, &cols, sizeof(int)));
	CHECK(cudaMemcpyToSymbol(__bgcolor__, &bgColor, sizeof(float3)));
}

// Make replicate border for matrix. dst must be pre-allocated.
template <typename T>
__global__ void makeBorderKernel(cv::cuda::PtrStep<T> src, cv::cuda::PtrStep<T> dst, cv::BorderTypes type) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (j < __rows__ && i < __cols__) {
		T border = src(j, i);

		if (0 == j) {					// First row
			dst(j, i + 1) = border;
		}
		else if (__rows__ - 1 == j) {	// Last row
			dst(j + 2, i + 1) = border;
		}

		if (0 == i) {					// First col
			dst(j + 1, i) = border;
		}
		else if (__cols__ - 1 == i) {	// Last col
			dst(j + 1, i + 2) = border;
		}
		dst(j + 1, i + 1) = border;
	}
}

// Gradient
__global__ void localDiffKernel(cv::cuda::PtrStep<float3> labDataWithBorder, cv::cuda::PtrStep<uchar> maskWithBorder ,cv::cuda::PtrStep<float> localDiff) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (j < __rows__ && i < __cols__) {
		if (maskWithBorder(j + 1, i + 1)) {
			float3 c = labDataWithBorder(j + 1, i + 1);

			float3 xl, xr, yl, yr;
			
			xl = c;
			xr = c;
			yl = c;
			yr = c;
			

			if (maskWithBorder(j + 1, i)) {
				xl = labDataWithBorder(j + 1, i);
			}
			if (maskWithBorder(j + 1, i + 2)) {
				xr = labDataWithBorder(j + 1, i + 2);
			}
			if (maskWithBorder(j, i + 1)) {
				yl = labDataWithBorder(j, i + 1);
			}
			if (maskWithBorder(j + 2, i + 1)) {
				yr = labDataWithBorder(j + 2, i + 1);
			}


			//两个斜线上的4个点
			float3 xy1, xy2, xy3, xy4;
			xy1 = c;
			xy2 = c;
			xy3 = c;
			xy4 = c;
			if (maskWithBorder(j + 2, i)) {
				xy1 = labDataWithBorder(j + 2, i);
			}
			if (maskWithBorder(j + 2, i + 2)) {
				xy2 = labDataWithBorder(j + 2, i + 2);
			}
			if (maskWithBorder(j, i )) {
				xy3 = labDataWithBorder(j, i );
			}
			if (maskWithBorder(j , i+2 )) {
				xy4 = labDataWithBorder(j , i +2);
			}



			//float3 gx = xl * (-0.5f) + xr * (0.5f);
			//float3 gy = yl * (-0.5f) + yr * (0.5f);

			//// Write into localDiff
			//float meanx = (gx.x + gx.y + gx.z) / 3;
			//float meany = (gy.x + gy.y + gy.z) / 3;
			//localDiff(j, i) = sqrt(meanx*meanx + meany*meany);

			float g1 = sqrt((xl.x - c.x)*(xl.x - c.x) + (xl.y- c.y)*(xl.y - c.y) + (xl.z - c.z)*(xl.z - c.z));
			float g2 = sqrt((xr.x - c.x)*(xr.x - c.x) + (xr.y - c.y)*(xr.y - c.y) + (xr.z - c.z)*(xr.z - c.z));
			float g3 = sqrt((yl.x - c.x)*(yl.x - c.x) + (yl.y - c.y)*(yl.y - c.y) + (yl.z - c.z)*(yl.z - c.z));
			float g4 = sqrt((yr.x - c.x)*(yr.x - c.x) + (yr.y - c.y)*(yr.y - c.y) + (yr.z - c.z)*(yr.z - c.z));

			float g5 = sqrt((xy1.x - c.x)*(xy1.x - c.x) + (xy1.y - c.y)*(xy1.y - c.y) + (xy1.z - c.z)*(xy1.z - c.z));
			float g6 = sqrt((xy2.x - c.x)*(xy2.x - c.x) + (xy2.y - c.y)*(xy2.y - c.y) + (xy2.z - c.z)*(xy2.z - c.z));
			float g7 = sqrt((xy3.x - c.x)*(xy3.x - c.x) + (xy3.y - c.y)*(xy3.y - c.y) + (xy3.z - c.z)*(xy3.z - c.z));
			float g8 = sqrt((xy4.x - c.x)*(xy4.x - c.x) + (xy4.y - c.y)*(xy4.y - c.y) + (xy4.z - c.z)*(xy4.z - c.z));

			localDiff(j, i) = (g1+g2+g3+g4+g5+g6+g7+g8)/8;
			//localDiff(j, i) = (g1 + g2 + g3 + g4 ) / 4;
		}
		else {
			localDiff(j, i) = NAN;
		}
	}
}

cv::cuda::GpuMat getLocalDiffGpu(const cv::cuda::GpuMat& labData, const cv::cuda::GpuMat& mask) 
{
	//std::chrono::steady_clock::time_point begin;
	//std::chrono::steady_clock::time_point end;

	dim3 threadsPerBlock(16, 16);
	uint block_num_vertical = (labData.rows + threadsPerBlock.x - 1) / threadsPerBlock.x;
	uint block_num_horizontal = (labData.cols + threadsPerBlock.y - 1) / threadsPerBlock.y;
	dim3 numBlocks(block_num_horizontal,
		block_num_vertical);



	cv::cuda::GpuMat labDataWithBorder;
	cv::cuda::copyMakeBorder(labData, labDataWithBorder, 1, 1, 1, 1, cv::BORDER_REPLICATE);

	cv::cuda::GpuMat maskWithBorder;
	cv::cuda::copyMakeBorder(mask, maskWithBorder, 1, 1, 1, 1, cv::BORDER_REPLICATE);

	//begin = std::chrono::steady_clock::now();

	//cv::cuda::GpuMat labDataWithBorder(labData.rows + 2, labData.cols + 2, CV_32FC3);
	//makeBorderKernel<float3> << <numBlocks, threadsPerBlock >> > (labData, labDataWithBorder, cv::BORDER_REPLICATE);

	//cv::cuda::GpuMat maskWithBorder(mask.rows + 2, mask.cols + 2, CV_8UC1);
	//makeBorderKernel<uchar> << <numBlocks, threadsPerBlock >> > (mask, maskWithBorder, cv::BORDER_REPLICATE);


	//end = std::chrono::steady_clock::now();
	//std::cout << "copyMakeBorder " << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() << std::endl;


	//begin = std::chrono::steady_clock::now();


	cv::cuda::GpuMat localDiff(labData.size(), CV_32FC1);
	localDiffKernel <<<numBlocks, threadsPerBlock >>> (labDataWithBorder, maskWithBorder, localDiff);
	CHECK(cudaDeviceSynchronize());

	//end = std::chrono::steady_clock::now();
	//std::cout << "localDiffKernel " << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() << std::endl;



	//begin = std::chrono::steady_clock::now();

	cv::cuda::GpuMat normLocalDiff;
	
	cv::cuda::normalize(localDiff, normLocalDiff,0,20, cv::NORM_MINMAX, -1, mask);// Will lead unmasked region to 0.
	
	//end = std::chrono::steady_clock::now();
	//std::cout << "normalize " << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() << std::endl;

#ifdef GRAY2COLOR
	/*cv::cuda::GpuMat rMask;
	cv::cuda::subtract(255, mask, rMask);
	normLocalDiff.setTo(NAN, rMask);*/
#endif // GRAY2COLOR


#ifdef LOCAL_DIFF_NORM
	return normLocalDiff;
	
	/*if (gradMean > grads_threshold)
	{
		return normLocalDiff;
	}
	else
	{
		return localDiff;
	}*/
	
#else
	return localDiff;
#endif 

}

// Neighbor CIE76 distance
__global__ void localDiffKernel76(cv::cuda::PtrStep<float3> labDataWithBorder, cv::cuda::PtrStep<uchar> maskWithBorder, cv::cuda::PtrStep<float> localDiff) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (j < __rows__ && i < __cols__) {
		if (maskWithBorder(j + 1, i + 1)) {
			float3 c = labDataWithBorder(j + 1, i + 1);

			float3 xl, xr, yl, yr;
			xl = c;
			xr = c;
			yl = c;
			yr = c;


			if (maskWithBorder(j + 1, i)) {
				xl = labDataWithBorder(j + 1, i);
			}
			if (maskWithBorder(j + 1, i + 2)) {
				xr = labDataWithBorder(j + 1, i + 2);
			}
			if (maskWithBorder(j, i + 1)) {
				yl = labDataWithBorder(j, i + 1);
			}
			if (maskWithBorder(j + 2, i + 1)) {
				yr = labDataWithBorder(j + 2, i + 1);
			}

			localDiff(j, i) = CIE76(c, xl) + CIE76(c, xr) + CIE76(c, yl) + CIE76(c, yr);
		}
		else {
			localDiff(j, i) = NAN;
		}
	}
}

cv::cuda::GpuMat getLocalDiffGpu76(const cv::cuda::GpuMat& labData, const cv::cuda::GpuMat& mask) {
	cv::cuda::GpuMat labDataWithBorder;
	cv::cuda::copyMakeBorder(labData, labDataWithBorder, 1, 1, 1, 1, cv::BORDER_REPLICATE);


	cv::cuda::GpuMat maskWithBorder;
	cv::cuda::copyMakeBorder(mask, maskWithBorder, 1, 1, 1, 1, cv::BORDER_REPLICATE);


	dim3 threadsPerBlock(16, 16);
	uint block_num_vertical = (labData.rows + threadsPerBlock.x - 1) / threadsPerBlock.x;
	uint block_num_horizontal = (labData.cols + threadsPerBlock.y - 1) / threadsPerBlock.y;
	dim3 numBlocks(block_num_horizontal,
		block_num_vertical);


	cv::cuda::GpuMat localDiff(labData.size(), CV_32FC1);
	localDiffKernel76 << <numBlocks, threadsPerBlock >> > (labDataWithBorder, maskWithBorder, localDiff);
	CHECK(cudaDeviceSynchronize());


	cv::cuda::GpuMat normLocalDiff;
	cv::cuda::normalize(localDiff, normLocalDiff, 0, 1, cv::NORM_MINMAX, -1, mask);

	return normLocalDiff;
}

// Neighbor CIEDE2000 distance
__global__ void localDiffKernel2000(cv::cuda::PtrStep<float3> labDataWithBorder, cv::cuda::PtrStep<uchar> maskWithBorder, cv::cuda::PtrStep<float> localDiff) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (j < __rows__ && i < __cols__) {
		if (maskWithBorder(j + 1, i + 1)) {
			float3 c = labDataWithBorder(j + 1, i + 1);

			float3 xl, xr, yl, yr;
			xl = c;
			xr = c;
			yl = c;
			yr = c;


			if (maskWithBorder(j + 1, i)) {
				xl = labDataWithBorder(j + 1, i);
			}
			if (maskWithBorder(j + 1, i + 2)) {
				xr = labDataWithBorder(j + 1, i + 2);
			}
			if (maskWithBorder(j, i + 1)) {
				yl = labDataWithBorder(j, i + 1);
			}
			if (maskWithBorder(j + 2, i + 1)) {
				yr = labDataWithBorder(j + 2, i + 1);
			}

			localDiff(j, i) = CIEDE2000(c, xl) + CIEDE2000(c, xr) + CIEDE2000(c, yl) + CIEDE2000(c, yr);
		}
		else {
			localDiff(j, i) = NAN;
		}
	}
}

cv::cuda::GpuMat getLocalDiffGpu2000(const cv::cuda::GpuMat& labData, const cv::cuda::GpuMat& mask) {
	cv::cuda::GpuMat labDataWithBorder;
	cv::cuda::copyMakeBorder(labData, labDataWithBorder, 1, 1, 1, 1, cv::BORDER_REPLICATE);


	cv::cuda::GpuMat maskWithBorder;
	cv::cuda::copyMakeBorder(mask, maskWithBorder, 1, 1, 1, 1, cv::BORDER_REPLICATE);


	dim3 threadsPerBlock(16, 16);
	uint block_num_vertical = (labData.rows + threadsPerBlock.x - 1) / threadsPerBlock.x;
	uint block_num_horizontal = (labData.cols + threadsPerBlock.y - 1) / threadsPerBlock.y;
	dim3 numBlocks(block_num_horizontal, 
		block_num_vertical); 


	cv::cuda::GpuMat localDiff(labData.size(), CV_32FC1);
	localDiffKernel2000 << <numBlocks, threadsPerBlock >> > (labDataWithBorder, maskWithBorder, localDiff);
	CHECK(cudaDeviceSynchronize());


	cv::cuda::GpuMat normLocalDiff;
	cv::cuda::normalize(localDiff, normLocalDiff, 0, 1, cv::NORM_MINMAX, -1, mask);

	return normLocalDiff;
}



__device__ inline float m_abs(float n) {
	if (n < 0) {
		return -n;
	}
	return n;
}

__global__ void contrastKernel(cv::cuda::PtrStep<float3> labData, 
	cv::cuda::PtrStep<float> bgImageL, 
	cv::cuda::PtrStep<float> contrastWeight, 
	cv::cuda::PtrStep<float> dst) {

	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (j < __rows__ && i < __cols__) {
		float contrast = m_abs(labData(j, i).x - bgImageL(j, i));
		dst(j, i) = contrastWeight(j, i) * contrast;
	}
}


float contrastFuncGpu(const cv::cuda::GpuMat& labData,
	const cv::cuda::GpuMat& bgImageL,
	const cv::cuda::GpuMat& contrastWeight,
	const cv::cuda::GpuMat& mask){

	dim3 threadsPerBlock(16, 16);
	uint block_num_vertical = (labData.rows + threadsPerBlock.x - 1) / threadsPerBlock.x;
	uint block_num_horizontal = (labData.cols + threadsPerBlock.y - 1) / threadsPerBlock.y;
	dim3 numBlocks(block_num_horizontal,
		block_num_vertical);

	cv::cuda::GpuMat dst(labData.size(), CV_32FC1);
	contrastKernel << < numBlocks, threadsPerBlock >> > (labData, bgImageL, contrastWeight, dst);
	CHECK(cudaDeviceSynchronize());


	float E = cv::cuda::sum(dst, mask)[0];
	return -E / (labData.rows * labData.cols);
}






// Binary search
__global__ void getColorKernel(cv::cuda::PtrStep<float> mAnchorPos, cv::cuda::PtrStep<float3> mAnchorColor, cv::cuda::PtrStep<float> pos, cv::cuda::PtrStep<float3> color) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (j < __rows__ && i < __cols__) {
		float pos_ = pos(j, i);
		if (isnan(pos_)) {
			color(j, i) = __bgcolor__;
			/*float3 a;
			a.x = 100; a.y = 0; a.z = 0;
			color(j, i) = a;*/
			return;
		}


		// Binary search in (m - 1) intervals
		int low = 0, high = BIN_NUM - 1;//mAnchorPos.cols - 2;
		while (low <= high) {
			int mid = low + ((high - low) >> 1); // /2

			float anchorPos0 = mAnchorPos(0, mid);
			float anchorPos1 = mAnchorPos(0, mid + 1);

			if (pos_ < anchorPos0) {
				high = mid - 1;
			}
			else if (pos_ > anchorPos1) {
				low = mid + 1;
			}
			else {
				float alpha = (pos_ - anchorPos0) / (anchorPos1 - anchorPos0);
				color(j, i) = mAnchorColor(0, mid) * (1 - alpha) + mAnchorColor(0, mid + 1) * alpha;
				return;
			}
		}

		assert(0);
		printf("Get error.\n");
		return;
	}
}

// Optimized for 256 interval colormap
__global__ void getColorKernel2(cv::cuda::PtrStep<float3> mAnchorColor, cv::cuda::PtrStep<float> pos, cv::cuda::PtrStep<float3> color) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (j < __rows__ && i < __cols__) {
		float pos_ = pos(j, i);

		float rawIdx = pos_ * BIN_NUM;
		int idx = (int)rawIdx;
		float alpha = rawIdx - idx;
		color(j, i) = mAnchorColor(0, idx) * (1 - alpha) + mAnchorColor(0, idx + 1) * alpha;
	}
}



// Wrapper function for color mapping
cv::cuda::GpuMat getColorGpu(const cv::cuda::GpuMat& mAnchorPos, const cv::cuda::GpuMat& mAnchorColor, const cv::cuda::GpuMat& pos) {
	dim3 threadsPerBlock(16, 16);
	uint block_num_vertical = (pos.rows + threadsPerBlock.x - 1) / threadsPerBlock.x;
	uint block_num_horizontal = (pos.cols + threadsPerBlock.y - 1) / threadsPerBlock.y;
	dim3 numBlocks(block_num_horizontal,
		block_num_vertical);

	cv::cuda::GpuMat color(pos.size(), CV_32FC3);
	getColorKernel << <numBlocks, threadsPerBlock>> > (mAnchorPos, mAnchorColor, pos, color);
	CHECK(cudaDeviceSynchronize());

	return color;
}

cv::cuda::GpuMat getColorGpu2(const cv::cuda::GpuMat& mAnchorColor, const cv::cuda::GpuMat& pos) {
	dim3 threadsPerBlock(16, 16);
	uint block_num_vertical = (pos.rows + threadsPerBlock.x - 1) / threadsPerBlock.x;
	uint block_num_horizontal = (pos.cols + threadsPerBlock.y - 1) / threadsPerBlock.y;
	dim3 numBlocks(block_num_horizontal,
		block_num_vertical);

	cv::cuda::GpuMat color(pos.size(), CV_32FC3);
	getColorKernel2 << <numBlocks, threadsPerBlock >> > (mAnchorColor, pos, color);
	CHECK(cudaDeviceSynchronize());

	return color;
}