#include "cMapOpt.h"

namespace cMapParam {
	float alpha = 100;//调
	float beta = 0.f;					
	float gamma = 0.f;					
	float eta = 0.1f;		//调			

	float t1 = 5.f;	//调				
	float t2 = 300.f;			

	int roiMag = 100;

	int rows;
	int cols;

	int varNum;
	int anchorNum;
}

namespace cMapBuffer {
	cMap oriColormap;
	float maxAnchorLength;

	cv::Mat oriAnchorColor;
	cv::Mat oriAnchorLengths;

	cv::Mat cpudata;
	cv::Mat cpumask;
	cv::Mat cpudiffWeight;

	cv::cuda::GpuMat data;
	cv::cuda::GpuMat mask;
	cv::cuda::GpuMat diffWeight;


	cv::cuda::GpuMat contrastWeight;
	cv::cuda::GpuMat bgImageL;

	std::vector<cv::Point> roi;
	std::vector<int> anchorFixed;
	cv::Vec3f bgColorRgb(255, 255, 255);
	cv::Vec3f bgColor(100,0,0);
	cv::Mat bgImg;

	void initOptBuffer(const cMap& oriColormap,
		const cv::Mat& oriAnchorColor,
		const cv::Mat& oriAnchorLengths,
		const cv::Mat& data,
		const cv::Mat& mask,
		const cv::Mat& bgImageL) {

		data.copyTo(cMapBuffer::cpudata);
		mask.copyTo(cMapBuffer::cpumask);

		cMapBuffer::oriColormap = oriColormap;
		cMapBuffer::maxAnchorLength = oriColormap.mAnchorLength.at<float>(BIN_NUM);

		oriAnchorColor.copyTo(cMapBuffer::oriAnchorColor);
		oriAnchorLengths.copyTo(cMapBuffer::oriAnchorLengths);

		
		cMapBuffer::data.upload(data);
		cMapBuffer::mask.upload(mask);


		// Compute gradient and laplacian of data
		cv::Mat dataGrad, dataLap;
		cMapUtils::getDvts(data, mask, dataGrad, dataLap);
		cv::Mat w = cMapUtils::getDiffWeight(data, dataGrad, dataLap, mask, roi);
		cMapBuffer::diffWeight.upload(w);

		w.copyTo(cMapBuffer::cpudiffWeight);

		std::vector<cv::Mat> hist = cMapUtils::getHist(data, mask);

		// Compute contrast weight
		cv::Mat cw, cp;
		cMapUtils::getContrastWeight(hist, mask, cw, cp);
		cMapBuffer::contrastWeight.upload(cw);

		// Find nearest position of background's pixel 
		cv::Mat tmp(cp.size(), CV_32FC1);
		for (int j = 0; j < cp.rows; j++) {
			for (int k = 0; k < cp.cols; k++) {
				tmp.at<float>(j, k) = bgImageL.at<float>(cp.at<cv::Point>(j, k));
			}
		}
		cMapBuffer::bgImageL.upload(tmp);

	}
}

namespace cMapOpt {
	//CPU版本的获取localdiff
	void getLocalDiffCPU(const cv::Mat& LabData, const cv::Mat& mask, cv::Mat& LocalDiff) {
		// 1. padding operation
		cv::Mat paddingLabData;
		cv::copyMakeBorder(LabData, paddingLabData, 1, 1, 1, 1, cv::BORDER_REPLICATE);

		cv::Mat paddingMask;
		cv::copyMakeBorder(mask, paddingMask, 1, 1, 1, 1, cv::BORDER_REPLICATE);

		cv::Mat localdiff(LabData.size(), CV_32FC1);
		for (int i = 1; i < paddingLabData.rows - 1; i++) {
			for (int j = 1; j < paddingLabData.cols - 1; j++) {
				if (paddingMask.at<uchar>(i, j)) {
					cv::Vec3f c = paddingLabData.at<cv::Vec3f>(i, j);
					cv::Vec3f xl, xr, yl, yr;
					xl = c;
					xr = c;
					yl = c;
					yr = c;

					if (paddingMask.at<uchar>(i, j - 1)) {
						xl = paddingLabData.at<cv::Vec3f>(i, j - 1);
					}
					if (paddingMask.at<uchar>(i, j + 1)) {
						xr = paddingLabData.at<cv::Vec3f>(i, j + 1);
					}
					if (paddingMask.at<uchar>(i - 1, j)) {
						yl = paddingLabData.at<cv::Vec3f>(i - 1, j);
					}
					if (paddingMask.at<uchar>(i + 1, j)) {
						yr = paddingLabData.at<cv::Vec3f>(i + 1, j);
					}

					cv::Vec3f xy1, xy2, xy3, xy4;
					xy1 = c;
					xy2 = c;
					xy3 = c;
					xy4 = c;

					if (paddingMask.at<uchar>(i-1, j-1)) {
						xy1 = paddingLabData.at<cv::Vec3f>(i-1, j - 1);
					}
					if (paddingMask.at<uchar>(i-1, j + 1)) {
						xy2 = paddingLabData.at<cv::Vec3f>(i-1, j + 1);
					}
					if (paddingMask.at<uchar>(i+1, j-1)) {
						xy3 = paddingLabData.at<cv::Vec3f>(i + 1, j - 1);
					}
					if (paddingMask.at<uchar>(i + 1, j+1)) {
						xy4 = paddingLabData.at<cv::Vec3f>(i + 1, j+1);
					}

					float g1 = sqrt((xl[0] - c[0])*(xl[0] - c[0]) + (xl[1] - c[1])*(xl[1] - c[1]) + (xl[2] - c[2])*(xl[2] - c[2]));
					float g2 = sqrt((xr[0] - c[0])*(xr[0] - c[0]) + (xr[1] - c[1])*(xr[1] - c[1]) + (xr[2] - c[2])*(xr[2] - c[2]));
					float g3 = sqrt((yl[0] - c[0])*(yl[0] - c[0]) + (yl[1] - c[1])*(yl[1] - c[1]) + (yl[2] - c[2])*(yl[2] - c[2]));
					float g4 = sqrt((yr[0] - c[0])*(yr[0] - c[0]) + (yr[1] - c[1])*(yr[1] - c[1]) + (yr[2] - c[2])*(yr[2] - c[2]));

					float g5 = sqrt((xy1[0] - c[0])*(xy1[0] - c[0]) + (xy1[1] - c[1])*(xy1[1] - c[1]) + (xy1[2] - c[2])*(xy1[2] - c[2]));
					float g6 = sqrt((xy2[0] - c[0])*(xy2[0] - c[0]) + (xy2[1] - c[1])*(xy2[1] - c[1]) + (xy2[2] - c[2])*(xy2[2] - c[2]));
					float g7 = sqrt((xy3[0] - c[0])*(xy3[0] - c[0]) + (xy3[1] - c[1])*(xy3[1] - c[1]) + (xy3[2] - c[2])*(xy3[2] - c[2]));
					float g8 = sqrt((xy4[0] - c[0])*(xy4[0] - c[0]) + (xy4[1] - c[1])*(xy4[1] - c[1]) + (xy4[2] - c[2])*(xy4[2] - c[2]));


					localdiff.at<float>(i - 1, j - 1) = (g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8;


				}
				else {
					localdiff.at<float>(i - 1, j - 1) = NAN;
				}
			}
		}

		cv::normalize(localdiff, localdiff, 0, 20, cv::NORM_MINMAX, -1, mask);
		
		LocalDiff = localdiff;

		
	}

	void getLocalDiffCPU_V2(const cv::Mat& LabData, const cv::Mat& mask, cv::Mat& LocalDiff) {
		// 1. padding operation
		cv::Mat paddingLabData;
		cv::copyMakeBorder(LabData, paddingLabData, 1, 1, 1, 1, cv::BORDER_REPLICATE);

		cv::Mat paddingMask;
		cv::copyMakeBorder(mask, paddingMask, 1, 1, 1, 1, cv::BORDER_REPLICATE);

		cv::Mat localdiff(LabData.size(), CV_32FC1);
		for (int i = 0; i < LabData.rows; i++) {
			for (int j =0; j < LabData.cols ; j++) {
				if (paddingMask.at<uchar>(i+1, j+1)) {
					cv::Vec3f c = paddingLabData.at<cv::Vec3f>(i+1, j+1);
					cv::Vec3f xl, xr, yl, yr;
					xl = c;
					xr = c;
					yl = c;
					yr = c;

					if (paddingMask.at<uchar>(i+1, j )) {
						xl = paddingLabData.at<cv::Vec3f>(i+1, j );
					}
					if (paddingMask.at<uchar>(i+1, j + 2)) {
						xr = paddingLabData.at<cv::Vec3f>(i+1, j + 2);
					}
					if (paddingMask.at<uchar>(i , j+1)) {
						yl = paddingLabData.at<cv::Vec3f>(i , j+1);
					}
					if (paddingMask.at<uchar>(i +2, j+1)) {
						yr = paddingLabData.at<cv::Vec3f>(i + 2, j + 1);
					}

					cv::Vec3f xy1, xy2, xy3, xy4;
					xy1 = c;
					xy2 = c;
					xy3 = c;
					xy4 = c;

					if (paddingMask.at<uchar>(i +2, j )) {
						xy1 = paddingLabData.at<cv::Vec3f>(i +2, j );
					}
					if (paddingMask.at<uchar>(i +2, j + 2)) {
						xy2 = paddingLabData.at<cv::Vec3f>(i +2, j +2);
					}
					if (paddingMask.at<uchar>(i , j )) {
						xy3 = paddingLabData.at<cv::Vec3f>(i , j );
					}
					if (paddingMask.at<uchar>(i , j + 2)) {
						xy4 = paddingLabData.at<cv::Vec3f>(i , j + 2);
					}

					float g1 = sqrt((xl[0] - c[0])*(xl[0] - c[0]) + (xl[1] - c[1])*(xl[1] - c[1]) + (xl[2] - c[2])*(xl[2] - c[2]));
					float g2 = sqrt((xr[0] - c[0])*(xr[0] - c[0]) + (xr[1] - c[1])*(xr[1] - c[1]) + (xr[2] - c[2])*(xr[2] - c[2]));
					float g3 = sqrt((yl[0] - c[0])*(yl[0] - c[0]) + (yl[1] - c[1])*(yl[1] - c[1]) + (yl[2] - c[2])*(yl[2] - c[2]));
					float g4 = sqrt((yr[0] - c[0])*(yr[0] - c[0]) + (yr[1] - c[1])*(yr[1] - c[1]) + (yr[2] - c[2])*(yr[2] - c[2]));

					float g5 = sqrt((xy1[0] - c[0])*(xy1[0] - c[0]) + (xy1[1] - c[1])*(xy1[1] - c[1]) + (xy1[2] - c[2])*(xy1[2] - c[2]));
					float g6 = sqrt((xy2[0] - c[0])*(xy2[0] - c[0]) + (xy2[1] - c[1])*(xy2[1] - c[1]) + (xy2[2] - c[2])*(xy2[2] - c[2]));
					float g7 = sqrt((xy3[0] - c[0])*(xy3[0] - c[0]) + (xy3[1] - c[1])*(xy3[1] - c[1]) + (xy3[2] - c[2])*(xy3[2] - c[2]));
					float g8 = sqrt((xy4[0] - c[0])*(xy4[0] - c[0]) + (xy4[1] - c[1])*(xy4[1] - c[1]) + (xy4[2] - c[2])*(xy4[2] - c[2]));


					localdiff.at<float>(i , j ) = (g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8;


				}
				else {
					localdiff.at<float>(i , j ) = NAN;
				}
			}
		}

		cv::normalize(localdiff, localdiff, 0, 20, cv::NORM_MINMAX, -1, mask);

		LocalDiff = localdiff;


	}

	float cpuboundFunc(const cv::Mat& localDiff, const cv::Mat& diffWeight, const cv::Mat& mask){
		float E = 0;

		cv::Mat dst;
		cv::multiply(localDiff, diffWeight, dst);

		E = cv::sum(dst)[0];
		return -E / (cv::sum(mask)[0] / 255);
	}

	float boundFunc(const cv::cuda::GpuMat& localDiff, 
		const cv::cuda::GpuMat& diffWeight, 
		const cv::cuda::GpuMat& mask) {

		cv::cuda::GpuMat dst;

		cv::cuda::multiply(localDiff, diffWeight, dst);
		
		float E = cv::cuda::sum(dst, mask)[0];
		//std::cout << -E / (cv::cuda::sum(mask)[0] / 255) << std::endl;

#pragma region 输出localdiff
		//强制转换
		/*cv::cuda::GpuMat local_diff;
		localDiff.copyTo(local_diff);
		cv::Mat Local_diff = cv::Mat(local_diff);
		cv::cuda::GpuMat diffweight;
		diffWeight.copyTo(diffweight);
		cv::Mat Diffweight = cv::Mat(diffweight);

		cv::Mat Mask = cv::Mat(mask);*/
		
		/*float num = 0;
		float s = 0;
		for(int i=0;i<Mask.rows;i++)
			for (int j = 0; j < Mask.cols; j++)
			{
				if (Mask.at<uchar>(i, j)) {
					num++;
					s = s+abs(Diffweight.at<float>(i, j) - Local_diff.at<float>(i, j));
				}
			}*/
		
		//先创建一个文件
		/*if (count % 7 == 0) {
			std::ostringstream Fe;
			Fe << count;
			std::string F(Fe.str());

			std::string localdiff_path = "C:/Users/15324/Desktop/Ratio_average/" + dataName + "_" + mapName + "_" + F + ".txt";

			
			cMapUtils::saveMatTotxt(localdiff_path, Local_diff);
		}
		  count = count + 1;*/
#pragma endregion
	
		//std::cout << -E / (cv::cuda::sum(mask)[0] / 255) << std::endl;

		return -E/(cv::cuda::sum(mask)[0] / 255);
		//std::cout << s / num << std::endl;
		//return s/num;
	}



	float boundFunc_1(const cv::cuda::GpuMat& Labdata,
		const cv::cuda::GpuMat& PVdata,
		const cv::cuda::GpuMat& mask) {
		cv::Mat LabData = cv::Mat(Labdata);
		cv::Mat PVData = cv::Mat(PVdata);
		cv::Mat Mask = cv::Mat(mask);


		cv::Mat R(Mask.size(), CV_32FC1);

		float num = 0;
		float s = 0;

		for (int i = 0; i<Mask.rows; i++)
			for (int j = 0; j < Mask.cols; j++)
			{
				if (Mask.at<uchar>(i, j)) {
					num++;
					cv::Vec3f a = LabData.at<cv::Vec3f>(i, j);
					cv::Vec3f b = PVData.at<cv::Vec3f>(i, j);
					cv::Vec3f diff =a - b;
					float d=sqrt((a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]) + (a[2] - b[2])*(a[2] - b[2]));
					//s = s + d;
					 s = s + abs(diff[0]) + abs(diff[1]) + abs(diff[2]);
					R.at<float>(i, j) = d;
				}
				else {
					R.at<float>(i, j) = NAN;
				}
			}
		//将结果归一化
		/*cv::normalize(R, R, 0, 1, cv::NORM_MINMAX, -1, Mask);
		float E = 0;
		for (int i = 0; i<Mask.rows; i++)
			for (int j = 0; j < Mask.cols; j++)
			{
				if (Mask.at<uchar>(i, j)) {
					E = E + R.at<float>(i, j);
				}

			}*/
		
		return -s / num;
		//return -E / num;
	}


	float boundFunc_2(const cv::cuda::GpuMat& localDiff,
		const cv::cuda::GpuMat& diffWeight,
		const cv::cuda::GpuMat& mask, const cv::cuda::GpuMat& data) {

		cv::Mat Local_diff = cv::Mat(localDiff);
		cv::Mat Diffweight = cv::Mat(diffWeight);
		cv::Mat Data = cv::Mat(data);
		cv::Mat Mask = cv::Mat(mask);

		//获取255个索引
		cv::Mat dataIdx;
		// NAN auto to be < 0
		Data.convertTo(dataIdx, CV_32SC1, BIN_NUM - 1);//data*(BIN_NUM-1)

		std::vector<cv::Mat> binIdx;
		for (int i = 0; i < BIN_NUM; i++) {
			cv::Mat idx;
			cv::findNonZero(dataIdx == i, idx);
			binIdx.push_back(idx);
		}
		cv::Mat pvVec(BIN_NUM, 1, CV_32FC1, 0.f);
		cv::Mat maxVec(BIN_NUM, 1, CV_32FC1, 0.f);
		cv::Mat minVec(BIN_NUM, 1, CV_32FC1, 0.f);
		for (int i = 0; i < BIN_NUM; i++) {
			float sum = 0.f;
			// Calculate mean value
			float max = 0;
			float min = 1;
			for (int j = 0; j < binIdx[i].total(); j++) {
				float value = Local_diff.at<float>(binIdx[i].at<cv::Point>(j));
				float pv_value = Diffweight.at<float>(binIdx[i].at<cv::Point>(j));
				if (!isnan(value)) {
					if (max < value)max = value;
					if (min > value)min = value;
				}
				if (!isnan(pv_value)) {
					sum += pv_value;
				}
				//sum += values.at<float>(binIdx[i].at<cv::Point>(j));
			}

			if (binIdx[i].total() != 0) {
				maxVec.at<float>(i, 0) = max;
				minVec.at<float>(i, 0) = min;
				pvVec.at<float>(i, 0) = sum/(float)binIdx[i].total();
			}
	
		}
		float E = 0;
		for (int i = 0; i < BIN_NUM; i++) {
			E = E + abs(pvVec.at<float>(i, 0) - minVec.at<float>(i, 0));
		}

		std::cout << E / 255 << std::endl;
		return E/255;

	}







	float contrastFunc(const cv::cuda::GpuMat& labData, 
		const cv::cuda::GpuMat& contrastWeight, 
		const cv::cuda::GpuMat& mask) {
		return contrastFuncGpu(labData, cMapBuffer::bgImageL, contrastWeight, mask);
	}

	float colormapFunc(const cv::Mat& newAnchorLengths) {
		
		float E = 0;
		for (int i = 0; i < newAnchorLengths.cols; i++) {
			float arc1 = cMapBuffer::oriAnchorLengths.at<float>(i) 
				/ cMapBuffer::maxAnchorLength;
			float arc2 = newAnchorLengths.at<float>(i)
				/ cMapBuffer::maxAnchorLength;
			E += abs(arc1 - arc2);
			
		}
		return E;
	}


	int KNEvalFC(KN_context_ptr             kc,
		CB_context_ptr             cb,
		KN_eval_request_ptr const  evalRequest,
		KN_eval_result_ptr  const  evalResult,
		void              * const  userParams)
	{
		const double *x;
		double *obj;
		double *c;

		if (evalRequest->type != KN_RC_EVALFC)
		{
			printf("*** callbackEvalFC incorrectly called with eval type %d\n",
				evalRequest->type);
			return(-1);
		}
		x = evalRequest->x;
		obj = evalResult->obj;
		c = evalResult->c;


		// Convert x to cv form. If opencv provides its numerical optimzation, this can be avoid.
		cv::Mat anchorPos(1, cMapParam::varNum, CV_32FC1);
		cMapUtils::array1d2matrix(x, anchorPos, cMapParam::varNum);
		cv::hconcat(cv::Mat(1, 1, CV_32FC1, 0.f), anchorPos, anchorPos);
		cv::hconcat(anchorPos, cv::Mat(1, 1, CV_32FC1, 1.f), anchorPos);

		
		cMap newColormap(anchorPos, cMapBuffer::oriAnchorColor);
		/*cv::Mat org_anchorPos = (cv::Mat_<float>(9, 1) << 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.825, 1);
		cMap orgColormap(anchorPos, cMapBuffer::oriAnchorColor);*/
		// Assign color for each data.
		cv::cuda::GpuMat labData = newColormap.getColor(cMapBuffer::data);
		


		//这里输出迭代过程中的图片
		/*if (count ==98) {
			cMapUtils::dispImage("outputs", cv::Mat(labData), false);
		}
		count = count + 1;*/

		float E1 = 0;

		// Compute local difference
#if (LOCALDIFF_METRIC == GRADIENT_METRIC)
		if (isGPU==1) {//为1就是GPU版本
			cv::cuda::GpuMat localDiff = getLocalDiffGpu(labData, cMapBuffer::mask);
			E1 = boundFunc(localDiff, cMapBuffer::diffWeight, cMapBuffer::mask);
		}
		else {//就是cpu版本，计算会慢些

			cv::Mat labData = newColormap.getColor(cMapBuffer::cpudata);
			/*cv::cuda::GpuMat localDiff = getLocalDiffGpu(cv::cuda::GpuMat(labData), cMapBuffer::mask);
			cv::Mat LocalDiff = cv::Mat(localDiff);*/
			cv::Mat LocalDiff;
				getLocalDiffCPU_V2(labData, cMapBuffer::cpumask,LocalDiff);
				E1 = boundFunc(cv::cuda::GpuMat(LocalDiff), cMapBuffer::diffWeight, cMapBuffer::mask);
			//E1 = cpuboundFunc(LocalDiff, cMapBuffer::cpudiffWeight, cMapBuffer::cpumask);
		}

		

#elif (LOCALDIFF_METRIC == DIFF_76_METRIC)
		cv::cuda::GpuMat localDiff = getLocalDiffGpu76(labData, cMapBuffer::mask);
#else
		localDiff = getLocalDiffGpu2000(labData, mask[i]);
#endif
		
   


  //float E1 = boundFunc_1(labData, PVData, cMapBuffer::mask);


   //float E1 = boundFunc_2(localDiff, cMapBuffer::diffWeight, cMapBuffer::mask, cMapBuffer::data);//不需要这一项




		// Compute anchors' arc lengths
		cv::Mat newAnchorLengths;
		newAnchorLengths = cMapBuffer::oriColormap.arcLength(anchorPos);
		float E3 = colormapFunc(newAnchorLengths);//保真项fedelity


		//std::cout << E1 << " "<<E3<<std::endl;
		

		//*obj = E1 * cMapParam::alpha + E3 * cMapParam::gamma*gradMean;// E2 * cMapParam::beta 

		*obj = E1 * cMapParam::alpha + E3 * cMapParam::gamma;
		//bfout << E1 << " " << E3 << " " << *obj;
		//bfout << std::endl;//换行


		//t1与t2的弧长限制
		 //li - li-1 < t2
		float T1 = cMapParam::t1;
		//float T1 = new_t1;
		float T2 = cMapParam::t2;
		
			for (int i = 1; i < cMapParam::anchorNum; i++) {
				c[i - 1] = newAnchorLengths.at<float>(i)
					- newAnchorLengths.at<float>(i - 1)
					- T2
					+ T1;
			}
			// li - li-1 > t1
			for (int i = 1; i < cMapParam::anchorNum; i++) {
				c[cMapParam::anchorNum - 2 + i] = newAnchorLengths.at<float>(i - 1)
					- newAnchorLengths.at<float>(i)
					+ T1;
			}

		
		return 0;
	}


	void KNOpt(cv::Mat& anchorPos) {

		/** Declare variables. */
		KN_context   *kc;
		int error;

		/** Create a new Knitro solver instance. */
		error = KN_new(&kc);
		if (error) exit(-1);
		if (kc == NULL)
		{
			printf("Failed to find a valid license.\n");
			exit(-1);
		}

		/** Override default options. */
		if (KN_set_int_param(kc, KN_PARAM_ALG, KN_ALG_ACT_SQP) != 0) //KN_ALG_ACT_SQP //KN_ALG_ACT_CG
			exit(-1);
		if (KN_set_int_param(kc, KN_PARAM_OUTMODE, KN_OUTMODE_SCREEN) != 0)
			exit(-1);
		if (KN_set_int_param(kc, KN_PARAM_OUTLEV, KN_OUTLEV_SUMMARY) != 0)
			exit(-1);
		if (KN_set_int_param(kc, KN_PARAM_GRADOPT, KN_GRADOPT_CENTRAL) != 0)
			exit(-1);
		if (KN_set_int_param(kc, KN_PARAM_LINSOLVER, KN_LINSOLVER_AUTO) != 0)
			exit(-1);
		if (KN_set_int_param(kc, KN_PARAM_LINESEARCH, KN_LINESEARCH_AUTO) != 0)
			exit(-1);
		if (KN_set_int_param(kc, KN_PARAM_ACT_QPALG, KN_ACT_QPALG_ACT_CG) != 0)
			exit(-1);
		if (KN_set_int_param(kc, KN_PARAM_HONORBNDS, KN_HONORBNDS_ALWAYS) != 0)
			exit(-1);


		int n, m1, m2;
		n = cMapParam::varNum;
		m1 = 2 * cMapParam::anchorNum - 2;	// First m1 nonlinear cons
		m2 = cMapParam::varNum - 1;		// Later m1 linear cons

		double *x = new double[n];
		cMapUtils::matrix2array1d(anchorPos, x);

		double *xLoBnds = new double[n];
		double *xUpBnds = new double[n];
		for (int i = 0; i < n; i++) {
			xLoBnds[i] = 0;//对于每一个变量，都需要大于0
			xUpBnds[i] = 1;//对于每一个变量，都需要小于1
		}

		// Fix points
		for (int i = 0; i < cMapBuffer::anchorFixed.size(); i++) {
			xLoBnds[cMapBuffer::anchorFixed[i]] = x[cMapBuffer::anchorFixed[i]];
			xUpBnds[cMapBuffer::anchorFixed[i]] = x[cMapBuffer::anchorFixed[i]];
		}

		int* xTypes = new int[n];
		for (int i = 0; i < n; i++) {
			xTypes[i] = KN_VARTYPE_CONTINUOUS;
		}


		/** Add the variables and set their bounds and types.
		*  Note: any unset lower bounds are assumed to be
		*  unbounded below and any unset upper bounds are
		*  assumed to be unbounded above. */
		error = KN_add_vars(kc, n, NULL);
		if (error) exit(-1);
		error = KN_set_var_lobnds_all(kc, xLoBnds);
		if (error) exit(-1);
		error = KN_set_var_upbnds_all(kc, xUpBnds);


		if (error) exit(-1);
		error = KN_set_var_types_all(kc, xTypes);
		if (error) exit(-1);
		/** Define an initial point.  If not set, Knitro will generate one. */
		error = KN_set_var_primal_init_values_all(kc, x);
		if (error) exit(-1);


		double* cUpBnds = new double[m1 + m2];
		// Nonlinear constraints up bound
		for (int i = 0; i < m1; i++) {
			cUpBnds[i] = 0;
		}
		// Linear constraints up bound
		for (int i = m1; i < m1 + m2; i++) {
			cUpBnds[i] = -epsilon;
		}


		/** Add the variables and set their bounds.
		*  Note: unset bounds assumed to be infinite. */
		error = KN_add_cons(kc, m1 + m2, NULL);
		if (error) exit(-1);
		error = KN_set_con_upbnds_all(kc, cUpBnds);
		if (error) exit(-1);

		/*double* cFeasTols = new double[m1 + m2];
		for (int i = 0; i < m1 + m2; i++) {
		cFeasTols[i] = 0.0;
		}
		KN_set_con_feastols_all(kc, cFeasTols);*/

		/** Used to define linear constraint structure */
		int* jacIndexCons = new int[m2 * 2];
		int* jacIndexVars = new int[m2 * 2];
		double* jacCoefs = new double[m2 * 2];

		/** Load the linear structure for all constraints at once. */
		int idx = 0;
		for (int i = m1; i < m1 + m2; i++) {
			int ii = i + 1;

			jacIndexCons[idx] = i;
			jacIndexCons[idx + 1] = i;

			jacIndexVars[idx] = i - m1;		// xi
			jacIndexVars[idx + 1] = i - m1 + 1;		// xi+1

			jacCoefs[idx] = 1;
			jacCoefs[idx + 1] = -1;

			idx += 2;
		}
		error = KN_add_con_linear_struct(kc, m2 * 2, jacIndexCons,
			jacIndexVars, jacCoefs);
		if (error) exit(-1);


		/** Add a callback function "callbackEvalFC" to evaluate the nonlinear
		*  structure in the objective and first two constraints.  Note that
		*  the linear terms in the objective and first two constraints were
		*  added above in "KN_add_obj_linear_struct()" and
		*  "KN_add_con_linear_struct()" and will not be specified in the
		*  callback. */
		CB_context   *cb;

		int* cIndices = new int[m1];
		for (int i = 0; i < m1; i++) {
			cIndices[i] = i;
		}

		error = KN_add_eval_callback(kc, KNTRUE, m1, cIndices, KNEvalFC, &cb);
		if (error) exit(-1);

		/** Set minimize or maximize (if not set, assumed minimize) */
		error = KN_set_obj_goal(kc, 0);//0是最小值优化，1是最大值优化
		if (error) exit(-1);


		/** Solve the problem.
		*
		*  Return status codes are defined in "knitro.h" and described
		*  in the Knitro manual.
		*/
		int  nStatus;
		double objSol;

		nStatus = KN_solve(kc);
		/** An example of obtaining solution information. */
		error = KN_get_solution(kc, &nStatus, &objSol, x, NULL);
		if (!error) {
			printf("Optimal objective value  = %e\n", objSol);
			printf("Optimal x\n");
			for (int i = 0; i < n; i++)
				printf("  x[%d] = %e\n", i, x[i]);
		}

		/** Delete the Knitro solver instance. */
		KN_free(&kc);

		/* Convert x to mat */
		cMapUtils::array1d2matrix(x, anchorPos, n);
	}


	void KNOpt2(cv::Mat& anchorPos) {
		std::cout << anchorPos.size();
		/** Declare variables. */
		KN_context   *kc;
		int error;

		/** Create a new Knitro solver instance. */
		error = KN_new(&kc);
		if (error) exit(-1);
		if (kc == NULL)
		{
			printf("Failed to find a valid license.\n");
			exit(-1);
		}

		/** Override default options. */
		if (KN_set_int_param(kc, KN_PARAM_ALG, KN_ALG_ACT_SQP) != 0) //KN_ALG_ACT_SQP //KN_ALG_ACT_CG//KN_ALG_BAR_DIRECT// KN_ALG_BAR_CG  
			exit(-1);
		/*if (KN_set_int_param(kc, KN_PARAM_MAXIT,50) != 0)
			exit(-1);*/
		if (KN_set_int_param(kc, KN_PARAM_OUTMODE, KN_OUTMODE_SCREEN) != 0)
			exit(-1);
		if (KN_set_int_param(kc, KN_PARAM_OUTLEV, KN_OUTLEV_SUMMARY) != 0)
			exit(-1);
		if (KN_set_int_param(kc, KN_PARAM_GRADOPT, KN_GRADOPT_CENTRAL) != 0)
			exit(-1);
		if (KN_set_int_param(kc, KN_PARAM_LINSOLVER, KN_LINSOLVER_AUTO) != 0)
			exit(-1);
		if (KN_set_int_param(kc, KN_PARAM_LINESEARCH, KN_LINESEARCH_AUTO) != 0)
			exit(-1);
		if (KN_set_int_param(kc, KN_PARAM_ACT_QPALG, KN_ACT_QPALG_ACT_CG) != 0)
			exit(-1);
		if (KN_set_int_param(kc, KN_PARAM_HONORBNDS, KN_HONORBNDS_ALWAYS) != 0)
			exit(-1);
		
		//设置参数
		/*int  status;
		status = KN_set_double_param(kc, KN_PARAM_ACT_LPFEASTOL, 1.0e-15);*/


		int n, m1;
		n = cMapParam::varNum;
		m1 = 2 * cMapParam::anchorNum - 2;	// First m1 nonlinear cons

		double *x = new double[n];
		cMapUtils::matrix2array1d(anchorPos, x);

		double *xLoBnds = new double[n];
		double *xUpBnds = new double[n];
		for (int i = 0; i < n; i++) {
			xLoBnds[i] = 0;
			xUpBnds[i] = 1;
		}

		// Fix points
		for (int i = 0; i < cMapBuffer::anchorFixed.size(); i++) {
			xLoBnds[cMapBuffer::anchorFixed[i]] = x[cMapBuffer::anchorFixed[i]];
			xUpBnds[cMapBuffer::anchorFixed[i]] = x[cMapBuffer::anchorFixed[i]];
		}

		int* xTypes = new int[n];
		for (int i = 0; i < n; i++) {
			xTypes[i] = KN_VARTYPE_CONTINUOUS;
		}


		/** Add the variables and set their bounds and types.
		*  Note: any unset lower bounds are assumed to be
		*  unbounded below and any unset upper bounds are
		*  assumed to be unbounded above. */
		error = KN_add_vars(kc, n, NULL);
		if (error) exit(-1);
		error = KN_set_var_lobnds_all(kc, xLoBnds);
		if (error) exit(-1);
		error = KN_set_var_upbnds_all(kc, xUpBnds);


		if (error) exit(-1);
		error = KN_set_var_types_all(kc, xTypes);
		if (error) exit(-1);
		/** Define an initial point.  If not set, Knitro will generate one. */
		error = KN_set_var_primal_init_values_all(kc, x);
		if (error) exit(-1);


		double* cUpBnds = new double[m1];
		// Nonlinear constraints up bound
		for (int i = 0; i < m1; i++) {
			cUpBnds[i] = 0;
		}


		/** Add the variables and set their bounds.
		*  Note: unset bounds assumed to be infinite. */
		error = KN_add_cons(kc, m1, NULL);
		if (error) exit(-1);
		error = KN_set_con_upbnds_all(kc, cUpBnds);
		if (error) exit(-1);


		/** Add a callback function "callbackEvalFC" to evaluate the nonlinear
		*  structure in the objective and first two constraints.  Note that
		*  the linear terms in the objective and first two constraints were
		*  added above in "KN_add_obj_linear_struct()" and
		*  "KN_add_con_linear_struct()" and will not be specified in the
		*  callback. */
		CB_context   *cb;

		int* cIndices = new int[m1];
		for (int i = 0; i < m1; i++) {
			cIndices[i] = i;
		}

		error = KN_add_eval_callback(kc, KNTRUE, m1, cIndices, KNEvalFC, &cb);
		if (error) exit(-1);

		/** Set minimize or maximize (if not set, assumed minimize) */
		error = KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE);
		if (error) exit(-1);

		///** Set a callback function that performs some user-defined task
		//*  after completion of each node in the branch-and-bound tree. */
		//error = KN_set_mip_node_callback(kc, &callbackProcessNode, kc);
		//if (error) exit(-1);

		/** Solve the problem.
		*
		*  Return status codes are defined in "knitro.h" and described
		*  in the Knitro manual.
		*/
		int  nStatus;
		double objSol;

		nStatus = KN_solve(kc);
		/** An example of obtaining solution information. */
		error = KN_get_solution(kc, &nStatus, &objSol, x, NULL);
		if (!error) {
			printf("Optimal objective value  = %e\n", objSol);
			printf("Optimal x\n");
			for (int i = 0; i < n; i++)
				printf("  x[%d] = %e\n", i, x[i]);
		}

		/** Delete the Knitro solver instance. */
		KN_free(&kc);

		/* Convert x to mat */
		cMapUtils::array1d2matrix(x, anchorPos, n);
	}
}

