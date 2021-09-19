#include "cMapUtils.h"
using namespace cv;

namespace cMapParam {
	extern float eta;
	extern int roiMag;
}


namespace cMapBuffer {
	extern cv::Vec3f bgColorRgb;
}


namespace cMapUtils {
	int Load_Txt_File(const std::string& path, cv::Mat& MA_Out, int chns) {
		// Variables
		std::ifstream        IS_File;
		std::string          ST_Line;
		std::stringstream    SS_Line;
		std::string          S_Value;
		float				 F_Value;
		unsigned int    rows = 0;
		unsigned int    cols = 0;
		bool			flag = false;

		IS_File.open(path);
		if (!IS_File.is_open()) {
			std::cout << "File not found. " << std::endl;
			return -1;
		}

		// Read data and get size
		std::vector<float> inData;

		while (getline(IS_File, ST_Line)) {
			SS_Line.clear();
			SS_Line << ST_Line;

			while (SS_Line >> S_Value) {
				if (!flag) {
					cols++;
				}

				std::transform(S_Value.begin(), S_Value.end(), S_Value.begin(), std::tolower);
				if ("nan" == S_Value) {
					F_Value = NAN;
				}
				else {
					F_Value = std::stof(S_Value);
				}
				inData.push_back(F_Value);

			}
			// Only get size from first row
			flag = true;
			rows++;
		}
		IS_File.close();


		std::cout << "Image: " << path << " size: " << rows << "x" << cols << " loaded" << std::endl;

		// Create image copy
		cv::Mat Tmp = cv::Mat(inData);
		MA_Out = Tmp.reshape(chns, rows).clone();
		return 0;
	}

	int Load_Txt_Folder(const std::string& path, std::vector<cv::Mat>& MA_Outs, int chns) {
		//std::vector<std::string> fileList;

		//intptr_t handle;
		//struct _finddata_t fileinfo;

		//handle = _findfirst(path.c_str(), &fileinfo);
		//if (handle == -1) {
		//	return -1;
		//}
		//do
		//{
		//	std::string  fileName = fileinfo.name;
		//	fileList.push_back(fileName);

		//} while (!_findnext(handle, &fileinfo));


		//// sort the vector 
		//std::sort(fileList.begin(), fileList.end(), [](const std::string& filename1, const std::string& filename2) {
		//	int id1 = std::stoi(filename1);
		//	int id2 = std::stoi(filename2);

		//	return id1 < id2;
		//});

		//// Split to get directory path
		//char drive[100], dir[999];
		//_splitpath(path.c_str(), drive, dir, NULL, NULL);
		//for (int i = 0; i < fileList.size(); i++) {
		//	cv::Mat data;
		//	cMapUtils::Load_Txt_File(drive + (dir + fileList[i]), data, chns);
		//	MA_Outs.push_back(data);
		//}

		//_findclose(handle);
		//return 0;

		// Split to get directory path
		char dirName[999];
		_splitpath(path.c_str(), NULL, dirName, NULL, NULL);

		intptr_t handle;
		struct _finddata_t fileinfo;

		handle = _findfirst(path.c_str(), &fileinfo);
		if (handle == -1) {
			return -1;
		}
		do
		{
			std::string  fileName = fileinfo.name;
			//std::cout << dirName+fileName << std::endl;
			cv::Mat data;
			cMapUtils::Load_Txt_File(dirName + fileName, data, chns);
			MA_Outs.push_back(data);	// 引用，可能出问题

		} while (!_findnext(handle, &fileinfo));

		_findclose(handle);
		return 0;
	}
	

	void saveVecToFIle(const std::string& path, const float* vec, int len) {
		std::ofstream fout(path, std::ofstream::out);
		if (!fout)
		{
			std::cout << "File Not Opened" << std::endl;
			exit(-1);
		}

		for (int i = 0; i < len; i++) {
			fout << vec[i];
			fout << std::endl;
		}
		fout.close();
	}


	void saveMatTotxt(const std::string& path, cv::Mat& data) {
		data = cv::abs(data);//防止出现-nan(ind)
		std::ofstream fout(path, std::ofstream::out);
		if (!fout)
		{
			std::cout << "File Not Opened" << std::endl;
			exit(-1);
		}

		for (int i = 0; i<data.rows; i++)
		{
			for (int j = 0; j<data.cols; j++)
			{
				fout << data.at<float>(i, j) << " ";
			}
			fout << std::endl;
		}
		fout.close();
		std::cout << "File save successful！！！" << std::endl;
	}



	void normalizeSymmetric(cv::Mat& data, const cv::Mat& mask) {
		double minVal;
		cv::minMaxLoc(data, &minVal, NULL, NULL, NULL, mask);

		assert(minVal < 0);

		for (int i = 0; i < data.rows; i++) {
			float* pdata = data.ptr<float>(i);
			for (int j = 0; j < data.cols; j++) {
				pdata[j] = (pdata[j] - minVal) / (-2 * minVal);
			}
		}
	}


	cv::Mat getDiffWeight(const cv::Mat& data, const cv::Mat& dataGrad, const cv::Mat& dataLap, const cv::Mat& mask, const std::vector<cv::Point>& roi) {
		
		// Binning operation, find all index of each bin
		cv::Mat dataIdx;
		// NAN auto to be < 0
		data.convertTo(dataIdx, CV_32SC1, BIN_NUM - 1);//data*(BIN_NUM-1)

		std::vector<cv::Mat> binIdx;
		for (int i = 0; i < BIN_NUM; i++) {
			cv::Mat idx;
			cv::findNonZero( dataIdx==i, idx);
			binIdx.push_back(idx);
		}
		//输出bins里面的数据个数
		/*for (int i = 0; i < BIN_NUM; i++) {
			std::cout <<"bins:"<<i<<"  "<< binIdx[i].total() << std::endl;
		}
*/
		


		double maxGrad, maxLap;
		cv::minMaxLoc(dataGrad, NULL, &maxGrad, NULL, NULL, mask);
		cv::minMaxLoc(dataLap, NULL, &maxLap, NULL, NULL, mask);//


		

																// Calculate delta
		float delta;
		if (0 == maxLap) {
			delta = 0;
		}
		else {
			delta = (float)(maxGrad / (maxLap * sqrt(exp(1))));
			
		}

		// Calculate gv and hv seperately
		
		////cv::normalize(dataGrad, dataGrad, 0, 1, cv::NORM_MINMAX, -1, mask);
		//cv::normalize(dataLap, dataLap, 0, 1, cv::NORM_MINMAX, -1, mask);
		cv::Mat g, h;
		g = valueAverage(dataGrad, binIdx);
		h = valueAverage(dataLap, binIdx);

		//std::cout << g << std::endl;
		const float* pg = (float*)g.data;
		const float* ph = (float*)h.data;

		//找到pg里面的最小值
		double Min_gv = +INFINITY;
		for (int i = 0; i < BIN_NUM; i++) {
			if (pg[i] > 0) {
				if (Min_gv >pg[i])
				{
					Min_gv =pg[i];
				}
			}
		}
		std::cout << "Min_gv:" << Min_gv << std::endl;


		//存储gv与hv
		/*temp = dataName;
		std::string hv_path = "C:/Users/15324/Desktop/hv/" + temp + "-hv.txt";
		std::string gv_path = "C:/Users/15324/Desktop/gv/" + temp + "-gv.txt";
		saveVecToFIle(hv_path, ph, BIN_NUM);
		saveVecToFIle(gv_path, pg, BIN_NUM);*/



		
		//float add =0;

		// Calulate v
		std::cout << "delta:" << delta << std::endl;
		float* v = new float[BIN_NUM];
		float* label = new float[BIN_NUM];
		double Max_v= -INFINITY;
		double Min_v = +INFINITY;
		for (int i = 0; i < BIN_NUM; i++) {
			v[i] = +INFINITY;
			if (pg[i]>0) {      //fabs(pg[i]) >= 1e-10f//0.0006
				//if (pg[i]==Min_gv)//判断梯度为最小值，一阶导为0的情况
				//{
				//	v[i] = +INFINITY;
				//}
				//else {
					v[i] = -delta*delta*ph[i] / pg[i];
					//std::cout << v[i] << "    ";
					if (Max_v < abs(v[i]))
					{
						Max_v = abs(v[i]);
					}
					if (Min_v > abs(v[i]))
					{
						Min_v = abs(v[i]);
					}
					label[i] = 1;
				}
			//}

			if (v[i] == +INFINITY) {
				label[i] = 2;
			}
		}
		std::cout << "Max_v:" <<Max_v<< std::endl;
		std::cout << "Min_v:" << Min_v << std::endl;

		
		

		//加一个检索,符合条件的先变成绝对值然后进行归一化,所以这里已经对w进行归一化了
		for (int i = 0; i < BIN_NUM; i++) {
			
			if (label[i] ==1) {

				v[i] = (abs(v[i])-Min_v)/(Max_v-Min_v);//归一化
				//v[i] = abs(v[i]) ;
			}
		//std::cout << v[i] << "    ";
		}

		


		//存储vw
		/*float* v_w = new float[BIN_NUM];
		for (int i = 0; i < BIN_NUM; i++) {

			v_w[i] = exp(-cMapParam::eta*v[i]);
		}
		temp = dataName;
		std::string v_w_path = "C:/Users/15324/Desktop/" + temp + "-vw.txt";
		saveVecToFIle(v_w_path, v_w, BIN_NUM);*/





		// Assign grayImage to each data
		cv::Mat w(data.size(), CV_32FC1);
		for (int i = 0; i < w.rows; i++) {
			const int* pidx = dataIdx.ptr<int>(i);
			float* pw = w.ptr<float>(i);

			for (int j = 0; j < w.cols; j++) {
				// If !nan
				if (pidx[j] >= 0) {
					pw[j] = v[pidx[j]];
				}
				else {
					pw[j] = NAN;
				}
			}
		}


		

#if (EXP_INNER == POW_INNER)
		cv::Mat dst;
		cv::pow(w, 2, dst);
		w = dst;
#else
		
#pragma region ratio相关计算

		//计算概率ratio

		// Binning operation, find all index of each bin
		cv::Mat R_dataIdx;
		int bins = Ratio_bins;
		// NAN auto to be < 0
		data.convertTo(R_dataIdx, CV_32SC1, bins-1);//data*(BIN_NUM-1)

		//将bins里面的索引存下来
		std::vector<cv::Mat> R_binIdx;
		for (int i = 0; i < bins; i++) {
			cv::Mat R_idx;
			cv::findNonZero( R_dataIdx==i, R_idx);
			R_binIdx.push_back(R_idx);
		}


		//求出每一个bins当中的比例
		cv::Mat ratio;
		ratio = Ratio(R_binIdx,bins,0);

		//再归一化到0-1
		cv::normalize(ratio, ratio, 0, 1, cv::NORM_MINMAX, -1);

		//std::cout << " ratio:" << std::endl << ratio << std::endl;
     //存储ratio
		/*std::string ratio_path = "C:/Users/15324/Desktop/D16_ratio.txt";
		saveMatTotxt(ratio_path, ratio);*/
		


		//输出bins里面的数据个数
		//for (int i = 0; i < bins; i++) {
		//std::cout <<"bins:"<<i+1<<"  "<< R_binIdx[i].total() << std::endl;
		//}
		//
		//cv::Mat ratio_grad;
		////求bins里面的平均梯度
		//ratio_grad=Ratio_average(dataGrad, R_binIdx, bins);
		//std::cout << " ratio_grad:" << std::endl << ratio_grad << std::endl;

		//安排到对应的pv数据
		cv::Mat ratio_pv(data.size(), CV_32FC1);
		for (int i = 0; i <ratio_pv.rows; i++) {
			const int* R_pidx = R_dataIdx.ptr<int>(i);
			float* R_pr = ratio_pv.ptr<float>(i);

			for (int j = 0; j < ratio_pv.cols; j++) {
				// If !nan
				if (R_pidx[j] >= 0) {
					//R_pr[j] = ratio.at<float>(R_pidx[j], 0)*ratio_grad.at<float>(R_pidx[j], 0);
					R_pr[j] = ratio.at<float>(R_pidx[j], 0);
				}
				else {
					R_pr[j] = NAN;
				}
			}
		}
		
		
		//将比例归一化
		
		//cv::normalize(ratio_pv, ratio_pv, 0, 1, cv::NORM_MINMAX, -1, mask);
		
		
		//存储ratio pv数据

		///*std::string ratio_path = "C:/Users/15324/Desktop/D16_ratio.txt";

		//saveMatTotxt(ratio_path, ratio_pv);*/
#pragma endregion



#pragma region 调节eta

		////进行代码的修改
		//	float c =0.5;
		//	float b =1;
		//	float d =0;
		//	  //w相当于矩阵，里面的每个元素都是x

		//	//将矩阵里的每个元素读出来
		//for(int i=0;i<w.rows;i++)
		//		for (int j = 0; j < w.cols; j++)
		//		{
		//			if (mask.at<uchar>(i, j)) {
		//				if (w.at<float>(i, j) < c&&w.at<float>(i, j) > -c)
		//					w.at<float>(i, j) = b;
		//				if (w.at<float>(i, j) >= c&&w.at<float>(i, j) <= -c)
		//					w.at<float>(i, j) = d;
		//			}
		//			else {
		//				w.at<float>(i, j) = NAN;
		//			}
		//		}
		 


		
#pragma endregion


		w = cv::abs(w);
		
		//将w存下来，也就是abs|X|

		/*std::string X_path = "C:/Users/15324/Desktop/org_x/"+dataName+".txt";

		saveMatTotxt(X_path, w);*/




#endif
		  //这里计算出平均梯度
		//cv::normalize(dataGrad, dataGrad, 0, 1, cv::NORM_MINMAX, -1, mask);//将梯度归一化
		//  float sum = 0;
		//  float count = 0;
		//       for (int i = 0; i<dataGrad.rows; i++)
		//	  		for (int j = 0; j < dataGrad.cols; j++)
		//	  		{
		//				if (mask.at<uchar>(i, j) && dataGrad.at<float>(i, j)>0) {//&& dataGrad.at<float>(i, j)>0
		//					sum = sum + dataGrad.at<float>(i, j);
		//					count = count + 1;
		//				}
		//	  		} 
		//	   gradMean = sum / count;


		//std::cout << "matMean:" << gradMean << std::endl;
		
		


		float s= -cMapParam::eta;
		//float s = -new_eta;
		w *= s;
		cv::exp(w, w);
		cv::normalize(w, w, 0, 1, cv::NORM_MINMAX, -1, mask);
		//imshow("pv_map", w);



		cv::Mat ratio_w = Ratio_average(w, R_binIdx, bins);
		//std::cout << "ratio_w" <<std::endl<< ratio_w << std::endl;
		cv::Mat rw;
		cv::multiply(ratio, ratio_w, rw);
		//std::cout << "ratio*ratio_w" << std::endl << rw << std::endl;


		cv::Mat PV;
		cv::multiply(ratio_pv, w,PV);

		 
		


		cv::normalize(PV, PV, 0, 1, cv::NORM_MINMAX, -1, mask);
		//将最终的pv存下来，

		//std::string PV_path = "C:/Users/15324/Desktop/data_pv/"+dataName+".txt";

		//saveMatTotxt(PV_path,PV);


		// Magnify roi
		for (int i = 0; i < roi.size(); i++) {
			PV.at<float>(roi[i]) *= cMapParam::roiMag;
		}


		return PV;
		//return w;
	}
















	cv::Mat valueAverage(const cv::Mat& values, const std::vector<cv::Mat>& binIdx) {
		cv::Mat averageVec(BIN_NUM, 1, CV_32FC1, 0.f);
		for (int i = 0; i < BIN_NUM; i++) {
			float sum = 0.f;
			// Calculate mean value
			for (int j = 0; j < binIdx[i].total(); j++) {
				float value = values.at<float>(binIdx[i].at<cv::Point>(j));
				if (!isnan(value)) {
					sum += value;	// Now should never get here.
				}
				//sum += values.at<float>(binIdx[i].at<cv::Point>(j));
			}

			if (binIdx[i].total() != 0) {
				sum /= (float)binIdx[i].total();
			}

			averageVec.at<float>(i, 0) = sum;
		}

		return averageVec;
	}

//求bins
	cv::Mat Ratio(const std::vector<cv::Mat>& binIdx,int bins,float threshould) {
		cv::Mat ratio(bins, 1, CV_32FC1, 0.f);
		float sum = 0;
		for (int i = 0; i < bins; i++) {
		//	std::cout << binIdx[i].total() << std::endl;
			sum = sum + (float)binIdx[i].total();
		}
		std::cout << "point sum:" <<sum<< std::endl;
		//sum是总和
		float v_min = +INFINITY;
		//求出符合阈值的最小bins个数
		for (int i = 0; i < bins; i++) {
			float v = binIdx[i].total() / sum;
			if (v > threshould&&v < v_min)
				v_min = v;
		}
		for (int i = 0; i < bins; i++) {
			float p;
			/*if (binIdx[i].total()/sum>threshould) {
				p=(float)binIdx[i].total()/sum;
				
			}
			else {
				p =  v_min/sum;
			}*/
			
		    p = binIdx[i].total();
			ratio.at<float>(i, 0) = p;

		}
		return ratio;
	}

	//求每一个bins的均值梯度
	cv::Mat Ratio_average(const cv::Mat& values, const std::vector<cv::Mat>& binIdx, int bins) {
		cv::Mat averageVec(bins, 1, CV_32FC1, 0.f);
		for (int i = 0; i < bins; i++) {
			float sum = 0.f;
			// Calculate mean value
			
			for (int j = 0; j < binIdx[i].total(); j++) {
				
				float value = values.at<float>(binIdx[i].at<cv::Point>(j));
				if (!isnan(value)) {
					sum += value;	// Now should never get here.
				}
				//sum += values.at<float>(binIdx[i].at<cv::Point>(j));
			}

			if (binIdx[i].total() != 0) {
				sum /= (float)binIdx[i].total();
			}

			averageVec.at<float>(i, 0) = sum;
		}

		return averageVec;
	}

	std::vector<cv::Mat> getHist(const cv::Mat& data, const cv::Mat& mask) {
		// Binning operation, find all index of each bin
		cv::Mat binData;
		// NAN auto to -xxxxx
		data.convertTo(binData, CV_32SC1, BIN_NUM - 1);

		std::vector<cv::Mat> hist;
		for (int i = 0; i < BIN_NUM; i++) {
			cv::Mat idx;
			cv::findNonZero(i == binData, idx);
			hist.push_back(idx);
		}

		return hist;
	}


	cv::Mat getContrastBinWeight(const std::vector<cv::Mat>& hist, const cv::Mat& mask) {
		cv::Mat paddingMask;
		cv::copyMakeBorder(mask, paddingMask, (N - 1) / 2, (N - 1) / 2, (N - 1) / 2, (N - 1) / 2, cv::BORDER_CONSTANT, 255);//可用非NAN的值去填充，这样不会对结果产生影响

		cv::Mat contrastWeight;

		for (int i = 0; i < hist.size(); i++) {
			cv::Mat pix = hist[i];
			int n = pix.rows;

			float w = 0;
			for (int j = 0; j < n; j++) {
				float temp = nearestDistance2nan(pix.at<cv::Point>(j), paddingMask);
				w += exp(-temp);
			}

			if (n != 0) {
				w /= n;
			}
			contrastWeight.push_back(w);
		}

		cv::normalize(contrastWeight, contrastWeight, 0, 1, cv::NORM_MINMAX, -1);
		return contrastWeight;
	}
	inline float nearestDistance2nan(const cv::Point& targetPosition, const cv::Mat& mask) {
		float distance = +M_INF;

		for (int i = -(N - 1) / 2; i < +(N - 1) / 2; i++) {
			for (int j = -(N - 1) / 2; j < (N - 1) / 2; j++) {
				if (!mask.at<uchar>((N - 1) / 2 + targetPosition.y + j, (N - 1) / 2 + targetPosition.x + i)) {
					float temp = sqrt(i*i + j*j);
					if (temp < distance) {
						distance = temp;
					}
				}
			}
		}

		return distance;
	}



	void getContrastWeight(const std::vector<cv::Mat>& hist, const cv::Mat& mask, cv::Mat& contrastWeight, cv::Mat& contrastPosition) {
		contrastWeight = cv::Mat(mask.size(), CV_32FC1, 0.f);
		contrastPosition = cv::Mat(mask.size(), CV_32SC2, cv::Vec2i(0, 0));

		cv::Mat paddingMask;
		cv::copyMakeBorder(mask, paddingMask, (N - 1) / 2, (N - 1) / 2, (N - 1) / 2, (N - 1) / 2, cv::BORDER_CONSTANT, 255);//可用非NAN的值去填充，这样不会对结果产生影响

																															// For each bin
		for (int i = 0; i < hist.size(); i++) {
			cv::Mat pix = hist[i];
			int n = pix.rows;

			// For each pixel which falls in this bin
			for (int j = 0; j < n; j++) {
				float nearestDistance;
				cv::Point nearstPosition;
				nearestDistance2nan(pix.at<cv::Point>(j), paddingMask, &nearestDistance, &nearstPosition);

				contrastWeight.at<float>(pix.at<cv::Point>(j)) = exp(-nearestDistance) / n;
				contrastPosition.at<cv::Point>(pix.at<cv::Point>(j)) = nearstPosition;
			}
		}

		cv::normalize(contrastWeight, contrastWeight, 0, 1, cv::NORM_MINMAX, -1, mask);
	}
	inline void nearestDistance2nan(const cv::Point& targetPosition, const cv::Mat& mask, float* distance, cv::Point* position) {
		*distance = +M_INF;

		for (int i = -(N - 1) / 2; i < +(N - 1) / 2; i++) {
			for (int j = -(N - 1) / 2; j < (N - 1) / 2; j++) {
				if (!mask.at<uchar>((N - 1) / 2 + targetPosition.y + j, (N - 1) / 2 + targetPosition.x + i)) {
					float temp = sqrt(i*i + j*j);
					if (temp < *distance) {
						*distance = temp;
						*position = cv::Point(targetPosition.x + i, targetPosition.y + j);
					}
				}
			}
		}
	}


	cv::Mat initAnchorPos(float anchor0, float anchor1, int num) {//初始化控制点的位置
		cv::Mat anchors(num, 1, CV_32FC1);
		for (int i = 0; i < num; i++) {
			float theta = (float)i / (num - 1);
			anchors.at<float>(i, 0) = anchor0 * (1 - theta) + anchor1 * theta;;
		}

		return anchors;
	}


	void insertRow(cv::Mat& matrix, const cv::Mat& row, int nrow) {
		assert(matrix.cols == row.cols && 1 == row.rows);

		assert(nrow >= 0 && nrow < matrix.rows);

		cv::Mat newMatrix(matrix.rows + 1, matrix.cols, matrix.type());


		matrix.rowRange(0, nrow).copyTo(newMatrix.rowRange(0, nrow));
		row.copyTo(newMatrix.row(nrow));
		matrix.rowRange(nrow, matrix.rows).copyTo(newMatrix.rowRange(nrow + 1, newMatrix.rows));


		matrix = newMatrix;
	}

	void getDvts(const cv::Mat& data, const cv::Mat& mask, cv::Mat& dataGrad, cv::Mat& dataLap) {
		// 1. padding operation
		cv::Mat paddingData;
		cv::copyMakeBorder(data, paddingData, 1, 1, 1, 1, cv::BORDER_REPLICATE);

		cv::Mat paddingMask;
		cv::copyMakeBorder(mask, paddingMask, 1, 1, 1, 1, cv::BORDER_REPLICATE);

		// 2. Compute gradient using central difference
		cv::Mat rawGrad(data.size(), CV_32FC1);
		for (int i = 1; i < paddingData.rows - 1; i++) {
			for (int j = 1; j < paddingData.cols - 1; j++) {
				if (paddingMask.at<uchar>(i, j)) {
					float c= paddingData.at<float>(i, j);
					float xl, xr, yl, yr;
					xl = paddingData.at<float>(i, j);
					xr = paddingData.at<float>(i, j);
					yl = paddingData.at<float>(i, j);
					yr = paddingData.at<float>(i, j);


					if (paddingMask.at<uchar>(i, j - 1)) {
						xl = paddingData.at<float>(i, j - 1);
					}
					if (paddingMask.at<uchar>(i, j + 1)) {
						xr = paddingData.at<float>(i, j + 1);
					}
					if (paddingMask.at<uchar>(i - 1, j)) {
						yl = paddingData.at<float>(i - 1, j);
					}
					if (paddingMask.at<uchar>(i + 1, j)) {
						yr = paddingData.at<float>(i + 1, j);
					}

					float gx, gy;
					gx = xl * (-0.5f) + xr * (0.5f);
					gy = yl * (-0.5f) + yr * (0.5f);

					rawGrad.at<float>(i - 1, j - 1) = sqrt(gx*gx + gy*gy);


					


				}
				else {
					rawGrad.at<float>(i - 1, j - 1) = NAN;
				}
			}
		}


		//cv::normalize(rawGrad, rawGrad, 0, 1, cv::NORM_MINMAX, -1, mask);
		dataGrad = rawGrad;


		// 3. compute laplacian using difference between pixel and mean value of its four neighbors
		cv::Mat rawLap(data.size(), CV_32FC1);
		for (int i = 1; i < paddingData.rows - 1; i++) {
			for (int j = 1; j < paddingData.cols - 1; j++) {

				if (paddingMask.at<uchar>(i, j)) {
					float sum = 0.f;

					if (paddingMask.at<uchar>(i, j + 1)) {
						sum += paddingData.at<float>(i, j + 1);
					}
					if (paddingMask.at<uchar>(i + 1, j)) {
						sum += paddingData.at<float>(i + 1, j);
					}
					if (paddingMask.at<uchar>(i, j - 1)) {
						sum += paddingData.at<float>(i, j - 1);
					}
					if (paddingMask.at<uchar>(i - 1, j)) {
						sum += paddingData.at<float>(i - 1, j);
					}

					sum /= 4;

					// Write into laplacian
					rawLap.at<float>(i - 1, j - 1) = sum - paddingData.at<float>(i, j);
				}
				else {
					rawLap.at<float>(i - 1, j - 1) = NAN;
				}
			}
		}

		//cv::normalize(rawLap, rawLap, 0, 1, cv::NORM_MINMAX, -1, mask);//归一化到-1到1，与论文中图一样
		dataLap = rawLap;
	}





	void matrix2array1d(const cv::Mat& cvMat, double* arr, int offset) {
		int len = cvMat.rows * cvMat.cols * cvMat.channels();

		float* data = (float*)cvMat.data;
		std::copy(data, data + len, arr + offset);
	}
	void array1d2matrix(const double* arr, cv::Mat& cvMat, int len, int offset) {
		std::copy(arr + offset, arr + offset + len, (float*)cvMat.data);
	}





	void dispImage(const std::string& windowName, const cv::cuda::GpuMat& image, bool isGray) {
		cv::Mat dispImage(image);

		if (!isGray) {
			cv::cvtColor(dispImage, dispImage, cv::COLOR_Lab2BGR);
		}

		cv::namedWindow(windowName, cv::WINDOW_NORMAL);

		cv::Size size = dispImage.size();
		if (size.width>600) {
		size = size/3;
		}
		cv::resizeWindow(windowName, size);
		cv::imshow(windowName, dispImage);
	}

	void dispImage(const std::string& windowName, const cv::Mat& image, bool isGray) {
		cv::Mat dispImage;
		image.copyTo(dispImage);

		if (!isGray) {
			cv::cvtColor(dispImage, dispImage, cv::COLOR_Lab2BGR);
		}

		cv::namedWindow(windowName, cv::WINDOW_NORMAL);

		
		cv::Size size = dispImage.size();
		if (size.width >600) {
			size = size / 4;
		}
		/*if (size.width < 150) {
		size *= 4;
		}*/
		cv::resizeWindow(windowName, size);
		cv::imshow(windowName, dispImage);
	}


	void saveImage(const std::string& imagePath, const cv::cuda::GpuMat& image, bool isGray) {
		cv::Mat savimg(image);

		if (!isGray) {
			cv::cvtColor(savimg, savimg, CV_Lab2BGR);
		}
		savimg.convertTo(savimg, CV_8UC3, 255);
		cv::imwrite(imagePath, savimg);
	}

	void saveImage(const std::string& imagePath, const cv::Mat& image, bool isGray) {
		cv::Mat savimg;
		image.copyTo(savimg);

		if (!isGray) {
			cv::cvtColor(savimg, savimg, CV_Lab2BGR);
		}

		savimg.convertTo(savimg, CV_8UC3, 255);
		cv::imwrite(imagePath, savimg);
	}


	void dispColormap(const std::string& windowName, const cv::Mat& anchorColor) {
		cv::Mat dispAnchorColor;
		cv::cvtColor(anchorColor, dispAnchorColor, cv::COLOR_Lab2BGR);
		cv::flip(dispAnchorColor, dispAnchorColor, 0);
		cv::namedWindow(windowName, cv::WINDOW_FREERATIO);
		cv::resizeWindow(windowName, cv::Size(1, dispAnchorColor.rows));
		cv::imshow(windowName, dispAnchorColor);
	}

	cv::Mat gray2bgr(const cv::Mat& grayImage, const cv::Mat& mask) {
		cv::Mat bgrImage(grayImage.size(), CV_32FC3);

		for (int i = 0; i < grayImage.rows; i++) {

			const uchar* pmask = mask.ptr<uchar>(i);
			const float* pgrayImage = grayImage.ptr<float>(i);
			cv::Vec3f* pbgrImage = bgrImage.ptr<cv::Vec3f>(i);

			for (int j = 0; j < grayImage.cols; j++) {
				if (pmask[j]) {
					pbgrImage[j] = cv::Vec3f(pgrayImage[j], pgrayImage[j], pgrayImage[j]);
				}
				else {
					// background rgb
					pbgrImage[j] = cMapBuffer::bgColorRgb;
				}
			}
		}

		return bgrImage;
	}

	cv::Mat overlay(const cv::Mat& srcTop, const cv::Mat& srcBottom, const cv::Mat& mask) {
		const float imgAlpha = 0.8;

		cv::Mat maskSrcTop;
		srcTop.copyTo(maskSrcTop, mask);
		maskSrcTop *= imgAlpha;


		cv::Mat tmp;
		srcBottom.copyTo(tmp);
		tmp *= (1 - imgAlpha);

		cv::Mat maskSrcBottom;
		srcBottom.copyTo(maskSrcBottom);
		tmp.copyTo(maskSrcBottom, mask);


		cv::Mat overlayImage;
		cv::add(maskSrcTop, maskSrcBottom, overlayImage);

		return overlayImage;
	}


	//cpu版本的计算localdiff
	 cv::Mat getLocalDiff( cv::Mat& labData, const cv::Mat& mask) {
		cv::Mat labDataWithBorder;
		cv::copyMakeBorder(labData, labDataWithBorder, 1, 1, 1, 1, cv::BORDER_REPLICATE);

		cv::Mat maskWithBorder;
		cv::copyMakeBorder(mask, maskWithBorder, 1, 1, 1, 1, cv::BORDER_REPLICATE);

		cv::Mat colorDiff(mask.size(), CV_32FC1);
		// Calculate local difference
		for (int i = 0; i < colorDiff.rows; i++) {
			const cv::Vec3f *pdata1, *pdata2, *pdata3;
			// i row pointer
			pdata1 = labDataWithBorder.ptr<cv::Vec3f>(i);
			// i + 1 row pointer
			pdata2 = labDataWithBorder.ptr<cv::Vec3f>(i + 1);
			// i + 2 row pointer
			pdata3 = labDataWithBorder.ptr<cv::Vec3f>(i + 2);


			const uchar *pmask1, *pmask2, *pmask3;
			pmask1 = maskWithBorder.ptr<uchar>(i);
			pmask2 = maskWithBorder.ptr<uchar>(i + 1);
			pmask3 = maskWithBorder.ptr<uchar>(i + 2);

			float* pdiff = colorDiff.ptr<float>(i);
			for (int j = 0; j < colorDiff.cols; j++) {
				if (pmask2[j + 1]) {//数据本身的中心点
					cv::Vec3f xl, xr, yl, yr;
					cv::Vec3f c = pdiff[j];
					xl = c;
					xr = c;
					yl = c;
					yr = c;
					if (pmask2[j]) {//判断左边
						xl = pdata2[j];
					}
					if (pmask2[j + 2]) {//右边
						xr = pdata2[j + 2];
					}
					if (pmask1[j + 1]) {//判断下边
						yl = pdata1[j + 1];
					}
					if (pmask3[j + 1]) {//判断上边
						yr = pdata3[j + 1];
					}

					//两个斜线上的4个点
					cv::Vec3f xy1, xy2, xy3, xy4;
					xy1 = c;
					xy2 = c;
					xy3 = c;
					xy4 = c;
					if (pmask3[j]) {//判断左上
						xy1 = pdata3[j];
					}
					if (pmask3[j + 2]) {//判断右上
						xy2 = pdata3[j + 2];
					}
					if (pmask1[j]) {//判断左下
						xy3 = pdata1[j];
					}
					if (pmask1[j + 2]) {//判断右下
						xy4 = pdata1[j + 2];
					}

					float g1 = sqrt((xl[0] - c[0])*(xl[0] - c[0]) + (xl[1] - c[1])*(xl[1] - c[1]) + (xl[2] - c[2])*(xl[2] - c[2]));
					float g2 = sqrt((xr[0] - c[0])*(xr[0] - c[0]) + (xr[1] - c[1])*(xr[1] - c[1]) + (xr[2] - c[2])*(xr[2] - c[2]));
					float g3 = sqrt((yl[0] - c[0])*(yl[0] - c[0]) + (yl[1] - c[1])*(yl[1] - c[1]) + (yl[2] - c[2])*(yl[2] - c[2]));
					float g4 = sqrt((yr[0] - c[0])*(yr[0] - c[0]) + (yr[1] - c[1])*(yr[1] - c[1]) + (yr[2] - c[2])*(yr[2] - c[2]));

					float g5 = sqrt((xy1[0] - c[0])*(xy1[0] - c[0]) + (xy1[1] - c[1])*(xy1[1] - c[1]) + (xy1[2] - c[2])*(xy1[2] - c[2]));
					float g6 = sqrt((xy2[0] - c[0])*(xy2[0] - c[0]) + (xy2[1] - c[1])*(xy2[1] - c[1]) + (xy2[2] - c[2])*(xy2[2] - c[2]));
					float g7 = sqrt((xy3[0] - c[0])*(xy3[0] - c[0]) + (xy3[1] - c[1])*(xy3[1] - c[1]) + (xy3[2] - c[2])*(xy3[2] - c[2]));
					float g8 = sqrt((xy4[0] - c[0])*(xy4[0] - c[0]) + (xy4[1] - c[1])*(xy4[1] - c[1]) + (xy4[2] - c[2])*(xy4[2] - c[2]));


					pdiff[j] = (g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) / 8;




				}
				else {
					pdiff[j] = NAN;
				}
			}
		}

		cv::Mat normLocalDiff;

		cv::normalize(colorDiff, normLocalDiff, 0, 20, cv::NORM_MINMAX, -1, mask);// Will lead unmasked region to 0.

		return normLocalDiff;

	}


};