#include "interface.h"
#include <Windows.h>

cv::Mat py_anchor_pos;
cv::Mat py_ori_anchor_color;

cv::Mat py_data;
cv::Mat py_mask;


/* @brief initialize parameters */
int init_param( float alpha,  float beta,  float gamma,  float eta) {//初始化参数
	DLL_CHECK(alpha >= 0 && beta >= 0 && gamma >= 0 && eta >= 0);

	cMapParam::alpha = alpha;
	cMapParam::beta = beta;
	cMapParam::gamma = gamma;
	cMapParam::eta = eta;

#ifdef M_DEBUG
	std::cout << "set parameters:" << std::endl;
	std::cout << "alpha:"<<cMapParam::alpha << std::endl;
	std::cout <<"beta:"<< cMapParam::beta << std::endl;
	std::cout << "gamma:"<<cMapParam::gamma << std::endl;
	std::cout << "eta:"<<cMapParam::eta << std::endl;
#endif // M_DEBUG



	return 0;
}

/* @brief initialize ROI magnitude */
int init_magnifier(const int magnitude) {
	DLL_CHECK(magnitude >= 1);
	cMapParam::roiMag = magnitude;

#ifdef M_DEBUG
	std::cout << "set magnifier:" << std::endl;
	std::cout << "roiMag:"<<cMapParam::roiMag << std::endl;
#endif // M_DEBUG


	return 0;
}


/* @brief scale lower and upper threshold for arcLengths' */
int init_threshold( float lt,  float ut) {

	cMapParam::t1 = lt;
	cMapParam::t2 = ut;
	DLL_CHECK(cMapParam::t1 > 0 &&
		cMapParam::t2 > 0 &&
		cMapParam::t1 < cMapParam::t2);

#ifdef M_DEBUG
	std::cout << "set thresholds:" << std::endl;//设置阈值
	std::cout << "t1:"<<cMapParam::t1 << std::endl;
	std::cout <<"t2:"<< cMapParam::t2 << std::endl;
#endif // M_DEBUG


	return 0;
}


/* @brief upload data from python to c++
* @param
*/
int init_data(float* data, int rows, int cols) {//初始化数据

	DLL_CHECK(rows > 0 &&
		cols > 0);
	cMapParam::rows = rows;
	cMapParam::cols = cols;

#ifdef M_DEBUG
	std::cout << "data size: " << std::endl;
	std::cout << cMapParam::rows << std::endl;
	std::cout << cMapParam::cols << std::endl;
#endif // M_DEBUG


	py_data.create(rows, cols, CV_32FC1);
	std::copy(data, data + rows * cols, (float*)py_data.data);
	py_mask = (py_data == py_data);

	return 0;
}


/* @brief upload anchor poses from python to c++
* 'anchor_color' is represented in rgb.
*/
int init_anchors(const float* anchor_pos, const float* anchor_color, int n) {//初始化颜色表进行输出
	DLL_CHECK(n > 0);

	cMapParam::varNum = n - 2;
	cMapParam::anchorNum = n;

#ifdef M_DEBUG
	std::cout << "anchor number: " << std::endl;
	std::cout << cMapParam::anchorNum << std::endl;
#endif // M_DEBUG


	py_anchor_pos = cv::Mat(1, cMapParam::anchorNum, CV_32FC1);
	std::copy(anchor_pos, anchor_pos + n, (float*)py_anchor_pos.data);

	py_ori_anchor_color = cv::Mat(1, cMapParam::anchorNum, CV_32FC3);
	std::copy(anchor_color, anchor_color + n * 3, (float*)py_ori_anchor_color.data);

	std::cout << "anchor color" << std::endl;
	std::cout << py_ori_anchor_color << std::endl;
	
	cv::cvtColor(py_ori_anchor_color, py_ori_anchor_color, cv::COLOR_RGB2Lab);


	std::cout << "anchor position" << std::endl;
	std::cout << py_anchor_pos << std::endl;




	return 0;
}


/* @brief upload fixed anchor from python to c++ */
int fix_anchors(const int* fixed_pos, int n) {
	DLL_CHECK(n >= 0);

#ifdef M_DEBUG
	std::cout << "fix anchor number:" << std::endl;
	std::cout << n << std::endl;
#endif // M_DEBUG


	for (int i = 0; i < n; i++) {
		DLL_CHECK(fixed_pos[i] > 0 && fixed_pos[i] < cMapParam::anchorNum - 1);
		cMapBuffer::anchorFixed.push_back(fixed_pos[i] - 1);
	}
	return 0;
}



///* @brief show 'pv' matrix */
//float** py_get_pvs() {
//	float** pv = new float*[py_pv.size()];
//	for (int i = 0; i < py_pv.size(); i++) {
//		pv[i] = (float*)py_pv[i].data;
//	}
//
//	return pv;
//}

/* @brief upload roi from python to c++ */
int input_roi(const int* x, const int* y, int m) {
	DLL_CHECK(m >= 0);

#ifdef M_DEBUG
	std::cout << "roi contour pixels:" << std::endl;
	std::cout << m << std::endl;
#endif // M_DEBUG

	if (0 == m) {
		return -1;
	}

	/* get region from contour */
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Point> contour;
	for (int i = 0; i < m; i++) {
		contour.push_back(cv::Point(x[i], y[i]));
	}
	contours.push_back(contour);


	cv::Mat contourdata(cMapParam::rows, cMapParam::cols, CV_8UC1, cv::Scalar(0));
	cv::drawContours(contourdata, contours, 0, 255, -1);//-1

	cv::findNonZero(contourdata == 255, cMapBuffer::roi);

#ifdef DISP
	cMapUtils::dispImage("roi", contourdata, true);
#endif // DISP



#ifdef M_DEBUG
	std::cout << "roi inner pixels:" << std::endl;
	std::cout << cMapBuffer::roi.size() << std::endl;
#endif // M_DEBUG


	contourdata.release();
	return 0;
}


/* @param rgb colors of background image */
int init_bg_img(const float* bg_img, int rows, int cols) {
	DLL_CHECK(rows == cMapParam::rows
		&& cols == cMapParam::cols);

#ifdef M_DEBUG
	std::cout << "set background image." << std::endl;
#endif // M_DEBUG


	cMapBuffer::bgImg.create(cMapParam::rows, cMapParam::cols, CV_32FC3);//创建三通道彩色图
	std::copy(bg_img, bg_img + cMapParam::rows * cMapParam::cols * 3, (float*)cMapBuffer::bgImg.data);

	//cMapBuffer::bgColorRgb = cMapBuffer::bgImg.at<cv::Vec3f>(0, 0);//原代码
	cMapBuffer::bgColorRgb = cv::Vec3f (255, 255, 255);//修改的
	cv::cvtColor(cMapBuffer::bgImg, cMapBuffer::bgImg, CV_RGB2Lab);
	//cMapBuffer::bgColor = cMapBuffer::bgImg.at<cv::Vec3f>(1, 0);//原代码
	cMapBuffer::bgColor = cv::Vec3f(100, 0, 0);//修改的


	return 0;
}





/* @brief optimization for anchors' position. "anchorPos" must be allocated by the user.
*/
int optimize_anchors(float* anchorPos) {//优化位置
	// Initialize arguments for cuda
	initCudaArgs(cMapParam::rows, cMapParam::cols, cMapBuffer::bgColor);

	// Variables
	std::copy((float*)py_anchor_pos.datastart, (float*)py_anchor_pos.dataend, anchorPos);//Mat.datastart为数据的首位存储地址
	cv::Mat cvAnchorPos(1, cMapParam::varNum, CV_32FC1, (float*)anchorPos + 1);
	

	// Normalize data
	cv::normalize(py_data, py_data, 0, 1, cv::NORM_MINMAX, -1, py_mask);

	cv::Mat bgImageL;
	cv::extractChannel(cMapBuffer::bgImg, bgImageL, 0);

	//

	// 2. Initialze parameter anchors, and consturct a (varNum x 1) colormap

	
	
	// Construct a colormap
	cMap oriColormap(py_anchor_pos, py_ori_anchor_color);

	//std::cout << oriColormap.mAnchorLength << std::endl;//输出原始弧长
	// Compute original anchors' arc lengths

	


	cv::Mat oriAnchorLengths;
	oriAnchorLengths = oriColormap.arcLength(py_anchor_pos);
	
	

	//
	//先创建一个文件
	/*std::ostringstream Fe;
	Fe << cMapParam::gamma;
	std::string F(Fe.str());
	std::ofstream createfile("C:/Users/15324/Desktop/BF/" + dataName + "_" + mapName  + "_diff_norm.txt");
	 bfout.open("C:/Users/15324/Desktop/BF/" + dataName + "_" + mapName + "_diff_norm.txt");*/

	// 3. OPTIMIZATION
	cMapBuffer::initOptBuffer(oriColormap, py_ori_anchor_color, oriAnchorLengths, py_data, py_mask, bgImageL);
	
#ifdef HARD_CONSTRAINT
	cMapOpt::KNOpt(cvAnchorPos);   // Hard constraints as x0 < x1 < x2...
	
#else
	cMapOpt::KNOpt2(cvAnchorPos);  // No hard constraints
#endif // HARD_CONSTRAINT


	/*bfout.close();
	createfile.close();*/
	//


#ifdef LOCAL_RES
	// 4. Display final results
	cv::Mat cvNewAnchorPos;
	cv::hconcat(cv::Mat(1, 1, CV_32FC1, 0.f), cvAnchorPos, cvNewAnchorPos);
	cv::hconcat(cvNewAnchorPos, cv::Mat(1, 1, CV_32FC1, 1.f), cvNewAnchorPos);
	cMap newColormap(cvNewAnchorPos, py_ori_anchor_color);


	cv::cuda::GpuMat py_data_gpu;
	py_data_gpu.upload(py_data);

	cv::cuda::GpuMat diff_weight;
	cMapBuffer::diffWeight.copyTo(diff_weight);


	cv::cuda::GpuMat oriLabData, newLabData;
	cv::cuda::GpuMat ori_local_diff, new_local_diff;

	// Get data
	oriLabData = oriColormap.getColor(py_data_gpu);
	newLabData = newColormap.getColor(py_data_gpu);


	// Get local difference
#if (LOCALDIFF_METRIC == GRADIENT_METRIC)
	ori_local_diff = getLocalDiffGpu(oriLabData, cv::cuda::GpuMat(py_mask));
	new_local_diff = getLocalDiffGpu(newLabData, cv::cuda::GpuMat(py_mask));
	//归一化显示
	cv::cuda::GpuMat ori_normLocalDiff;
	cv::cuda::normalize(ori_local_diff, ori_normLocalDiff, 0, 1, cv::NORM_MINMAX, -1, cv::cuda::GpuMat(py_mask));
	cv::cuda::GpuMat new_normLocalDiff;
	cv::cuda::normalize(new_local_diff, new_normLocalDiff, 0, 1, cv::NORM_MINMAX, -1, cv::cuda::GpuMat(py_mask));
#elif (LOCALDIFF_METRIC == DIFF_76_METRIC)
	ori_local_diff = getLocalDiffGpu76(oriLabData, cv::cuda::GpuMat(py_mask));
	new_local_diff = getLocalDiffGpu76(newLabData, cv::cuda::GpuMat(py_mask));
#else
	ori_local_diff = getLocalDiffGpu2000(oriLabData, cv::cuda::GpuMat(py_mask));
	new_local_diff = getLocalDiffGpu2000(newLabData, cv::cuda::GpuMat(py_mask));
#endif



#ifdef GRAY2COLOR
	/*cv::cuda::GpuMat pv();
	cv::cuda::GpuMat labpv(oriColormap.getColor(pv));*/
	cv::Mat labpv(oriColormap.getColor(cMapBuffer::diffWeight));
	cv::Mat lab_ori_local_diff(oriColormap.getColor(ori_normLocalDiff));
	cv::Mat lab_new_local_diff(oriColormap.getColor(new_normLocalDiff));
#else
	//归一化显示
	cv::cuda::GpuMat new_diff_weight;
	cv::cuda::normalize(cv::Mat(diff_weight), new_diff_weight, 0, 1, cv::NORM_MINMAX, -1, cv::cuda::GpuMat(py_mask));
	cv::Mat graypv = cMapUtils::gray2bgr(cv::Mat(new_diff_weight), py_mask);//原来这里可以强制转换成mat
	cv::Mat gray_ori_local_diff = cMapUtils::gray2bgr(cv::Mat(ori_local_diff), py_mask);
	cv::Mat gray_new_local_diff = cMapUtils::gray2bgr(cv::Mat(new_local_diff), py_mask);
#endif // GRAY2COLOR


#ifdef DISP
	 //display colormap
	cMapUtils::dispColormap("Original colormap", oriColormap.mAnchorColor);
	cMapUtils::dispColormap("New colormap", newColormap.mAnchorColor);

	// display data
	cMapUtils::dispImage("Original color mapped data", oriLabData, false);
	cMapUtils::dispImage("New color mapped data", newLabData, false);

#ifdef GRAY2COLOR
	cMapUtils::dispImage("pv", labpv, false);
	cMapUtils::dispImage("Original local difference", lab_ori_local_diff, false);
	cMapUtils::dispImage("New local difference", lab_new_local_diff, false);
#else
	 //display pv
	cMapUtils::dispImage("pv", graypv, true);
	 /* display local difference
	cMapUtils::dispImage("Original local difference", gray_ori_local_diff, true);
	cMapUtils::dispImage("New local difference", gray_new_local_diff, true);
*/

	
	//cv::waitKey(1000);
#endif // GRAY2COLOR


#ifdef OVERLAY_BG
	// display overlayed images
	cv::Mat oriOverlayImage = cMapUtils::overlay(cv::Mat(oriLabData), py_bg_img, py_mask[i]);
	cMapUtils::dispImage("Original overlaid image", oriOverlayImage, false);

	cv::Mat newOverlayImage = cMapUtils::overlay(cv::Mat(newLabData), py_bg_img, py_mask[i]);
	cMapUtils::dispImage("New overlaid image", newOverlayImage, false);
#endif // OVERLAY_BG


#endif // DISP


#ifdef SAVE
	std::ostringstream ss;
	ss << "α_";
	ss << cMapParam::alpha;
	//ss << "_β_";
	//ss << cMapParam::beta;
	ss << "_γ_";
	ss << cMapParam::gamma;
	ss << "_η_";
	ss << cMapParam::eta;
	ss << "_t1_";
	ss << cMapParam::t1;
	ss << "_t2_";
	ss << cMapParam::t2;
	std::string s(ss.str());

	std::ostringstream ff;
	ff<< cMapParam::gamma;
	std::string f(ff.str());
#pragma region 二维数据
	//std::string prefix = "C:/Users/15324/Desktop/outputs/anchorNum_" + std::to_string(cMapParam::anchorNum) + "_" + dataName + "/";
	std::string prefix = "C:/Users/15324/Desktop/Zeng/";

	// Check for existence
	if (CreateDirectory(prefix.c_str(), NULL) ||
		ERROR_ALREADY_EXISTS == GetLastError())
	{
		std::cout << "Create file folder successful." << std::endl;
	}
	else
	{
		std::cout << "File folder already exists." << std::endl;
	}


	// save new colormap

	std::ofstream fout("C:/Users/15324/Desktop/Zeng_colormap/"+ dataName + "_" + mapName  + "_Zeng_map.txt");

	if (!fout)
	{
		std::cout << "File Not Opened" << std::endl;

	}

	//
	cv::Mat rgb_map;
	cv::cvtColor(newColormap.mAnchorColor, rgb_map, CV_Lab2RGB);//转成rgb
	for (int i = 0; i < rgb_map.cols; i++)
	{
		fout <<rgb_map.at<cv::Vec3f>(0,i)[0] << " "<<rgb_map.at<cv::Vec3f>(0, i)[1] << " "<<rgb_map.at<cv::Vec3f>(0, i)[2];
		fout << std::endl;//换行
	}
	std::cout << "Saving file successfully." << std::endl;
	fout.close();


	//将颜色图复制扩充到20行
	cv::copyMakeBorder(oriColormap.mAnchorColor, oriColormap.mAnchorColor, 0, 19, 0, 0, cv::BORDER_REPLICATE);
	cv::copyMakeBorder(newColormap.mAnchorColor, newColormap.mAnchorColor, 0, 19, 0, 0, cv::BORDER_REPLICATE);


	//cMapUtils::saveImage(prefix  + dataName + "_" + mapName +"_org_map.png", oriColormap.mAnchorColor, false);
	// save new colormap
    cMapUtils::saveImage(prefix + dataName + "_" + mapName+ "_Zeng_map.png", newColormap.mAnchorColor, false);



	// save original data
	//cMapUtils::saveImage(prefix + dataName + "_" + mapName  + "_org.png", oriLabData, false);
	// save new data
	cMapUtils::saveImage(prefix + dataName + "_" + mapName +"_Zeng.png", newLabData, false);
#pragma endregion

	



#ifdef GRAY2COLOR
	cMapUtils::saveImage(prefix + dataName + "_" + mapName + "_pv_" +s+ ".png", labpv, false);
	cMapUtils::saveImage(prefix + dataName + "&" + mapName + "_first_local_diff_" + s + ".png", lab_ori_local_diff, false);
	cMapUtils::saveImage(prefix + dataName + "&" + mapName + "_last_local_diff_" + s + ".png", lab_new_local_diff, false);
#else
	// save pv
	//cMapUtils::saveImage(prefix + dataName + "&" + mapName + "_pv_" +  ".png", graypv, true);
	// save local difference
	//cMapUtils::saveImage(prefix + dataName + "&" + mapName + "_first_local_diff_" + s + ".png", gray_ori_local_diff, true);
	//cMapUtils::saveImage(prefix + dataName + "&" + mapName + "_last_local_diff_" + s + ".png", gray_new_local_diff, true);
#endif // GRAY2COLOR


#ifdef USE_BG_IMG
	// save overlay image
	cMapUtils::saveImage(prefix + dataName + "&" + mapName + "_originaloverlay_" + s + ".png", oriOverlayImage, false);
	cMapUtils::saveImage(prefix + dataName + "&" + mapName + "_newoverlay_" + s + ".png", newOverlayImage, false);
#endif


#endif // SAVE


	//cv::waitKey();


#endif // M_DEBUG

	return 0;
}





