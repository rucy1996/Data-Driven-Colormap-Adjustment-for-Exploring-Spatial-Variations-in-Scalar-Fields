// Invoke colormap-optimization example

#include "stdafx.h"
#include "interface.h"


std::string dataName = "D3";//dose-NaN-bigger//s75_part
std::string dataPath = "D:/insert_color_(项目四）/项目四_(确定)/data/" + dataName + ".txt";//E:/colormap项目整理wyq/data/
//std::string dataPath = "E:/colormap项目整理wyq/data/" + dataName + ".txt";

//const std::string bgPath = "E:/colormap项目整理wyq/data/lung-ct.txt";

 std::string mapName = "viridis";
 std::string mapPath = "D:/insert_color_(项目四）/项目四_(确定)/colormaps/" + mapName + ".txt";
// std::string mapPath = "E:/extracMap/colormaps/" + mapName + ".txt";

 float grads_threshold = 0.1;//梯度阈值
 int Ratio_bins =255;
 int count=0;
 float gradMean;

 float new_eta;
 float new_t1;

  int isGPU = 1;//为1就是GPU版本，为0就是cpu版本

 std::ofstream bfout;//文件流
 std::string temp;
const int signal =0;//signal=0是处理单个文件，单个colormap，为其他数字是处理多个文件，多个colormap
int main()
{
	std::cout << "CUDA version." << std::endl;                     
	if (signal == 0) {
#pragma region 单个文件处理


		cv::Mat data;
		cMapUtils::Load_Txt_File(dataPath, data);//加载图像数据
												 //data += 1000.f;//防止数据精度问题

		cv::Mat inAnchorColor;
		cMapUtils::Load_Txt_File(mapPath, inAnchorColor, 3);//加载颜色图
															//std::cout << "inAnchorColor.rows:" << inAnchorColor.rows << std::endl;
															//cv::flip(inAnchorColor,inAnchorColor, - 1);//进行颜色表的翻转
															//将初始数据转换成float型
		float* c_data = (float*)data.data;//Mat::data的默认类型为uchar*，但很多时候需要处理其它类型，如float、int，此时需要将data强制类型转换

										  /* initialize readin data */
		init_data(c_data, data.rows, data.cols);

		const int c_varNum =4;//定义控制点
		const int c_anchorNum = c_varNum + 2;

		cv::Mat anchorPos;
		anchorPos = cMapUtils::initAnchorPos(0.f, 1.f, c_anchorNum);//控制在0-1之间
		//anchorPos = (cv::Mat_<float>(9, 1) << 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.825, 1);
		//anchorPos = (cv::Mat_<float>(9, 1) << 0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0);
		//anchorPos = (cv::Mat_<float>(9, 1) << 0, 0.0680625, 0.133496, 0.185437, 0.237379, 0.307365, 0.388796, 0.543048, 1);



		cv::Mat anchorColor(anchorPos.rows, 1, CV_32FC3);//初始化颜色表的位置
		for (int i = 0; i < anchorPos.rows; i++) {
			anchorColor.at<cv::Vec3f>(i) = inAnchorColor.at<cv::Vec3f>((int)round(anchorPos.at<float>(i) * (BIN_NUM - 1)));
		}

		//std::cout << anchorColor.size();

		//根据直方图来求解初始点
		//cv::Mat mask;
		//mask = (data == data);
		//cv::normalize(data, data, 0, 1, cv::NORM_MINMAX, -1, mask);
		//cv::Mat Hist = cMapUtils::getHist(data, c_anchorNum-1);// c_anchorNum-1为箱子数
		//cv::vconcat(cv::Mat(1, 1, CV_32FC1, 0.f), Hist, Hist);
		//cv::Mat newAnchorPos = cMapUtils::get_histpos(anchorPos, Hist);//用原始直方图计算出来的新的位置
		//
		//newAnchorPos.copyTo(anchorPos);//复制给初始位置
		//std::cout << " newAnchorPos :" << anchorPos << std::endl;
		

		//批处理
	/*	for(int i=1;i<=10;i++)
			for (int j = 1; j <=15; j++)
			{*/



				/* initialize balance factor */
				// alpha defualt to 1. Users can only set gamma
				const float _alpha_ = 1.f;
				const float _gamma_ = 0.0001f;//  0.0001f
				float eta =5;
				init_param(_alpha_, 0.f, _gamma_*_alpha_, eta); //mandlebrot 100.f //默认0.01f

											/* scale thresholds */
				init_threshold(2.9, 300.f);

				/* ... */
				init_magnifier(490); //100 //10 //1


				 // Velocity----leg
				/*int c_roi_x[] = { 84, 81, 81, 92, 104, 121, 137, 152, 166, 175, 177, 176, 166, 155, 146, 132, 117, 106, 96, 87 };
				int c_roi_y[] = { 273, 261, 235, 208, 192, 180, 172, 167, 168, 175, 188, 208, 236, 260, 272, 282, 288, 289, 286, 280 };
				input_roi(c_roi_x, c_roi_y, 20);*/


				// dose----core
				/*int c_roi_x[] = { 344, 340, 342, 349, 356, 372, 381, 403, 417, 496, 519, 537, 555, 603, 635, 646, 642, 628, 617, 608, 585, 562, 553, 519, 494, 476, 406, 385, 374, 363, 356, 349};
				int c_roi_y[] = { 650, 589, 576, 555, 542, 524, 515, 501, 494, 476, 479, 483, 490, 512, 544, 564, 605, 648, 666, 679, 707, 729, 736, 756, 763, 763, 754, 740, 731, 713, 698, 679 };
				input_roi(c_roi_x, c_roi_y, 32);*/



				float* c_anchorPos;
				c_anchorPos = (float*)anchorPos.data;//强制转换成float类型


				float *c_anchorColor;
				c_anchorColor = (float*)anchorColor.data;////强制转换成float类型


														 /* initialize initial anchors' position and colors */
				init_anchors(c_anchorPos, c_anchorColor, c_anchorNum);


				cv::Mat bgImage(data.rows, data.cols, CV_32FC1, cv::Scalar(0.f));//scalar是将图像设置成单一灰度和颜色,为0就是将图像变成黑色


				cv::Mat bgImageColor;
				cv::cvtColor(bgImage, bgImageColor, cv::COLOR_GRAY2RGB);//颜色空间转化
				init_bg_img((float*)bgImageColor.data, bgImageColor.rows, bgImageColor.cols);

				// fix position of given anchors 
				/*int c_anchorFixed[1] = { 4};
				fix_anchors(c_anchorFixed, 1);*/
				clock_t startTime, endTime;
				startTime = clock();
						optimize_anchors(c_anchorPos);
				endTime = clock();
				for (int i = 0; i < c_anchorNum; i++) {//输出控制点的位置
					std::cout << c_anchorPos[i] << std::endl;
				}
				std::cout << "The whole optimzation use time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
			//}
		cv::waitKey(0);
#pragma endregion
	}
	else {
#pragma region 读入所有数组和colormap，然后进行归一化

		//std::string data_path = "D:/项目一修改/项目一修改-终版/data/";
		std::string data_path = "D:/insert_color_(项目四）/项目四_(确定)/data/";
		std::vector<cv::String> data_filenames;
		cv::glob(data_path + "*.txt", data_filenames, false);
		int data_lenfiles = data_filenames.size();//获取该文件夹下的数量


		//std::string map_path = "D:/项目一修改/项目一修改-终版/colormaps/";
		std::string map_path = "D:/insert_color_(项目四）/项目四_(确定)/colormaps/";
		std::vector<cv::String> map_filenames;
		cv::glob(map_path + "*.txt", map_filenames, false);
		int map_lenfiles = map_filenames.size();//获取该文件夹下的数量

		std::cout << data_lenfiles << " " << map_lenfiles << std::endl;


		//先创建一个文件
		
		std::ofstream createfile("C:/Users/15324/Desktop/Zeng-GPU.txt");
		bfout.open("C:/Users/15324/Desktop/Zeng-GPU.txt");


		for (int i = 0; i < data_lenfiles; i++)
			for (int j = 0; j < map_lenfiles; j++)
			{
				cv::Mat data;//建立数组
				cMapUtils::Load_Txt_File(data_filenames[i], data);//加载初始文件数据
				int start_bg = strlen(data_path.c_str());
				int end_bg = strlen(data_filenames[i].c_str()) - 4;
				 dataName=data_filenames[i].substr(start_bg, end_bg - start_bg);//获取数据的名字


				cv::Mat inAnchorColor;//储存颜色表的矩阵
				cMapUtils::Load_Txt_File(map_filenames[j], inAnchorColor,3);//加载初始文件数据
				 start_bg = strlen(map_path.c_str());
				 end_bg = strlen(map_filenames[j].c_str()) - 4;
				 mapName = map_filenames[j].substr(start_bg, end_bg - start_bg);//获取颜色图的名字
				
				
#pragma region 进行运算

				float* c_data = (float*)data.data;//Mat::data的默认类型为uchar*，但很多时候需要处理其它类型，如float、int，此时需要将data强制类型转换

												  /* initialize readin data */
				init_data(c_data, data.rows, data.cols);

				const int c_varNum =4;//定义控制点
				const int c_anchorNum = c_varNum + 2;

				cv::Mat anchorPos;
				anchorPos = cMapUtils::initAnchorPos(0.f, 1.f, c_anchorNum);//控制在0-1之间
				//anchorPos = (cv::Mat_<float>(9, 1) << 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.825, 1);


				cv::Mat anchorColor(anchorPos.rows, 1, CV_32FC3);//初始化颜色表的位置
				for (int i = 0; i < anchorPos.rows; i++) {
					anchorColor.at<cv::Vec3f>(i) = inAnchorColor.at<cv::Vec3f>((int)round(anchorPos.at<float>(i) * (BIN_NUM - 1)));
				}
				
				//根据直方图来求解初始点
				//cv::Mat mask;
				//mask = (data == data);
				//cv::normalize(data, data, 0, 1, cv::NORM_MINMAX, -1, mask);
				//cv::Mat Hist = cMapUtils::getHist(data, c_anchorNum - 1);// c_anchorNum-1为箱子数
				//cv::vconcat(cv::Mat(1, 1, CV_32FC1, 0.f), Hist, Hist);
				//cv::Mat newAnchorPos = cMapUtils::get_histpos(anchorPos, Hist);//用原始直方图计算出来的新的位置

				//newAnchorPos.copyTo(anchorPos);//复制给初始位置
				//							   //std::cout << " newAnchorPos :" << anchorPos << std::endl



				float* c_anchorPos;
				c_anchorPos = (float*)anchorPos.data;//强制转换成float类型

				float *c_anchorColor;
				c_anchorColor = (float*)anchorColor.data;////强制转换成float类型

														 /* initialize initial anchors' position and colors */
				init_anchors(c_anchorPos, c_anchorColor, c_anchorNum);



				/* initialize balance factor */
				// alpha defualt to 1. Users can only set gamma
				const float _alpha_ = 1.f;
				const float _gamma_ =0.0001f;
				init_param(_alpha_, 0.f, _gamma_*_alpha_, 5.f); //mandlebrot 100.f //默认0.01f

																  /* scale thresholds */
				init_threshold(2.9f, 300.f);

				/* ... */
				init_magnifier(100); //100 //10 //1


									 /* initialize background image */
									 /*cv::Mat bgImage;
									 cMapUtils::Load_Txt_File(bgPath, bgImage);
									 cv::normalize(bgImage, bgImage, 0, 1, cv::NORM_MINMAX, -1);*/
				cv::Mat bgImage(data.rows, data.cols, CV_32FC1, cv::Scalar(0.f));//scalar是将图像设置成单一灰度和颜色,为0就是将图像变成黑色


				cv::Mat bgImageColor;
				cv::cvtColor(bgImage, bgImageColor, cv::COLOR_GRAY2RGB);//颜色空间转化
				init_bg_img((float*)bgImageColor.data, bgImageColor.rows, bgImageColor.cols);



				
				clock_t startTime, endTime;
				startTime = clock();
				optimize_anchors(c_anchorPos);
				endTime = clock();

				float times = (double)(endTime - startTime) / CLOCKS_PER_SEC;
				std::cout << "The whole optimzation use time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
				bfout << dataName << " " << mapName << " " << times;
				bfout << std::endl;//换行
#pragma endregion



			}
			std::cout << "Saving file successfully." << std::endl;
			bfout.close();
#pragma endregion
	}
}

	



