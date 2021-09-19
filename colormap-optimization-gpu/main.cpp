// Invoke colormap-optimization example

#include "stdafx.h"
#include "interface.h"


std::string dataName = "D3";//dose-NaN-bigger//s75_part
std::string dataPath = "D:/insert_color_(��Ŀ�ģ�/��Ŀ��_(ȷ��)/data/" + dataName + ".txt";//E:/colormap��Ŀ����wyq/data/
//std::string dataPath = "E:/colormap��Ŀ����wyq/data/" + dataName + ".txt";

//const std::string bgPath = "E:/colormap��Ŀ����wyq/data/lung-ct.txt";

 std::string mapName = "viridis";
 std::string mapPath = "D:/insert_color_(��Ŀ�ģ�/��Ŀ��_(ȷ��)/colormaps/" + mapName + ".txt";
// std::string mapPath = "E:/extracMap/colormaps/" + mapName + ".txt";

 float grads_threshold = 0.1;//�ݶ���ֵ
 int Ratio_bins =255;
 int count=0;
 float gradMean;

 float new_eta;
 float new_t1;

  int isGPU = 1;//Ϊ1����GPU�汾��Ϊ0����cpu�汾

 std::ofstream bfout;//�ļ���
 std::string temp;
const int signal =0;//signal=0�Ǵ������ļ�������colormap��Ϊ���������Ǵ������ļ������colormap
int main()
{
	std::cout << "CUDA version." << std::endl;                     
	if (signal == 0) {
#pragma region �����ļ�����


		cv::Mat data;
		cMapUtils::Load_Txt_File(dataPath, data);//����ͼ������
												 //data += 1000.f;//��ֹ���ݾ�������

		cv::Mat inAnchorColor;
		cMapUtils::Load_Txt_File(mapPath, inAnchorColor, 3);//������ɫͼ
															//std::cout << "inAnchorColor.rows:" << inAnchorColor.rows << std::endl;
															//cv::flip(inAnchorColor,inAnchorColor, - 1);//������ɫ��ķ�ת
															//����ʼ����ת����float��
		float* c_data = (float*)data.data;//Mat::data��Ĭ������Ϊuchar*�����ܶ�ʱ����Ҫ�����������ͣ���float��int����ʱ��Ҫ��dataǿ������ת��

										  /* initialize readin data */
		init_data(c_data, data.rows, data.cols);

		const int c_varNum =4;//������Ƶ�
		const int c_anchorNum = c_varNum + 2;

		cv::Mat anchorPos;
		anchorPos = cMapUtils::initAnchorPos(0.f, 1.f, c_anchorNum);//������0-1֮��
		//anchorPos = (cv::Mat_<float>(9, 1) << 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.825, 1);
		//anchorPos = (cv::Mat_<float>(9, 1) << 0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0);
		//anchorPos = (cv::Mat_<float>(9, 1) << 0, 0.0680625, 0.133496, 0.185437, 0.237379, 0.307365, 0.388796, 0.543048, 1);



		cv::Mat anchorColor(anchorPos.rows, 1, CV_32FC3);//��ʼ����ɫ���λ��
		for (int i = 0; i < anchorPos.rows; i++) {
			anchorColor.at<cv::Vec3f>(i) = inAnchorColor.at<cv::Vec3f>((int)round(anchorPos.at<float>(i) * (BIN_NUM - 1)));
		}

		//std::cout << anchorColor.size();

		//����ֱ��ͼ������ʼ��
		//cv::Mat mask;
		//mask = (data == data);
		//cv::normalize(data, data, 0, 1, cv::NORM_MINMAX, -1, mask);
		//cv::Mat Hist = cMapUtils::getHist(data, c_anchorNum-1);// c_anchorNum-1Ϊ������
		//cv::vconcat(cv::Mat(1, 1, CV_32FC1, 0.f), Hist, Hist);
		//cv::Mat newAnchorPos = cMapUtils::get_histpos(anchorPos, Hist);//��ԭʼֱ��ͼ����������µ�λ��
		//
		//newAnchorPos.copyTo(anchorPos);//���Ƹ���ʼλ��
		//std::cout << " newAnchorPos :" << anchorPos << std::endl;
		

		//������
	/*	for(int i=1;i<=10;i++)
			for (int j = 1; j <=15; j++)
			{*/



				/* initialize balance factor */
				// alpha defualt to 1. Users can only set gamma
				const float _alpha_ = 1.f;
				const float _gamma_ = 0.0001f;//  0.0001f
				float eta =5;
				init_param(_alpha_, 0.f, _gamma_*_alpha_, eta); //mandlebrot 100.f //Ĭ��0.01f

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
				c_anchorPos = (float*)anchorPos.data;//ǿ��ת����float����


				float *c_anchorColor;
				c_anchorColor = (float*)anchorColor.data;////ǿ��ת����float����


														 /* initialize initial anchors' position and colors */
				init_anchors(c_anchorPos, c_anchorColor, c_anchorNum);


				cv::Mat bgImage(data.rows, data.cols, CV_32FC1, cv::Scalar(0.f));//scalar�ǽ�ͼ�����óɵ�һ�ҶȺ���ɫ,Ϊ0���ǽ�ͼ���ɺ�ɫ


				cv::Mat bgImageColor;
				cv::cvtColor(bgImage, bgImageColor, cv::COLOR_GRAY2RGB);//��ɫ�ռ�ת��
				init_bg_img((float*)bgImageColor.data, bgImageColor.rows, bgImageColor.cols);

				// fix position of given anchors 
				/*int c_anchorFixed[1] = { 4};
				fix_anchors(c_anchorFixed, 1);*/
				clock_t startTime, endTime;
				startTime = clock();
						optimize_anchors(c_anchorPos);
				endTime = clock();
				for (int i = 0; i < c_anchorNum; i++) {//������Ƶ��λ��
					std::cout << c_anchorPos[i] << std::endl;
				}
				std::cout << "The whole optimzation use time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
			//}
		cv::waitKey(0);
#pragma endregion
	}
	else {
#pragma region �������������colormap��Ȼ����й�һ��

		//std::string data_path = "D:/��Ŀһ�޸�/��Ŀһ�޸�-�հ�/data/";
		std::string data_path = "D:/insert_color_(��Ŀ�ģ�/��Ŀ��_(ȷ��)/data/";
		std::vector<cv::String> data_filenames;
		cv::glob(data_path + "*.txt", data_filenames, false);
		int data_lenfiles = data_filenames.size();//��ȡ���ļ����µ�����


		//std::string map_path = "D:/��Ŀһ�޸�/��Ŀһ�޸�-�հ�/colormaps/";
		std::string map_path = "D:/insert_color_(��Ŀ�ģ�/��Ŀ��_(ȷ��)/colormaps/";
		std::vector<cv::String> map_filenames;
		cv::glob(map_path + "*.txt", map_filenames, false);
		int map_lenfiles = map_filenames.size();//��ȡ���ļ����µ�����

		std::cout << data_lenfiles << " " << map_lenfiles << std::endl;


		//�ȴ���һ���ļ�
		
		std::ofstream createfile("C:/Users/15324/Desktop/Zeng-GPU.txt");
		bfout.open("C:/Users/15324/Desktop/Zeng-GPU.txt");


		for (int i = 0; i < data_lenfiles; i++)
			for (int j = 0; j < map_lenfiles; j++)
			{
				cv::Mat data;//��������
				cMapUtils::Load_Txt_File(data_filenames[i], data);//���س�ʼ�ļ�����
				int start_bg = strlen(data_path.c_str());
				int end_bg = strlen(data_filenames[i].c_str()) - 4;
				 dataName=data_filenames[i].substr(start_bg, end_bg - start_bg);//��ȡ���ݵ�����


				cv::Mat inAnchorColor;//������ɫ��ľ���
				cMapUtils::Load_Txt_File(map_filenames[j], inAnchorColor,3);//���س�ʼ�ļ�����
				 start_bg = strlen(map_path.c_str());
				 end_bg = strlen(map_filenames[j].c_str()) - 4;
				 mapName = map_filenames[j].substr(start_bg, end_bg - start_bg);//��ȡ��ɫͼ������
				
				
#pragma region ��������

				float* c_data = (float*)data.data;//Mat::data��Ĭ������Ϊuchar*�����ܶ�ʱ����Ҫ�����������ͣ���float��int����ʱ��Ҫ��dataǿ������ת��

												  /* initialize readin data */
				init_data(c_data, data.rows, data.cols);

				const int c_varNum =4;//������Ƶ�
				const int c_anchorNum = c_varNum + 2;

				cv::Mat anchorPos;
				anchorPos = cMapUtils::initAnchorPos(0.f, 1.f, c_anchorNum);//������0-1֮��
				//anchorPos = (cv::Mat_<float>(9, 1) << 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.825, 1);


				cv::Mat anchorColor(anchorPos.rows, 1, CV_32FC3);//��ʼ����ɫ���λ��
				for (int i = 0; i < anchorPos.rows; i++) {
					anchorColor.at<cv::Vec3f>(i) = inAnchorColor.at<cv::Vec3f>((int)round(anchorPos.at<float>(i) * (BIN_NUM - 1)));
				}
				
				//����ֱ��ͼ������ʼ��
				//cv::Mat mask;
				//mask = (data == data);
				//cv::normalize(data, data, 0, 1, cv::NORM_MINMAX, -1, mask);
				//cv::Mat Hist = cMapUtils::getHist(data, c_anchorNum - 1);// c_anchorNum-1Ϊ������
				//cv::vconcat(cv::Mat(1, 1, CV_32FC1, 0.f), Hist, Hist);
				//cv::Mat newAnchorPos = cMapUtils::get_histpos(anchorPos, Hist);//��ԭʼֱ��ͼ����������µ�λ��

				//newAnchorPos.copyTo(anchorPos);//���Ƹ���ʼλ��
				//							   //std::cout << " newAnchorPos :" << anchorPos << std::endl



				float* c_anchorPos;
				c_anchorPos = (float*)anchorPos.data;//ǿ��ת����float����

				float *c_anchorColor;
				c_anchorColor = (float*)anchorColor.data;////ǿ��ת����float����

														 /* initialize initial anchors' position and colors */
				init_anchors(c_anchorPos, c_anchorColor, c_anchorNum);



				/* initialize balance factor */
				// alpha defualt to 1. Users can only set gamma
				const float _alpha_ = 1.f;
				const float _gamma_ =0.0001f;
				init_param(_alpha_, 0.f, _gamma_*_alpha_, 5.f); //mandlebrot 100.f //Ĭ��0.01f

																  /* scale thresholds */
				init_threshold(2.9f, 300.f);

				/* ... */
				init_magnifier(100); //100 //10 //1


									 /* initialize background image */
									 /*cv::Mat bgImage;
									 cMapUtils::Load_Txt_File(bgPath, bgImage);
									 cv::normalize(bgImage, bgImage, 0, 1, cv::NORM_MINMAX, -1);*/
				cv::Mat bgImage(data.rows, data.cols, CV_32FC1, cv::Scalar(0.f));//scalar�ǽ�ͼ�����óɵ�һ�ҶȺ���ɫ,Ϊ0���ǽ�ͼ���ɺ�ɫ


				cv::Mat bgImageColor;
				cv::cvtColor(bgImage, bgImageColor, cv::COLOR_GRAY2RGB);//��ɫ�ռ�ת��
				init_bg_img((float*)bgImageColor.data, bgImageColor.rows, bgImageColor.cols);



				
				clock_t startTime, endTime;
				startTime = clock();
				optimize_anchors(c_anchorPos);
				endTime = clock();

				float times = (double)(endTime - startTime) / CLOCKS_PER_SEC;
				std::cout << "The whole optimzation use time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
				bfout << dataName << " " << mapName << " " << times;
				bfout << std::endl;//����
#pragma endregion



			}
			std::cout << "Saving file successfully." << std::endl;
			bfout.close();
#pragma endregion
	}
}

	



