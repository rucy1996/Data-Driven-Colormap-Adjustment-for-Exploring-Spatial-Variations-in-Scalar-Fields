#ifndef COLOR_HARMONY_H
#define COLOR_HARMONY_H

#include "utils.h"


class harmony {
private:
	//region1Arcs is degree of region1 of templates
	static int region1Arcs[8];
	//region2Arcs is degree of region2 of templates
	static int region2Arcs[8];
	//region2shift is arc length between two sector center
	static int region2Shift[8];
	// template name
	static char names[8];


	static float computeDistance(cv::Vec3f* image, int size, int id) {

		//int resultArc = 0;
		clock_t startTime, endTime;

		startTime = clock();
		float resultDistance = helpComputeDistance(image, size, 0, id);
		endTime = clock();
		//std::cout << "helpComputeDistance use time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

		// For each angle
		for (int i = 1; i < 360; i++) {
			float d = helpComputeDistance(image, size, i, id);
			if (d < resultDistance) {
				//resultArc = i;
				resultDistance = d;
			}
		}

		return resultDistance;
	}
	// image is represented as hsv colors
	static float helpComputeDistance(cv::Vec3f* image, int size, int arc, int id) {
		float dis = 0;
		for (int i = 0; i < size; i++) {
			float hue = image[i][0];
			float s = image[i][1];

			//compute F measuring the harmony of image under scheme(id,arc)
			dis += computeArcDistance(arc, (int)hue, id) * s;
		}
		return dis;
	}



	static float computeDistanceFast(int** hsHist, int id) {

		//int resultArc = 0;
		clock_t startTime, endTime;

		startTime = clock();
		float resultDistance = helpComputeDistanceFast(hsHist, 0, id);
		endTime = clock();
		//std::cout << "helpComputeDistance use time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

		// For each angle. cuda...
		for (int i = 1; i < 360; i++) {
			float d = helpComputeDistanceFast(hsHist, i, id);
			if (d < resultDistance) {
				//resultArc = i;
				resultDistance = d;
			}
		}

		return resultDistance;
	}

	static float computeDistanceFastGpu(int** hsHist, int id) {

		float resultDistance = helpComputeDistanceFast(hsHist, 0, id);
		// For each angle.
		for (int i = 1; i < 360; i++) {
			float d = helpComputeDistanceFast(hsHist, i, id);
			if (d < resultDistance) {
				//resultArc = i;
				resultDistance = d;
			}
		}

		return resultDistance;
	}

	static float helpComputeDistanceFast(int** hsHist, int arc, int id) {
		float dis = 0;
		// cuda...
		for (int i = 0; i < 360; i++) {
			for (int j = 0; j < 256; j++) {
				//compute F measuring the harmony of image under scheme(id,arc)
				int cnt = hsHist[i][j];
				dis += cnt * computeArcDistance(arc, i, id) * j;
			}
		}
		return dis;
	}


	static float computeArcDistance(int arc, int hue, int id) {
		int dis = 0;
		// use border1 as zero degree;
		if (region1Arcs[id] != 0) {
			int border1 = (arc - region1Arcs[id] / 2 + 360) % 360;
			int border2 = region1Arcs[id];
			int shiftHue = (hue - border1 + 360) % 360; //represent hue using border1 as  zero degree
			if (shiftHue < border2)                     //hues that inside the sector
				return 0;
			border1 = 0;
			int d1 = nearestDistance(border1, shiftHue);
			int d2 = nearestDistance(border2, shiftHue);
			dis = d1 < d2 ? d1 : d2;
		}
		if (region2Arcs[id] != 0) {
			int border1 = (arc + region2Shift[id] - region2Arcs[id] / 2 + 360) % 360;
			int border2 = region2Arcs[id];
			int shiftHue = (hue - border1 + 360) % 360;
			if (shiftHue < border2)
				return 0;
			border1 = 0;
			int d1 = nearestDistance(border1, shiftHue);
			int d2 = nearestDistance(border2, shiftHue);
			int dis2 = d1 < d2 ? d1 : d2;
			if (dis2 < dis)
				dis = dis2;
		}
		return (float)dis;
	}

	static int nearestDistance(int hue1, int hue2) {
		int d = (hue1 - hue2 + 360) % 360;
		if (d > 180)
			d = 360 - d;
		return d;
	}

public:
	static float computeHarmonyScore(const cv::Mat& rawImage) {

		clock_t startTime, endTime;

		cv::Vec3f* image;
		image = (cv::Vec3f*)rawImage.data;

		int size = (int)rawImage.total();

		startTime = clock();
		float score = computeDistance(image, size, 0);
		endTime = clock();
		//std::cout << "compute distance use time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

		for (int i = 1; i < 7; i++) {
			startTime = clock();
			float tmpscore = computeDistance(image, size, i);
			endTime = clock();
			//std::cout << "compute distance use time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

			if (tmpscore <score) {
				score = tmpscore;
			}
		}

		return score;
	}

	// Use hue histogram to accelerate computation
	static float computeHarmonyScoreFast(const cv::Mat& rawImage) {
		clock_t startTime, endTime;


		cv::Vec3f* image;
		image = (cv::Vec3f*)rawImage.data;

		// Compute hue and saturation histogram
		int size = (int)rawImage.total();
		int** hsHist = computeHsHist(image, size);

		startTime = clock();
		float score = computeDistanceFast(hsHist, 0);
		endTime = clock();

		for (int i = 1; i < 7; i++) {

			startTime = clock();
			float tmpscore = computeDistanceFast(hsHist, i);
			endTime = clock();
			//std::cout << "compute distance use time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

			if (tmpscore <score) {
				score = tmpscore;
			}
		}

		return score;
	}

	// Use hue histogram to accelerate computation on GPU
	static float computeHarmonyScoreFastGpu(const cv::cuda::GpuMat& rawImage) {

		clock_t startTime, endTime;
		startTime = clock();

		cv::Mat rawImage_(rawImage);

		cv::Vec3f* image;
		image = (cv::Vec3f*)rawImage_.data;

		// Compute hue and saturation histogram
		int size = rawImage_.rows * rawImage_.cols;
		int** hsHist = computeHsHist(image, size);

		float score = computeDistanceFastGpu(hsHist, 0);
		for (int i = 1; i < 7; i++) {
			float tmpscore = computeDistanceFastGpu(hsHist, i);
			if (tmpscore <score) {
				score = tmpscore;
			}
		}
		endTime = clock();
		std::cout << "Harmony score use time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
		std::cout << "Harmony score: " << score << std::endl;
		return score;
	}

	// Compute hue and saturation histogram
	static int** computeHsHist(const cv::Vec3f* image, int size) {

		int **hsHist;
		hsHist = new int*[360];
		for (int i = 0; i < 360; i++) {
			hsHist[i] = new int[256];
			for (int j = 0; j < 256; j++) {
				hsHist[i][j] = 0;
			}
		}

		for (int i = 0; i < size; i++) {
			cv::Vec3f hsv = image[i];
			hsHist[(int)hsv[0]][(int)(hsv[1] * 255)]++;
		}

		return hsHist;
	}

};
#endif // !COLOR_HARMONY_H
