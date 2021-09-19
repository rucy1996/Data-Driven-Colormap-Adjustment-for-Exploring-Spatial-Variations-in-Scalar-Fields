#pragma once
#include <stdio.h>
#include <tchar.h>
#include <string>
#include <chrono>

#define M_INF 1 << 30

#define BIN_NUM 255

#define M_DEBUG
#define LOCAL_RES
//#define DISP
//#define SAVE
//#define GRAY2COLOR//画pv图
//#define OVERLAY_BG

// Determine which CIE76 metric to use 
#define CONTRAST_METRIC		LUMN_METRIC
	#define LUMN_METRIC			0
	#define CIE76_METRIC		1
	#define CIEDE2000_METRIC	2

// Determine which local difference metric to use 
#define LOCALDIFF_METRIC	GRADIENT_METRIC
	#define GRADIENT_METRIC		3
	#define DIFF_76_METRIC		4
	#define DIFF_2000_METRIC	5

// Determine which function in exp() to use
#define EXP_INNER			ABS_INNER
	#define ABS_INNER			6
	#define POW_INNER			7



#define HARD_CONSTRAINT
#define LOCAL_DIFF_NORM			//注释与不注释有影响

/*
Algorithm built-in parameters
*/
extern const float epsilon;


extern const int N; //201


extern  std::string dataName;
extern  std::string dataPath;
extern  std::string bgPath;

extern std::ofstream bfout;//记录boundary和fidelity的数值
extern  std::string temp;//记录***的数值

extern  std::string mapName;
extern  std::string mapPath;

extern float gradMean;
extern int count;
extern int Ratio_bins;
extern float grads_threshold;

extern float new_eta;
extern float new_t1;
extern int isGPU;