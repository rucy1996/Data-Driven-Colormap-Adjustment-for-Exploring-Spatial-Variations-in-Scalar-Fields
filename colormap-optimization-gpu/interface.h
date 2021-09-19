// This is the interface from any other languages to c.
// Please note  that before calling these funtions, any color related stuff must keep rgb representation.

#ifndef PYINTERFACE_H
#define PYINTERFACE_H

#include <vector>
#include <assert.h>
#include "cMapOpt.h"


// Dll Error handler
#define DLL_CHECK(expr)															\
{																				\
	bool dllSuccess	= expr;																		\
	if (!dllSuccess)																	\
	{																			\
		printf("Argument error at file: %s, function: %s, line: %d ", __FILE__, __FUNCTION__  , __LINE__);	\
		exit(1);																\
	}																			\
}


#define DLLEXPORT extern "C" __declspec(dllexport)

DLLEXPORT int init_data(float* data, int rows, int cols);

DLLEXPORT int init_anchors(const float* anchor_pos, const float* anchor_color, int n);

DLLEXPORT int init_param( float alpha,  float beta,  float gamma,  float eta);

DLLEXPORT int init_magnifier(const int magnitude);

DLLEXPORT int init_bg_img(const float* bg_img, int rows, int cols);

DLLEXPORT int init_threshold( float lt,  float ut);

DLLEXPORT int fix_anchors(const int* fixed_pos, int n);

DLLEXPORT int input_roi(const int* x, const int* y, int m);

DLLEXPORT int optimize_anchors(float* anchors);



#endif // !PYINTERFACE_H