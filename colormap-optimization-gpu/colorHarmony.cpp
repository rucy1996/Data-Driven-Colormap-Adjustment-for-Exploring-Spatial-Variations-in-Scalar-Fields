#include "colorHarmony.h"

//region1Arcs is degree of region1 of templates
int harmony::region1Arcs[8] = { 18, 94, 18, 18, 180, 94, 94, 0 };
//region2Arcs is degree of region2 of templates
int harmony::region2Arcs[8] = { 0,  0, 80, 18,   0, 18, 94, 0 };
//region2shift is arc length between two sector center
int harmony::region2Shift[8] = { 0, 0, 270, 180, 0, 180, 180, 0 };
// template name
char harmony::names[8] = { 'i', 'V', 'L', 'I', 'T', 'Y', 'X', 'N' };