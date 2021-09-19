/*
 * CIEDE2000.cpp
 * Part of http://github.com/gfiumara/CIEDE2000 by Gregory Fiumara.
 * See LICENSE for details.
 */

#include <cmath>
#include "myColor.h"

/*******************************************************************************
 * Conversions.
 ******************************************************************************/
 
constexpr double
MyColor::deg2Rad(
    const double deg)
{
	return (deg * (M_PI / 180.0));
}

constexpr double
MyColor::rad2Deg(
    const double rad)
{
	return ((180.0 / M_PI) * rad);
}

double
MyColor::CIEDE2000(
    const LAB &lab1,
    const LAB &lab2)
{
	/* 
	 * "For these and all other numerical/graphical 􏰀delta E00 values
	 * reported in this article, we set the parametric weighting factors
	 * to unity(i.e., k_L = k_C = k_H = 1.0)." (Page 27).
	 */
	const double k_L = 1.0, k_C = 1.0, k_H = 1.0;
	const double deg360InRad = MyColor::deg2Rad(360.0);
	const double deg180InRad = MyColor::deg2Rad(180.0);
	const double pow25To7 = 6103515625.0; /* pow(25, 7) */
	
	/*
	 * Step 1 
	 */
	/* Equation 2 */
	double C1 = sqrt((lab1[1] * lab1[1]) + (lab1[2] * lab1[2]));
	double C2 = sqrt((lab2[1] * lab2[1]) + (lab2[2] * lab2[2]));
	/* Equation 3 */
	double barC = (C1 + C2) / 2.0;
	/* Equation 4 */
	double G = 0.5 * (1 - sqrt(pow(barC, 7) / (pow(barC, 7) + pow25To7)));
	/* Equation 5 */
	double a1Prime = (1.0 + G) * lab1[1];
	double a2Prime = (1.0 + G) * lab2[1];
	/* Equation 6 */
	double CPrime1 = sqrt((a1Prime * a1Prime) + (lab1[2] * lab1[2]));
	double CPrime2 = sqrt((a2Prime * a2Prime) + (lab2[2] * lab2[2]));
	/* Equation 7 */
	double hPrime1;
	if (lab1[2] == 0 && a1Prime == 0)
		hPrime1 = 0.0;
	else {
		hPrime1 = atan2(lab1[2], a1Prime);
		/* 
		 * This must be converted to a hue angle in degrees between 0 
		 * and 360 by addition of 2􏰏 to negative hue angles.
		 */
		if (hPrime1 < 0)
			hPrime1 += deg360InRad;
	}
	double hPrime2;
	if (lab2[2] == 0 && a2Prime == 0)
		hPrime2 = 0.0;
	else {
		hPrime2 = atan2(lab2[2], a2Prime);
		/* 
		 * This must be converted to a hue angle in degrees between 0 
		 * and 360 by addition of 2􏰏 to negative hue angles.
		 */
		if (hPrime2 < 0)
			hPrime2 += deg360InRad;
	}
	
	/*
	 * Step 2
	 */
	/* Equation 8 */
	double deltaLPrime = lab2[0] - lab1[0];
	/* Equation 9 */
	double deltaCPrime = CPrime2 - CPrime1;
	/* Equation 10 */
	double deltahPrime;
	double CPrimeProduct = CPrime1 * CPrime2;
	if (CPrimeProduct == 0)
		deltahPrime = 0;
	else {
		/* Avoid the fabs() call */
		deltahPrime = hPrime2 - hPrime1;
		if (deltahPrime < -deg180InRad)
			deltahPrime += deg360InRad;
		else if (deltahPrime > deg180InRad)
			deltahPrime -= deg360InRad;
	}
	/* Equation 11 */
	double deltaHPrime = 2.0 * sqrt(CPrimeProduct) *
	    sin(deltahPrime / 2.0);
	
	/*
	 * Step 3
	 */
	/* Equation 12 */
	double barLPrime = (lab1[0] + lab2[0]) / 2.0;
	/* Equation 13 */
	double barCPrime = (CPrime1 + CPrime2) / 2.0;
	/* Equation 14 */
	double barhPrime, hPrimeSum = hPrime1 + hPrime2;
	if (CPrime1 * CPrime2 == 0) {
		barhPrime = hPrimeSum;
	} else {
		if (fabs(hPrime1 - hPrime2) <= deg180InRad)
			barhPrime = hPrimeSum / 2.0;
		else {
			if (hPrimeSum < deg360InRad)
				barhPrime = (hPrimeSum + deg360InRad) / 2.0;
			else
				barhPrime = (hPrimeSum - deg360InRad) / 2.0;
		}
	}
	/* Equation 15 */
	double T = 1.0 - (0.17 * cos(barhPrime - MyColor::deg2Rad(30.0))) +
	    (0.24 * cos(2.0 * barhPrime)) +
	    (0.32 * cos((3.0 * barhPrime) + MyColor::deg2Rad(6.0))) - 
	    (0.20 * cos((4.0 * barhPrime) - MyColor::deg2Rad(63.0)));
	/* Equation 16 */
	double deltaTheta = MyColor::deg2Rad(30.0) *
	    exp(-pow((barhPrime - deg2Rad(275.0)) / deg2Rad(25.0), 2.0));
	/* Equation 17 */
	double R_C = 2.0 * sqrt(pow(barCPrime, 7.0) /
	    (pow(barCPrime, 7.0) + pow25To7));
	/* Equation 18 */
	double S_L = 1 + ((0.015 * pow(barLPrime - 50.0, 2.0)) /
	    sqrt(20 + pow(barLPrime - 50.0, 2.0)));
	/* Equation 19 */
	double S_C = 1 + (0.045 * barCPrime);
	/* Equation 20 */
	double S_H = 1 + (0.015 * barCPrime * T);
	/* Equation 21 */
	double R_T = (-sin(2.0 * deltaTheta)) * R_C;
	
	/* Equation 22 */
	double deltaE = sqrt(
	    pow(deltaLPrime / (k_L * S_L), 2.0) +
	    pow(deltaCPrime / (k_C * S_C), 2.0) +
	    pow(deltaHPrime / (k_H * S_H), 2.0) + 
	    (R_T * (deltaCPrime / (k_C * S_C)) * (deltaHPrime / (k_H * S_H))));
	
	return (deltaE);
}


float 
MyColor::CIE76(const LAB& labColor1, const LAB& labColor2) {
	return sqrt((labColor1[0] - labColor2[0])*(labColor1[0] - labColor2[0]) +
		(labColor1[1] - labColor2[1])*(labColor1[1] - labColor2[1]) +
		(labColor1[2] - labColor2[2])*(labColor1[2] - labColor2[2]));
}


float 
MyColor::LumnContrast(const cv::Vec3f& labColor1, const cv::Vec3f& labColor2) {
	return abs(labColor1[0] - labColor2[0]);
}
