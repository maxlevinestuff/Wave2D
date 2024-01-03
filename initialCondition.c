#include <math.h>

double initialCondition(double x, double y) {
	double sigma=0.01;//tight point
	//double sigma=0.1;//wider point
	double mu=0.5;//center
	double max = (1.0/(2.0*M_PI*sigma*sigma))*exp(-0.5*( ((0.5-mu)/sigma)*((0.5-mu)/sigma) +  ((0.5-mu)/sigma)*((0.5-mu)/sigma)   ));
	double result = (1.0/(2.0*M_PI*sigma*sigma))*exp(-0.5*( ((x-mu)/sigma)*((x-mu)/sigma) +  ((y-mu)/sigma)*((y-mu)/sigma)   ))/max;
	return result;
}
