#include "initialCondition.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

double f(int x, int y, int s, double*** array, int xdim, int ydim, int zdim, double h, double k) {
	if (x == 0 || x == 1 || y == 0 || y == 1)
		return 0;
	if (s == 0 || s == 1) {
		double xDouble = 1.0/xdim*x;
        double yDouble = 1.0/ydim*y;
		return initialCondition(xDouble,yDouble);
	}

	double upNeighbor, downNeighbor, rightNeighbor, leftNeighbor;
	if (y == ydim-1)
		upNeighbor = 0;
	else
		upNeighbor = array[x][y+1][s-1];

	if (y==0)
		downNeighbor = 0;
	else
		downNeighbor = array[x][y-1][s-1];

	if (x == xdim-1)
		rightNeighbor = 0;
	else
		rightNeighbor = array[x+1][y][s-1];

	if (x == 0)
		leftNeighbor = 0;
	else
		leftNeighbor = array[x-1][y][s-1];

	return ((k*k)/(h*h))*(leftNeighbor+rightNeighbor+downNeighbor+upNeighbor - 4*array[x][y][s-1]) + 2*array[x][y][s-1]-array[x][y][s-2];
}

int mapValues(double input, double min, double max) {
	double slope = (255 - 0) / (max - min);
	return 0 + round(slope * (input - min));
}

int main(int argc, char *argv[]) {

	double spacestep = atof(argv[1]);
	double timestep = atof(argv[2]);
	double endtime = atof(argv[3]);
	int timeskip = atoi(argv[4]);
	int gridskip = atoi(argv[5]);

	int N = 1 / spacestep + 1;
	int totalSteps = ceil ( endtime / timestep ) ;

	//allocate matrix
	int xdim = N; int ydim = N; int zdim = totalSteps;
	double ***output = malloc(xdim * sizeof(double **));
	for(int i = 0; i < xdim; i++) {
		output[i] = malloc(ydim * sizeof(double *));
		for(int j = 0; j < ydim; j++)
			output[i][j] = malloc(zdim * sizeof(double));
	}

	omp_set_num_threads(2);

	//calculate/fill matrix
	for (int ziter = 0; ziter <= totalSteps; ziter++) {

		#pragma omp parallel
		{

			//find the y bounds for each of the 2 threads
			int startY; int endY;
			switch(omp_get_thread_num()) {
				case 0:
					startY = 0;
					endY = ydim / 2;
					break;
				case 1:
					startY = ydim / 2;
					endY = ydim;
					break;
			}

			for (int yiter = startY; yiter < endY; yiter++) {
				for (int xiter = 0; xiter < xdim; xiter++) {
					output[xiter][yiter][ziter] = f(xiter,yiter,ziter,output,xdim,ydim,zdim, spacestep, timestep);
				}
			}

		}

	}

	//write output
	for (int ziter = 0; ziter <= totalSteps; ziter += timeskip) {
		FILE * fout;
		char filename[13];
		sprintf(filename, "./%06d.pgm", ziter/timeskip);
		fout = fopen(filename,"w");
		fprintf(fout,"P2\n%d %d\n255\n",N/gridskip,N/gridskip);

		//find max and min and bounds
		double max = output[0][0][ziter];
		double min = output[0][0][ziter];
		for (int yiter = 1; yiter < ydim; yiter++) {
			for (int xiter = 1; xiter < xdim; xiter++) {
				if (output[xiter][yiter][ziter] > max)
					max = output[xiter][yiter][ziter];
				if (output[xiter][yiter][ziter] < min)
					min = output[xiter][yiter][ziter];
			}
		}
		double absMax = fabs(max);
		double absMin = fabs(min);
		double bound;
		if (absMax > absMin)
			bound = absMax;
		else
			bound = absMin;

		for (int yiter = 0; yiter < ydim; yiter += gridskip) {
			int counter = 0; //insures no more than N/gridskip is outputted
			for (int xiter = 0; xiter < xdim; xiter += gridskip) {
				if (xiter >= xdim-gridskip*2 || yiter >= ydim-gridskip*2)
					fprintf(fout,"%d ",mapValues(0,-bound,bound)); //makes that gray border
				else {
					fprintf(fout,"%d ",mapValues(output[xiter][yiter][ziter],-bound,bound));
					printf("%s, t=%g\n",filename+2, ziter*timestep);
				}
				counter++;
				if (counter == N/gridskip)
					break;
			}
			fprintf(fout,"\n");
		}

		fclose(fout);
	}

	//free
	for(int i = 0; i < xdim; i++) {
    	for(int j = 0; j < ydim; j++)
        	free(output[i][j]);
    	free(output[i]);
	}
	free(output);
}
