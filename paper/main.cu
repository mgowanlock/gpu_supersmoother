// MIT License

// Copyright (c) 2021 Mike Gowanlock and Nat Butler

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <fstream>
#include <istream>
#include <iostream>
#include <string>
#include <string.h>
#include <sstream>
#include <cstdlib>
#include <stdio.h>
#include <random>
#include "omp.h"
#include <algorithm> 
#include <queue>
#include <iomanip>
#include <set>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <cuda_profiler_api.h>
#include <thrust/extrema.h>
#include "structs.h"
#include "params.h"
#include "supsmu.h"
#include "kernel.h"
#include "main.h"



//for printing defines as strings

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)





using namespace std;

int main(int argc, char *argv[])
{

	warmUpGPU();
	cudaProfilerStart();
	omp_set_nested(1);

	//validation and output to file
	char fname[]="gpu_stats.txt";
	ofstream gpu_stats;
	gpu_stats.open(fname,ios::app);	

	/////////////////////////
	// Get information from command line
	/////////////////////////

	
	//Input data filename (objId, time, amplitude), minimum frequency, maximum frequency, numer of frequencies, mode
	if (argc!=7)
	{
	cout <<"\n\nIncorrect number of input parameters.\nExpected values: Data filename (objId, time, amplitude, error), minimum frequency, maximum frequency, number of frequencies, mode, alpha (default 9.0)\n";
	cout <<"\nModes: 1- GPU Original SuperSmoother";
	cout <<"\nModes: 2- GPU Single Pass SuperSmoother";
	cout <<"\nModes: 3- CPU Original SuperSmoother";
	cout <<"\nModes: 4- CPU Single Pass SuperSmoother";
	cout<<"\n\n";
	return 0;
	}
	
	
	char inputFname[500];
	strcpy(inputFname,argv[1]);
	double minFreq=atof(argv[2]); //inclusive
	double maxFreq=atof(argv[3]); //exclusive
	const unsigned int freqToTest=atoi(argv[4]);
    int MODE = atoi(argv[5]);
    double alpha=atof(argv[6]); //exclusive

	printf("\nDataset file: %s",inputFname);
	printf("\nMinimum Frequency: %f",minFreq);
	printf("\nMaximum Frequency: %f",maxFreq);
	printf("\nNumber of frequencies to test: %u", freqToTest);
	printf("\nMode: %d", MODE);

	// #if ERROR==1
	// printf("\nExecuting L-S variant from AstroPy that propogates error and floats the mean");
	// #endif
	
	
	/////////////
	//Import Data
	/////////////
	unsigned int * objectId=NULL; 
	DTYPE * timeX=NULL; 
	DTYPE * magY=NULL;
	DTYPE * magDY=NULL;
	unsigned int sizeData;
	importObjXYData(inputFname, &sizeData, &objectId, &timeX, &magY, &magDY);	

	
	
	
	
	//pgram allocated in the functions below
	//Stores the LS power for each frequency
	DTYPE * pgram=NULL;


	//GPU Original SuperSmoother. Process a batch of objects or a single object.
	if (MODE==1)
	{
		
		DTYPE sumPeriods=0;
		
		double tstart=omp_get_wtime();
		
		//Original supersmoother algorithm
		//0- refers to original algorithm
		supsmu_gpu_batch(0, objectId, sizeData, minFreq, maxFreq, freqToTest, timeX, magY, magDY, alpha, &pgram, &sumPeriods);

		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);

		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<", NTHREADSCPU/SMALLBLOCKSIZE/LARGEBLOCKSIZE/NUMGPU/NSTREAMSPERGPU/ORIGINALMODE/SINGLEPASSMODE/COALESCED/DTYPE: "<<NTHREADSCPU<<", "<<SMALLBLOCKSIZE<<", "<<LARGEBLOCKSIZE<<", "<<NUMGPU<<", "<<NSTREAMSPERGPU<<", "<<ORIGINALMODE<<", "<<SINGLEPASSMODE<<", "<<COALESCED<<", "<<STR(DTYPE)<<endl;
	}

	//GPU Single Pass SuperSmoother. Process a batch of objects or a single object.
	else if (MODE==2)
	{
		
		DTYPE sumPeriods=0;
		
		double tstart=omp_get_wtime();
		
		//Single pass supersmoother -- process a single object or batch
		//1-refers to the single pass mode
		supsmu_gpu_batch(1, objectId, sizeData, minFreq, maxFreq, freqToTest, timeX, magY, magDY, alpha, &pgram, &sumPeriods);

		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);

		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<", NTHREADSCPU/SMALLBLOCKSIZE/LARGEBLOCKSIZE/NUMGPU/NSTREAMSPERGPU/ORIGINALMODE/SINGLEPASSMODE/COALESCED/DTYPE: "<<NTHREADSCPU<<", "<<SMALLBLOCKSIZE<<", "<<LARGEBLOCKSIZE<<", "<<NUMGPU<<", "<<NSTREAMSPERGPU<<", "<<ORIGINALMODE<<", "<<SINGLEPASSMODE<<", "<<COALESCED<<", "<<STR(DTYPE)<<endl;
	}
	//CPU- Original SuperSmoother. Process a batch of objects or a single object.
	else if (MODE==3)
	{
		
	  printf("\nSize data: %d", sizeData);	
	  //Allocate once 
	  DTYPE * chi2=(DTYPE *)malloc(sizeof(DTYPE)*freqToTest);
  	  // DTYPE * y = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);
  	  DTYPE * weights = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);
      DTYPE * tt = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);
      
      //Arrays that need to be allocated for each thread
      DTYPE * sc=(DTYPE *)malloc(sizeof(DTYPE)*sizeData*8*NTHREADSCPU);
      DTYPE * smo=(DTYPE *)malloc(sizeof(DTYPE)*sizeData*NTHREADSCPU); 
      DTYPE * t1=(DTYPE *)malloc(sizeof(DTYPE)*sizeData*NTHREADSCPU); 
      int * argkeys=(int *)malloc(sizeof(int)*sizeData*NTHREADSCPU); 
      DTYPE * t1_sortby_argkeys=(DTYPE *)malloc(sizeof(DTYPE)*sizeData*NTHREADSCPU); 
      DTYPE * data_sortby_argkeys=(DTYPE *)malloc(sizeof(DTYPE)*sizeData*NTHREADSCPU); 
      DTYPE * weights_sortby_argkeys=(DTYPE *)malloc(sizeof(DTYPE)*sizeData*NTHREADSCPU); 
      

		DTYPE sumPeriods=0;
		double tstart=omp_get_wtime();
		DTYPE * pgram=NULL;
		DTYPE foundPeriod=0;

	  //0- refers to default supersmoother (multipass)
	  supersmootherCPUBatch(0, objectId, timeX,  magY, magDY, sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, &foundPeriod, alpha,
		chi2, sc, smo, t1, argkeys, t1_sortby_argkeys, data_sortby_argkeys, weights_sortby_argkeys, weights, tt);
	  double tend=omp_get_wtime();
	  double totalTime=tend-tstart;
	  printf("\nTotal time to compute batch: %f", totalTime);
	  printf("\n[Validation] Sum of all periods: %f", sumPeriods);
	  gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<", NTHREADSCPU: "<<NTHREADSCPU<<", DTYPE: "<<STR(DTYPE)<<endl;

	  
	  free(sc);
	  free(chi2);	
      free(smo);
      free(t1);
      free(argkeys);
      free(t1_sortby_argkeys);
      free(data_sortby_argkeys);
      free(weights_sortby_argkeys);
      free(weights); 
      free(tt); 
      
      
	}
	//CPU- Single pass SuperSmoother. Process a batch of objects or a single object.
	else if (MODE==4)
	{
		
		    printf("\nSize data: %d", sizeData);	
			//Allocate once 
			DTYPE * chi2=(DTYPE *)malloc(sizeof(DTYPE)*freqToTest);
		  	// DTYPE * y = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);
		  	DTYPE * weights = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);
		    DTYPE * tt = (DTYPE *)malloc(sizeof(DTYPE)*sizeData);
		    
		    //Arrays that need to be allocated for each thread
		    DTYPE * sc=NULL; //sc not needed for single pass implementation
		    DTYPE * smo=(DTYPE *)malloc(sizeof(DTYPE)*sizeData*NTHREADSCPU); 
		    DTYPE * t1=(DTYPE *)malloc(sizeof(DTYPE)*sizeData*NTHREADSCPU); 
		    int * argkeys=(int *)malloc(sizeof(int)*sizeData*NTHREADSCPU); 
		    DTYPE * t1_sortby_argkeys=(DTYPE *)malloc(sizeof(DTYPE)*sizeData*NTHREADSCPU); 
		    DTYPE * data_sortby_argkeys=(DTYPE *)malloc(sizeof(DTYPE)*sizeData*NTHREADSCPU); 
		    DTYPE * weights_sortby_argkeys=(DTYPE *)malloc(sizeof(DTYPE)*sizeData*NTHREADSCPU); 
      

			DTYPE sumPeriods=0;
			double tstart=omp_get_wtime();
			

			//allocate pgram once we know the number of unique objects
			DTYPE * pgram=NULL;
			DTYPE foundPeriod=0;

			//1- refers to single pass 
			supersmootherCPUBatch(1, objectId, timeX,  magY, magDY, sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, &foundPeriod, alpha,
				chi2, sc, smo, t1, argkeys, t1_sortby_argkeys, data_sortby_argkeys, weights_sortby_argkeys, weights, tt);
			double tend=omp_get_wtime();
			double totalTime=tend-tstart;
			printf("\nTotal time to compute batch: %f", totalTime);
			printf("\n[Validation] Sum of all periods: %f", sumPeriods);
			gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<", NTHREADSCPU: "<<NTHREADSCPU<<", DTYPE: "<<STR(DTYPE)<<endl;

		  
		  
			free(chi2);	
		    free(smo);
		    free(t1);
		    free(argkeys);
		    free(t1_sortby_argkeys);
		    free(data_sortby_argkeys);
		    free(weights_sortby_argkeys);
		    free(weights); 
		    free(tt); 
      
      
	}


	//free memory
	free(objectId);
	free(timeX);
	free(magY);
	free(magDY);
	free(pgram);


	cudaProfilerStop();

	gpu_stats.close();
	printf("\n");
	return 0;
}


void computeObjectRanges(unsigned int * objectId, unsigned int * sizeData, struct lookupObj ** objectLookup, unsigned int * numUniqueObjects)
{
	//Scan to find unique object ids;
	unsigned int lastId=objectId[0];
	unsigned int cntUnique=1;
	for (unsigned int i=1; i<*sizeData; i++)
	{
		if (lastId!=objectId[i])
		{
			cntUnique++;
			lastId=objectId[i];
		}
	}

	//allocate memory for the struct
	*objectLookup=(lookupObj*)malloc(sizeof(lookupObj)*cntUnique);

	*numUniqueObjects=cntUnique;
	printf("\nUnique objects in file: %u",*numUniqueObjects);



	lastId=objectId[0];
	unsigned int cnt=0;
	for (unsigned int i=1; i<*sizeData; i++)
	{
		if (lastId!=objectId[i])
		{
			(*objectLookup)[cnt].objId=lastId;
			(*objectLookup)[cnt+1].idxMin=i;
			(*objectLookup)[cnt].idxMax=i-1;
			cnt++;
			lastId=objectId[i];
		}
	}

	//first and last ones
	(*objectLookup)[0].idxMin=0;
	(*objectLookup)[cnt].objId=objectId[(*sizeData)-1];
	(*objectLookup)[cnt].idxMax=(*sizeData)-1;

}



void importObjXYData(char * fnamedata, unsigned int * sizeData, unsigned int ** objectId, DTYPE ** timeX, DTYPE ** magY, DTYPE ** magDY)
{

	//import objectId, time, mag, error
	std::vector<DTYPE>tmpAllData;
	std::ifstream in(fnamedata);
	unsigned int cnt=0;
	for (std::string f; getline(in, f, ',');){

	DTYPE i;
		 std::stringstream ss(f);
	    while (ss >> i)
	    {
	        tmpAllData.push_back(i);
	        // array[cnt]=i;
	        cnt++;
	        if (ss.peek() == ',')
	            ss.ignore();
	    }

  	}

  	*sizeData=(unsigned int)tmpAllData.size()/4;
  	printf("\nData import: Total rows: %u",*sizeData);
  	
  	*objectId=(unsigned int *)malloc(sizeof(DTYPE)*(*sizeData));
  	*timeX=   (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));
  	*magY=    (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));
  	*magDY=    (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));
  	
  	for (unsigned int i=0; i<*sizeData; i++){
  		(*objectId)[i]=tmpAllData[(i*4)+0];
  		(*timeX)[i]   =tmpAllData[(i*4)+1];
  		(*magY)[i]    =tmpAllData[(i*4)+2];
  		(*magDY)[i]    =tmpAllData[(i*4)+3];
  	}
  	
}




void warmUpGPU(){
printf("\nLoad CUDA runtime (initialization overhead)\n");

for (int i=0; i<NUMGPU; i++)
{
cudaSetDevice(i); 	
cudaDeviceSynchronize();
}

}