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

//Only include parameters file if we're not creating the shared library
#ifndef PYTHON
#include "params.h"
#endif


#include "supsmu.h"
#include "kernel.h"
#include "main.h"



//for printing defines as strings

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)





using namespace std;

#ifndef PYTHON
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
	cout <<"\n\nIncorrect number of input parameters.\nExpected values: Data filename (objId, time, amplitude), minimum frequency, maximum frequency, number of frequencies, alpha (0.0-10.0, default=9.0), mode\n";
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
	double alphain=atof(argv[5]);
    int MODE = atoi(argv[6]);

	printf("\nDataset file: %s",inputFname);
	printf("\nMinimum Frequency: %f",minFreq);
	printf("\nMaximum Frequency: %f",maxFreq);
	printf("\nNumber of frequencies to test: %u", freqToTest);
	printf("\nAlpha (outlier penalization parameter): %f", alphain);
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

	
	

	// // int keys[5]={0,1,2,3,4};
	// // double values[5]={0, 1, 2, 3, 0};
	

	// // sortKeyValuePairsIntDouble(keys, values, 5);

	// // for (int i=0; i<5; i++)
	// // {
	// // printf("\nKeys: %d", keys[i]);
	// // }	

	// // for (int i=0; i<5; i++)
	// // {
	// // printf("\nValues: %f", values[i]);
	// // }

	



	// // return 0;


	//Nat's number of frequency picker
	/*
	double tmin=timeX[0];
	double tmax=timeX[0];
	for (int i=0; i<sizeData; i++)
	{
		if (timeX[i]<tmin)
		{
			tmin=timeX[i];
		}

		if (timeX[i]>tmax)
		{
			tmax=timeX[i];
		}
	}

	double Xmax=tmax-tmin;

	double f0=1.0/Xmax;
	double df=0.1/Xmax;
	double fe=10;
	int numFreqs=int((fe-f0)/df);
	printf("\nnum freqs nat: %d",numFreqs);

	

	printf("\nXmax: %f", Xmax);
	printf("\nMin freq: %f (Period: %f)", f0, 1.0/f0);
	printf("\nMax freq: %f (Period: %f)", f0+(df*numFreqs), 1.0/(f0+(df*numFreqs)));
	*/

	// printf("\n*********\nAssuming zero error\n************\n");
	// for (int i=0; i<sizeData; i++)
	// {
	// 	magDY[i]=0.0001;
	// }


	
	
	double tstart=omp_get_wtime();
	// double * psd=(double *)malloc(sizeof(double)*numFreqs);
	//Port directly from Nat's code
	// supsmu_periodogram(sizeData, f0, f0+(df*numFreqs), numFreqs, timeX, magY, magDY, alpha, psd);
	double tend=omp_get_wtime();
	printf("\nTotal time: %f", tend - tstart);

	// printf("\nNumber of frequencies: %d", numFreqs);
	printf("\n************\n");
	
	//pgram allocated in the functions below
	//Stores the LS power for each frequency
	DTYPE * pgram=NULL;


	//GPU Original SuperSmoother. Process a batch of objects or a single object.
	if (MODE==1)
	{
		// DTYPE foundPeriod=0;
		DTYPE sumPeriods=0;
		
		double tstart=omp_get_wtime();
		
		//Original supersmoother algorithm -- process a single object
		// supsmu_original_single_object(objectId, sizeData, minFreq, maxFreq, freqToTest, timeX, magY, magDY, alpha, &pgram, &foundPeriod);
		//0-refers to the original
		supsmu_gpu_batch(0, objectId, sizeData, minFreq, maxFreq, freqToTest, timeX, magY, magDY, alphain, &pgram, &sumPeriods);

		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);

		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<\
		minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<\
		", NTHREADSCPU/SMALLBLOCKSIZE/LARGEBLOCKSIZE/NUMGPU/NSTREAMSPERGPU/ORIGINALMODE/SINGLEPASSMODE/COALESCED/ALPHA/BETA/DTYPE: "<<\
		NTHREADSCPU<<", "<<SMALLBLOCKSIZE<<", "<<LARGEBLOCKSIZE<<", "<<NUMGPU<<", "<<NSTREAMSPERGPU<<", "<<ORIGINALMODE<<", "\
		<<SINGLEPASSMODE<<", "<<COALESCED<<", "<<alphain<<", "<<BETA<<", "<<STR(DTYPE)<<endl;
	}

	//GPU Single Pass SuperSmoother. Process a batch of objects or a single object.
	else if (MODE==2)
	{
		
		DTYPE sumPeriods=0;
		
		double tstart=omp_get_wtime();
		
		//Single pass supersmoother -- process a single object
		//1-refers to the single pass mode
		supsmu_gpu_batch(1, objectId, sizeData, minFreq, maxFreq, freqToTest, timeX, magY, magDY, alphain, &pgram, &sumPeriods);

		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);

		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<\
		minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<\
		", NTHREADSCPU/SMALLBLOCKSIZE/LARGEBLOCKSIZE/NUMGPU/NSTREAMSPERGPU/ORIGINALMODE/SINGLEPASSMODE/COALESCED/ALPHA/BETA/DTYPE: "<<\
		NTHREADSCPU<<", "<<SMALLBLOCKSIZE<<", "<<LARGEBLOCKSIZE<<", "<<NUMGPU<<", "<<NSTREAMSPERGPU<<", "<<ORIGINALMODE<<", "\
		<<SINGLEPASSMODE<<", "<<COALESCED<<", "<<alphain<<", "<<BETA<<", "<<STR(DTYPE)<<endl;
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
		supersmootherCPUBatch(0, objectId, timeX,  magY, magDY, sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, &foundPeriod, alphain,
			chi2, sc, smo, t1, argkeys, t1_sortby_argkeys, data_sortby_argkeys, weights_sortby_argkeys, weights, tt);
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);
		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<", NTHREADSCPU: "<<NTHREADSCPU<<", ALPHA: "<<alphain<<", DTYPE: "<<STR(DTYPE)<<endl;

	  
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
		// #if ERROR==0
		// lombscargleCPUBatch(objectId, timeX, magY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, pgram, foundPeriod);
		// #endif
		// #if ERROR==1
		// lombscargleCPUBatchError(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, pgram, foundPeriod);
		// #endif

		// double omegamin=1.00531;
		// double omegamax=150.796;
		// double numFreqs=100000;
		// double f0=omegamin/(2.0*M_PI);
		// double fmax=omegamax/(2.0*M_PI);
		// double df=(fmax-f0)/numFreqs;

		// printf("\nfmin/fmax: %f, %f", f0, fmax);

		// return;

		//allocate pgram once we know the number of unique objects
		DTYPE * pgram=NULL;
		DTYPE foundPeriod=0;

		//1- refers to single pass 
		supersmootherCPUBatch(1, objectId, timeX,  magY, magDY, sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, &foundPeriod, alphain,
			chi2, sc, smo, t1, argkeys, t1_sortby_argkeys, data_sortby_argkeys, weights_sortby_argkeys, weights, tt);
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);
		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<", NTHREADSCPU: "<<NTHREADSCPU<<", ALPHA: "<<alphain<<", DTYPE: "<<STR(DTYPE)<<endl;

	  
	  
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
#endif


#ifdef PYTHON
// extern "C" void SuperSmootherPy(char * inputFname, double minFreq, double maxFreq, unsigned int freqToTest, double alphain, int MODE)
extern "C" void SuperSmootherPy(unsigned int * objectId, DTYPE * timeX, DTYPE * magY, DTYPE * magDY, unsigned int sizeData, double minFreq, double maxFreq, unsigned int freqToTest, double alphain, int MODE, DTYPE * pgram)
{


	omp_set_nested(1);

	//validation and output to file
	char fname[]="gpu_stats.txt";
	ofstream gpu_stats;
	gpu_stats.open(fname,ios::app);	

	/////////////////////////
	// Get information from command line
	/////////////////////////

	
	//Input data filename (objId, time, amplitude), minimum frequency, maximum frequency, numer of frequencies, mode
	// if (argc!=7)
	// {
	// cout <<"\n\nIncorrect number of input parameters.\nExpected values: Data filename (objId, time, amplitude), minimum frequency, maximum frequency, number of frequencies, alpha (0.0-10.0, default=9.0), mode\n";
	// cout <<"\nModes: 1- GPU Original SuperSmoother";
	// cout <<"\nModes: 2- GPU Single Pass SuperSmoother";
	// cout <<"\nModes: 3- CPU Original SuperSmoother";
	// cout <<"\nModes: 4- CPU Single Pass SuperSmoother";
	// cout<<"\n\n";
	// }
	
	
	// char inputFname[500];
	// strcpy(inputFname,argv[1]);
	// double minFreq=atof(argv[2]); //inclusive
	// double maxFreq=atof(argv[3]); //exclusive
	// const unsigned int freqToTest=atoi(argv[4]);
	// double alphain=atof(argv[5]);
 //    int MODE = atoi(argv[6]);

	// printf("\nDataset file: %s",inputFname);
	printf("\nMinimum Frequency: %f",minFreq);
	printf("\nMaximum Frequency: %f",maxFreq);
	printf("\nNumber of frequencies to test: %u", freqToTest);
	printf("\nAlpha (outlier penalization parameter): %f", alphain);
	printf("\nMode: %d", MODE);

	// #if ERROR==1
	// printf("\nExecuting L-S variant from AstroPy that propogates error and floats the mean");
	// #endif
	
	
	/////////////
	//Import Data
	/////////////
	// unsigned int * objectId=NULL; 
	// DTYPE * timeX=NULL; 
	// DTYPE * magY=NULL;
	// DTYPE * magDY=NULL;
	// unsigned int sizeData;
	// importObjXYData(inputFname, &sizeData, &objectId, &timeX, &magY, &magDY);	

	//pgram allocated in the functions below
	//Stores the LS power for each frequency
	// DTYPE * pgram=NULL;


	//GPU Original SuperSmoother. Process a batch of objects or a single object.
	if (MODE==1)
	{
		// DTYPE foundPeriod=0;
		DTYPE sumPeriods=0;
		
		double tstart=omp_get_wtime();
		
		//Original supersmoother algorithm -- process a single object
		// supsmu_original_single_object(objectId, sizeData, minFreq, maxFreq, freqToTest, timeX, magY, magDY, alpha, &pgram, &foundPeriod);
		//0-refers to the original
		supsmu_gpu_batch(0, objectId, sizeData, minFreq, maxFreq, freqToTest, timeX, magY, magDY, alphain, &pgram, &sumPeriods);

		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);

		gpu_stats<<totalTime<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<\
		minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<\
		", NTHREADSCPU/SMALLBLOCKSIZE/LARGEBLOCKSIZE/NUMGPU/NSTREAMSPERGPU/ORIGINALMODE/SINGLEPASSMODE/COALESCED/ALPHA/BETA/DTYPE: "<<\
		NTHREADSCPU<<", "<<SMALLBLOCKSIZE<<", "<<LARGEBLOCKSIZE<<", "<<NUMGPU<<", "<<NSTREAMSPERGPU<<", "<<ORIGINALMODE<<", "\
		<<SINGLEPASSMODE<<", "<<COALESCED<<", "<<alphain<<", "<<BETA<<", "<<STR(DTYPE)<<endl;
	}

	//GPU Single Pass SuperSmoother. Process a batch of objects or a single object.
	else if (MODE==2)
	{
		
		DTYPE sumPeriods=0;
		
		double tstart=omp_get_wtime();
		
		//Single pass supersmoother -- process a single object
		//1-refers to the single pass mode
		supsmu_gpu_batch(1, objectId, sizeData, minFreq, maxFreq, freqToTest, timeX, magY, magDY, alphain, &pgram, &sumPeriods);

		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);

		gpu_stats<<totalTime<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<\
		minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<\
		", NTHREADSCPU/SMALLBLOCKSIZE/LARGEBLOCKSIZE/NUMGPU/NSTREAMSPERGPU/ORIGINALMODE/SINGLEPASSMODE/COALESCED/ALPHA/BETA/DTYPE: "<<\
		NTHREADSCPU<<", "<<SMALLBLOCKSIZE<<", "<<LARGEBLOCKSIZE<<", "<<NUMGPU<<", "<<NSTREAMSPERGPU<<", "<<ORIGINALMODE<<", "\
		<<SINGLEPASSMODE<<", "<<COALESCED<<", "<<alphain<<", "<<BETA<<", "<<STR(DTYPE)<<endl;
	}
	//CPU- Original SuperSmoother. Process a batch of objects or a single object.
	else if (MODE==3)
	{
		
	  double tstart=omp_get_wtime();
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
		DTYPE foundPeriod=0;

		//0- refers to default supersmoother (multipass)
		supersmootherCPUBatch(0, objectId, timeX,  magY, magDY, sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, &foundPeriod, alphain,
			chi2, sc, smo, t1, argkeys, t1_sortby_argkeys, data_sortby_argkeys, weights_sortby_argkeys, weights, tt);
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);
		gpu_stats<<totalTime<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<", NTHREADSCPU: "<<NTHREADSCPU<<", ALPHA: "<<alphain<<", DTYPE: "<<STR(DTYPE)<<endl;

	  
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

	  double tstart=omp_get_wtime();	
		
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
	  DTYPE foundPeriod=0;

	  //1- refers to single pass 
	  supersmootherCPUBatch(1, objectId, timeX,  magY, magDY, sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, &foundPeriod, alphain,
		chi2, sc, smo, t1, argkeys, t1_sortby_argkeys, data_sortby_argkeys, weights_sortby_argkeys, weights, tt);
	  double tend=omp_get_wtime();
	  double totalTime=tend-tstart;
	  printf("\nTotal time to compute batch: %f", totalTime);
	  printf("\n[Validation] Sum of all periods: %f", sumPeriods);
	  gpu_stats<<totalTime<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<", NTHREADSCPU: "<<NTHREADSCPU<<", ALPHA: "<<alphain<<", DTYPE: "<<STR(DTYPE)<<endl;

	  
	  
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


	
	// free(pgram);

	gpu_stats.close();
}
#endif

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