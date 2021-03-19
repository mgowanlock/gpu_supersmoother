#include "params.h"

int supsmu (int n, DTYPE * x, DTYPE * y, DTYPE * w, int iper, DTYPE span, DTYPE alpha, DTYPE * smo, DTYPE * sc);
int smooth (int n, DTYPE * x, DTYPE * y, DTYPE * w, DTYPE span, int iper, DTYPE vsmlsq, DTYPE * smo, DTYPE * acvr);
DTYPE supsmu_chi2(int n, DTYPE * time, DTYPE * data, DTYPE * weights , DTYPE * smo, DTYPE * sc, DTYPE alpha);

//Original port from Nat with parallelized frequencies
void supsmu_periodogram(int n, const double minFreq, const double maxFreq, int numFreq, double * time, double * data, double * error, double alpha, double * pgram);



//CPU functions for batched and single object processing
void supsmu_periodogram_innerloopcpu(int iteration, int n, DTYPE freqToTest, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE * pgram,
  DTYPE * tt, DTYPE * weights, DTYPE * chi2, DTYPE * sc, DTYPE * smo, DTYPE * t1, int * argkeys, DTYPE * t1_sortby_argkeys,
  DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys);

void supersmootherCPUBatch(bool MODEFLAG, unsigned int * objectId, DTYPE * time,  DTYPE * data, DTYPE * error, unsigned int sizeData, const DTYPE minFreq, const DTYPE maxFreq, 
  const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE ** pgram, DTYPE * foundPeriod, DTYPE alpha, 
  DTYPE * chi2, DTYPE * sc, DTYPE * smo, DTYPE * t1, int * argkeys, 
  DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys, DTYPE * weights, DTYPE * tt);


void supersmoothercpusingleobject(bool MODEFLAG, DTYPE * time, DTYPE * data, DTYPE * error, const unsigned int sizeData, 
  const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE alpha, 
  DTYPE * pgram, DTYPE * foundPeriod, DTYPE * chi2, DTYPE * sc, DTYPE * smo, DTYPE * t1, int * argkeys, 
  DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys, DTYPE * weights, DTYPE * tt);

void computePeriodSuperSmoother(DTYPE * pgram, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, DTYPE * foundPeriod);

//utility
// void sortKeyValuePairsIntDouble(int * keys, double * values, int n);
// void sortKeyValuePairsIntFloat(int * keys, float * values, int n);
void backToBackSort(int * dev_argkeys, int * dev_freqarr, DTYPE * dev_t1, int sizeData, int numFreq, cudaStream_t stream);
void compute_chi0_tt_weights(unsigned int sizeData, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE * chi0, DTYPE * tt, DTYPE * weights);

//used to compute delta f for the batch
double computedeltaf(struct lookupObj * objectLookup,  DTYPE * time, unsigned int numUniqueObjects);

//overload this function
void sortKeyValuePairsIntFloatDouble(int * keys, float * values, int n);
void sortKeyValuePairsIntFloatDouble(int * keys, double * values, int n);

//overload this function
void mapArr(double * inArr, double * outArr, int * keys, int n);
void mapArr(float * inArr, float * outArr, int * keys, int n);

//supersmoother main batch function
//one object on multiple GPUs
void supsmu_original_single_object(unsigned int * objectId, unsigned int sizeData, const DTYPE minFreq, const DTYPE maxFreq, int numFreq, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE * pgram, DTYPE * foundPeriod, double underestGPUcapacityGiB);

//one object on one GPU
void supsmu_original_single_gpu(unsigned int * objectId, unsigned int sizeData, const DTYPE minFreq, const DTYPE maxFreq, int numFreq, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE * pgram, DTYPE * foundPeriod, double underestGPUcapacityGiB, int gpuid);


//supersmoother single pass
void supsmu_singlepass_single_object(unsigned int * objectId, unsigned int sizeData, const DTYPE minFreq, const DTYPE maxFreq, int numFreq, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE * pgram, DTYPE * foundPeriod, double underestGPUcapacityGiB);

//one object on one GPU
void supsmu_singlepass_single_gpu(unsigned int * objectId, unsigned int sizeData, const DTYPE minFreq, const DTYPE maxFreq, int numFreq, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE * pgram, DTYPE * foundPeriod, double underestGPUcapacityGiB, int gpuid);

//CPU- supersmoother single pass inner loop
void supsmu_singlepass_periodogram_innerloopcpu(int iteration, int n, DTYPE freqToTest, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE * pgram,
  DTYPE * tt, DTYPE * weights, DTYPE * chi2, DTYPE * smo, DTYPE * t1, int * argkeys, DTYPE * t1_sortby_argkeys,
  DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys);

//CPU- single pass inner loop
DTYPE supsmu_singlepass_chi2(int n, DTYPE * time, DTYPE * data, DTYPE * weights , DTYPE * smo, DTYPE alpha);

void smoothsinglepass(int n, int *ibw, DTYPE *x, DTYPE *y, DTYPE *w, DTYPE vsmlsq, int alpha, DTYPE *smo);
int supsmusinglepass(int n, DTYPE * x, DTYPE * y, DTYPE * w, int iper, DTYPE span, DTYPE alpha, DTYPE * smo);

unsigned int computeNumBatches(bool mode, unsigned int sizeData, unsigned int numFreq, double underestGPUcapacityGiB, bool singlegpuflag);

//function for processing batch of objects
void supsmu_gpu_batch(bool mode, unsigned int * objectId, unsigned int sizeData, const DTYPE minFreq, const DTYPE maxFreq, unsigned int numFreq, DTYPE * time, DTYPE * data, DTYPE * error, DTYPE alpha, DTYPE ** pgram, DTYPE * foundPeriod);


//streams for multi-GPU for a single object
void createStreams(cudaStream_t * streams, unsigned int num_gpus, unsigned int streams_per_gpu);

//streams for a single GPU for a single object
void createStreamsOneGPU(cudaStream_t * streams, unsigned int streams_per_gpu, int gpuid);
void destroyStreamsOneGPU(cudaStream_t * streams, unsigned int streams_per_gpu, int gpuid);

//output to files and stdout:
void outputPeriodsToFile(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod);
void outputPgramToFile(struct lookupObj * objectLookup, unsigned int numUniqueObjects, unsigned int numFreqs, DTYPE ** pgram);
void outputPeriodsToStdout(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod);