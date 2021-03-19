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

#include "structs.h"
#include "params.h"
#include <stdint.h>
#include <math.h>


////////////////////////////////////////
//original supersmoother main functions 
__device__ int smoothkernel(int n, DTYPE * x, DTYPE * y, DTYPE * w, DTYPE span, int iper, DTYPE vsmlsq, DTYPE * smo, DTYPE * acvr);

//original supsmu- global memory baseline
__global__ void supsmukernel(const int numThreads, const int n, const int iper, const DTYPE span, const DTYPE alpha, DTYPE * smo, 
  DTYPE * sc, DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys);

//original supsmu- One thread per frequency with SM for each thread storing x, y, w
__global__ void supsmukernelSMOneThreadPerFreq(const int numFreq, const int n, const int iper, const DTYPE span, const DTYPE alpha, 
  DTYPE * smo, DTYPE * sc, DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys);

//end original supersmoother
////////////////////////////////////////




////////////////////////////////////////
//single pass supersmoother main functions and utility that only apply to single pass
__device__ void smoothSinglePass(const int n, int * ibw, DTYPE * x, DTYPE * y, DTYPE * w, const DTYPE vsmlsq, const int alpha, DTYPE * smo);


__global__ void supsmukernelSinglePassGlobalMemoryCoalesced(const int numFreq, const int n, 
   const DTYPE inalpha, DTYPE * smo, DTYPE * tt, DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys);

//combine coalesced memory accesses remap with the mapping procedure
__global__ void mapUsingArgKeysOneThreadPerUpdateAndReorderCoalesced(const int n, const int numFreq, int * argkeys, DTYPE * data, DTYPE * weights, DTYPE * t1, DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys);

//coalesced acccesses to data (y) and weights (w)
// __global__ void computePgramReductionCoalesced(const int batchwriteoffset, const int numThreadsPerFreq, const DTYPE chi0, const int n, const int numFreq, DTYPE * chi2, DTYPE * smo, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys, DTYPE * pgram);
__global__ void computePgramReductionCoalesced(const int batchwriteoffset, const int numThreadsPerFreq, const DTYPE chi0, const int n, const int numFreq, DTYPE * smo, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys, DTYPE * pgram);

//coalesced accesses to time (x) data (y) and weights (w)
__device__ void smoothSinglePassCoalesced(const int n, const int freqNum, const int numFreq, int * ibw, DTYPE * x, DTYPE * y, DTYPE * w, const DTYPE vsmlsq, const int alpha, DTYPE * smo); 

//end single pass
////////////////////////////////////////

////////////////////////////////////////
//utility and pre/post main kernel functions 

//compute pgram after running supsmu
__global__ void computePgramReduction(const int batchwriteoffset, const int numThreadsPerFreq, const DTYPE chi0, const int n, const int numFreq, DTYPE * smo, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys, DTYPE * pgram);

//compute period mod f
__global__ void computePeriodModFOneThreadPerUpdate(const int n, const int numFreq, const DTYPE minFreq, const uint64_t freqOffset, const DTYPE deltaf, DTYPE * t1, DTYPE * tt);

//initialize key arrays
__global__ void initializeKeyArraysOneThreadPerUpdate(const int n, const int numFreq, int * argkeys, int * freqId);

//map function
__global__ void mapUsingArgKeysOneThreadPerUpdate(const int n, const int numFreq, int * argkeys, DTYPE * data, DTYPE * weights, DTYPE * t1, DTYPE * t1_sortby_argkeys, DTYPE * data_sortby_argkeys, DTYPE * weights_sortby_argkeys);
///////////////////////////////////////

