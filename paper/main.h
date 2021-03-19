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

//function prototypes
void importObjXYData(char * fnamedata, unsigned int * sizeData, unsigned int ** objectId, DTYPE ** timeX, DTYPE ** magY, DTYPE ** magDY);

//CPU L-S Functions:
void lombscarglecpu(bool mode, DTYPE * x, DTYPE * y, const unsigned int sizeData, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE * pgram);
void lombscarglecpuinnerloop(int iteration, DTYPE * x, DTYPE * y, DTYPE * pgram, DTYPE * freqToTest, const unsigned int sizeData);
void lombscargleCPUOneObject(DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * pgram);
void lombscargleCPUBatch(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod);

//With error
void lombscargleCPUOneObjectError(DTYPE * time, DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * pgram);
void lombscarglecpuError(bool mode, DTYPE * x, DTYPE * y, DTYPE *dy, const unsigned int sizeData, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE * pgram);
void lombscarglecpuinnerloopAstroPy(int iteration, DTYPE * x, DTYPE * y, DTYPE * dy, DTYPE * pgram, DTYPE * freqToTest, const unsigned int sizeData);
void lombscargleCPUBatchError(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod);
void updateYerrorfactor(DTYPE * y, DTYPE *dy, const unsigned int sizeData);

//GPU functions
void batchGPULS(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE ** pgram, DTYPE * foundPeriod);
void GPULSOneObject(DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * periodFound, DTYPE ** pgram);
void computeObjectRanges(unsigned int * objectId, unsigned int * sizeData, struct lookupObj ** objectLookup, unsigned int * numUniqueObjects);
void pinnedMemoryCopyDtoH(DTYPE * pinned_buffer, unsigned int sizeBufferElems, DTYPE * dev_data, DTYPE * pageable, unsigned int sizeTotalData, cudaStream_t * streams);

//overloaded
void pinnedMemoryCopyHtoD(DTYPE * pinned_buffer, unsigned int sizeBufferElems, DTYPE * dev_data, DTYPE * pageable, unsigned int sizeTotalData, cudaStream_t * streams);
void pinnedMemoryCopyHtoD(int * pinned_buffer, unsigned int sizeBufferElems, int * dev_data, int * pageable, unsigned int sizeTotalData, cudaStream_t * streams);

void computePeriod(DTYPE * pgram, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE freqStep, DTYPE * foundPeriod);

void warmUpGPU();

//Error checking GPU calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}