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

#define NTHREADSCPU 16 //used for parallelizing GPU tasks and the number of threads used in the CPU implementations
#define DTYPE double  //float or double
#define NUMGPU 1 //the number of GPUs

#define PRINTPGRAM 0 //0- do not print pgram
					 //1- print pgram to file (pgram_SS.txt)

#define OBSTHRESH 4 //When computing on a catalog of objects, ignore computing objects with < this number of data points
					//e.g., may want to ignore objects with <50 data points
					//Must be at least 4 because original SS requires this in part of the code



//do not change
#define SMALLBLOCKSIZE 32 //CUDA block size for kernels that should use small block sizes
#define LARGEBLOCKSIZE 1024 //CUDA block size for kernels that should use large block sizes

//This will throw a bunch of compiler warnings if OBSTHRESH<4
#if OBSTHRESH<4
#undef OBSTHRESH
#define OBSTHRESH 4
#endif
//end do not change
