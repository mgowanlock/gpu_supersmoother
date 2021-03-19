//do not change
#define SMALLBLOCKSIZE 32 //CUDA block size for kernels that should use small block sizes
#define LARGEBLOCKSIZE 1024 //CUDA block size for kernels that should use large block sizes

//This will throw a bunch of compiler warnings if OBSTHRESH<4
#if OBSTHRESH<4
#undef OBSTHRESH
#define OBSTHRESH 4
#endif
//end do not change


#define NTHREADSCPU 16 //used for parallelizing GPU tasks and the number of threads used in the CPU implementations
#define DTYPE double  //float or double
#define NUMGPU 1 //the number of GPUs

#define PRINTPGRAM 0 //0- do not print pgram
					 //1- print pgram to file (pgram_SS.txt)

#define OBSTHRESH 4 //When computing on a catalog of objects, ignore computing objects with < this number of data points
					//e.g., may want to ignore objects with <50 data points
					//Must be at least 4 because original SS requires this in part of the code


