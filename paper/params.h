#define NTHREADSCPU 16 //used for parallelizing GPU tasks and the number of threads used in the CPU implementations
#define DTYPE double  //float or double
#define SMALLBLOCKSIZE 32 //CUDA block size for kernels that should use small block sizes
#define LARGEBLOCKSIZE 1024 //CUDA block size for kernels that should use large block sizes
#define NUMGPU 1 //the number of GPUs
#define NSTREAMSPERGPU 1 //streams per GPU for batching the frequences (use 1)




//Original Supersmoother with multiple passes over the data
#define ORIGINALMODE -1 //-1 -- cascade: first try SM 1 thread per freq, then try SM 1 block per freq, then run global memory kernel
                       //0 -- global memory baseline
					   //1 -- shared memory for x, y, z arrays, with one small block per frequency
					   //2 -- shared memory for x, y, z arrays, one thread per frequency with small block


//SuperSmoother with Single Pass
#define SINGLEPASSMODE 0 //0 -- global memory baseline
					   //1 -- shared memory for x, y, z arrays, one small block per frequency
					   //2 -- shared memory for x, y, z arrays, one thread per frequency with small block

#define COALESCED 1 //for the single pass global memory kernel (no equivalent for original multi-pass)
					//when SINGLEPASSMODE==0
					//uses coalesced accesses to x, y, z


#define PRINTPERIODS 0 //0- do not print found periods
					   //1-print found periods to stdout							
					   //2- print found periods to file (bestperiods_SS.txt)

#define PRINTPGRAM 0 //0- do not print pgram
					 //1- print pgram to file (pgram_SS.txt)

#define OBSTHRESH 4 //When batching, ignore computing objects with < this number of data points
					//e.g., may want to ignore objects with <50 data points
					//Must be at least 4 because original SS requires this in part of the code

//This will throw a bunch of compiler warnings if OBSTHRESH<4
#if OBSTHRESH<4
#undef OBSTHRESH
#define OBSTHRESH 4
#endif
