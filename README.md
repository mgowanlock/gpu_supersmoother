# GPU-Accelerated Super Smoother Algorithm

Accompanying paper, "GPU-Enabled Searches for Periodic Signals of Unknown Shape" is under review.

Code authors: Mike Gowanlock and Nat Butler

## Questions/Comments
Feel free to e-mail Mike Gowanlock. 

## There are four directories:
* data
* example
* paper
* release

The data directory includes test data (136 RR-Lyrae from SDSS Stripe 82, which was used in the paper). The paper directory contains the source code used for the experimental evaluation in the paper. The difference between the paper and release code is that many of the GPU performance parameters have been selected for the user so that a reasonable default configuration can be used without extensive knowledge of the details in the paper. However, if the user is interested in all of the bells and whistles included in the paper, then they should use the paper implementation.

## Single Object and Batched Modes
As described in the paper, the GPU algorithm allows for both a single object to be processed (e.g., a user wants to process a large time series or a large number of frequencies need to be searched). And it also allows for a batch of objects to be processed (e.g., deriving periods for numerous objects in an astronomical catalog). The algorithm will automatically determine whether you have input a file with a single or multiple objects and execute the correct version of the code.

## Modes: The original Super Smoother algorithm, and a single-pass variant
We include two versions of the algorithm. The original Super Smoother algorithm, which uses cross-validation to locally fit line segments to the data. We also have a single-pass variant of the algorithm that performs generalized validation. This main difference in terms of computational complexity is that the original algorithm requires several scans over the sorted time series for each search frequency. In contrast, the generalized validation version of the algorithm only requires a single scan over the time series.

## Data Directory
The data directory contains the file "SDSS_stripe82_band_z.txt". The file has measurements of 136 RR-Lyrae from SDSS Stripe 82.

The dataset files should be in the format: object id, time, mag, dmag (error). See the file in the data directory as an example.

Note that if you do not have error on your magnitude measurements, you should add an error column to your file.  Use an error of 1.0 for all measurements, and not 0.0 (or a very small number), because it may cause issues related to numerical overflow.

## Makefile
A makefile has been included for each implementation. Make sure to update the compute capability flag to ensure you compile for the correct architecture. To find out which compute capability your GPU has, please refer to the compute capability table on Wikipedia: https://en.wikipedia.org/wiki/CUDA.

## Running the program:
After compiling the computer program, you must enter the following command line arguments:
\<dataset file name\> \<minimum frequency\> \<maximum frequency\> \<number of frequencies to search\> \<mode\> \<alpha\>
  
Modes are as follows:
* 1- [Original Super Smoother] GPU to process a single object or batch of objects
* 2- [Single-pass Super Smoother] GPU to process a single object or batch of objects
* 3- [Original Super Smoother] CPU to process a single object or batch of objects
* 4- [Single-pass Super Smoother] CPU to process a single object or batch of objects

Example execution of the original Super Smoother on the GPU (computing the periods of 136 RR-Lyrae)

```
$ ./main ../data/SDSS_stripe82_band_z.txt 0.1 10.0 330000 1 9.0

Load CUDA runtime (initialization overhead)

Dataset file: ../data/SDSS_stripe82_band_z.txt
Minimum Frequency: 0.100000
Maximum Frequency: 10.000000
Number of frequencies to test: 330000
Mode: 1
Data import: Total rows: 7064
[Device name: Quadro RTX 5000, Detecting GPU Global Memory Capacity] Size in GiB: 15.747375
[Underestimating GPU Global Memory Capacity] Size in GiB: 11.810532
Unique objects in file: 136

...

Number of objects skipped because they didn't have 4 observations: 0
Printing the best periods to file: bestperiods_SS.txt
Total time to compute batch: 47.682473
[Validation] Sum of all periods: 67.809079
```


Observe the following: 
* The program automatically detects the amount of memory on the GPU which is used to batch the execution such that global memory capacity is not exceeded. 
* The periods are output to bestperiods_SS.txt.
* The total time required to compute the periods is output to the console.
* The sum of all periods is also output. This is probably not useful; it was used for validation when testing the algorithm. It may be useful for some users, so we left it in the code.
* A summary of the execution will be stored in gpu_stats.txt.

## Example Directory
  * The example directory contains the derived periods for two implementations: 1) the original Super Smoother algorithm; and 2) the more efficient single-pass variant of the algorithm. The parameters were as follows:

Minimum Frequency: 0.100000
Maximum Frequency: 10.000000
Number of frequencies to test: 330000


## Parameters in params.h

The paper version of the code lists several parameters, whereas the release version of the code has many of the parameters selected and are thus removed for the user. Below are the parameters in the file which can be changed based on user preferences for the paper implementation. Default values are given below.


* NTHREADSCPU 16 --- Use the number of physical cores in your system. Used for parallelizing host-side tasks in the GPU implementation and it's the number of threads that will be used in the parallel CPU implementation. Values: >=1
* DTYPE double  --- The precision that will be used for the computation. Values: float or double
* NUMGPU 1 --- The number of GPUs in your system. Values: >=1
* NSTREAMSPERGPU 1 --- Streams per GPU for batching the frequences. This is used to overlap GPU and host-side tasks. It may be useful in the future if GPUs continue to increase in performance over the CPU. Values: >=1
* ORIGINALMODE -1 --- GPU Kernel used by the original algorithm.
    -1: Cascade: first try SM 1 thread per freq, then run global memory kernel (2 then 0 below)	
    0: global memory baseline
    1: shared memory for x, y, z arrays, with one small block per frequency
    2: shared memory for x, y, z arrays, one thread per frequency with small block
* SINGLEPASSMODE 0 --- GPU Kernel used by the original algorithm. 
    0: global memory baseline
    1: shared memory for x, y, z arrays, with one small block per frequency
    2: shared memory for x, y, z arrays, one thread per frequency with small block
* COALESCED 1 --- Enables coalesced memory optimization in the single pass mode when global memory baseline is enabled
    1: Enables coalesced memory optimization to x, y, z arrays.
    0: Uses the global memory baseline					
* PRINTPERIODS 2 --- Print the periods corresponding to the maximum power found in the periodogram for each object
    0: Do not print the periods
    1: Print periods to stdout
    2: Print the periods to file (bestperiods_SS.txt)  					  
* PRINTPGRAM 0 --- Enable to print the periodogram
    0: Do not print
    1: Print to pgram_SS.txt					
* OBSTHRESH 4 --- Ignore computing objects with < this number of data points (e.g., may want to ignore objects with <30 data points because solution may not be very accurate). Must be at least 4 because original Super Smoother requires this or a segmentation fault will occur. Values: >=4.
