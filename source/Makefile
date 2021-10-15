
#####################################
#Build binaries for the code in the paper
#see params.h for the parameters
SOURCES = main.cu 
CUDAOBJECTS = main.o kernel.o supsmu.o
CC = nvcc
EXECUTABLE = main

#select the compute capability of your GPU
COMPUTE_CAPABILITY = 75

FLAGS =  -std=c++14 -Xcompiler "-W" -O3 -Xcompiler -fopenmp -use_fast_math -lcuda -lineinfo 

#used for the C version of the code
COMPUTE_CAPABILITY_FLAGS = -arch=compute_$(COMPUTE_CAPABILITY) -code=sm_$(COMPUTE_CAPABILITY)
CFLAGS = -c 

#used for the Python wrapper to maximize GPU compatibility
CUDAARCHFLAGS = -arch=compute_$(COMPUTE_CAPABILITY) \
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_70,code=sm_70 \
-gencode=arch=compute_72,code=sm_72 \
-gencode=arch=compute_75,code=sm_75 \
-gencode=arch=compute_80,code=sm_80 \
#put future generations here.

all: $(EXECUTABLE)

main.o: main.cu params.h structs.h
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) main.cu 

supsmu.o: supsmu.cu params.h structs.h supsmu.h
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) supsmu.cu 

kernel.o: kernel.cu params.h structs.h kernel.h
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) kernel.cu 

$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@


###################################
#target for creating shared library for Python

#These parameters should be changed as a function of the machine
#NUMGPU- the number of GPUs (default=1), 
#BETA- global memory underestimation fraction (default=0.75), 
#NTHREADSCPU- the number of physical cores on the platform (default=16)
PARAMSFORSYSTEM=\
-DNUMGPU=1\
-DBETA=0.75\
-DNTHREADSCPU=16\




#Default parameter set; parameters achieve the best performance, and return the periods and pgram
PARAMSDEFAULTS=\
-DSMALLBLOCKSIZE=32\
-DLARGEBLOCKSIZE=1024\
-DNSTREAMSPERGPU=1\
-DORIGINALMODE=-1\
-DSINGLEPASSMODE=0\
-DCOALESCED=1\
-DPRINTPERIODS=0\
-DPRINTPGRAM=0\
-DOBSTHRESH=4\


#make the float and double targets
make_python_shared_libs: make_python_float make_python_double

make_python_float: main.cu structs.h supsmu.cu supsmu.h
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DDTYPE=float $(FLAGS) $(CUDAARCHFLAGS) -c main.cu 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DDTYPE=float $(FLAGS) $(CUDAARCHFLAGS) -c kernel.cu 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DDTYPE=float $(FLAGS) $(CUDAARCHFLAGS) -c supsmu.cu
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DDTYPE=float $(FLAGS) $(CUDAARCHFLAGS) main.o kernel.o supsmu.o -o libpysupsmufloat.so

make_python_double: main.cu structs.h supsmu.cu supsmu.h
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DDTYPE=double $(FLAGS) $(CUDAARCHFLAGS) -c main.cu 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DDTYPE=double $(FLAGS) $(CUDAARCHFLAGS) -c kernel.cu 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DDTYPE=double $(FLAGS) $(CUDAARCHFLAGS) -c supsmu.cu
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DDTYPE=double $(FLAGS) $(CUDAARCHFLAGS) main.o kernel.o supsmu.o -o libpysupsmudouble.so	
	

###################################


clean:
	rm $(CUDAOBJECTS) $(OBJECTS)
	rm main libpysupsmufloat.so libpysupsmudouble.so




