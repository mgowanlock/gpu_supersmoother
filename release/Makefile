
#see params.h for the parameters




SOURCES = main.cu 
CUDAOBJECTS = main.o kernel.o supsmu.o
CC = nvcc
EXECUTABLE = main

FLAGS =  -std=c++11 -Xcompiler "-W" -O3 -Xcompiler -fopenmp -arch=compute_75 -code=sm_75 -lcuda -lineinfo 
CFLAGS = -c 


all: $(EXECUTABLE)





main.o: main.cu params.h structs.h
	$(CC) $(FLAGS) $(CFLAGS) main.cu 

supsmu.o: supsmu.cu params.h structs.h supsmu.h
	$(CC) $(FLAGS) $(CFLAGS) supsmu.cu 

kernel.o: kernel.cu params.h structs.h kernel.h
	$(CC) $(FLAGS) $(CFLAGS) kernel.cu 

$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@




clean:
	rm $(CUDAOBJECTS) $(OBJECTS)
	rm main




