NVCC = nvcc
CC=g++
NVCCFLAGS= -arch=sm_35  
CFLAGS=-Wall 
SOURCES=mmio.cpp

CUSOURCES=balance_csr.cu csr.cu coo.cu read_mtx.cu
OBJECTS=$(SOURCES:.cpp=.o)
CUOBJECTS=$(CUSOURCES:.cu=.o)
LIBS=

all:CFLAGS+=-O3
all:NVCCFLAGS+=-O3
all:target

debug:clean
debug:CFLAGS+=-g -ggdb -DDEBUG
debug:NVCCFLAGS+=-g -G -DDEBUG
debug:target

target:spmv 

spmv:$(CUOBJECTS) $(OBJECTS) main.cu
	$(NVCC) $(NVCCFLAGS) -o spmv main.cu $(CUOBJECTS) $(OBJECTS) $(LIBS)
%.o : %.cpp
	$(CC) $(CFLAGS) -c $< 
%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -dc $<


clean:
	rm -f *.o spmv 
