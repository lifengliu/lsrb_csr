#ifndef __SPMV_H__
#define __SPMV_H__
#include "coo.h"
#include "csr.h"

#define TIMING 
#define NUM_RUN 200
//#define NUM_RUN 1 

#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }

#define BLOCK_SIZE 256
#define MAX_FILE_NAME 200
#define MAX_CHUNK 1024*1024*10 //10M

//csr5
struct anonymouslib_timer {
    cudaEvent_t start_event, stop_event;

    void start() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event, 0);
        cudaDeviceSynchronize();
    }

    float stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start_event, stop_event);
        return elapsedTime;
    }
};

typedef unsigned long int uint64_t;
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
#endif
