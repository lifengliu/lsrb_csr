/*********************************
Local segmented reduction based CSR
author : Lifeng Liu
**********************************/
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"csr.h"
#include"spmv.h"
/*
Destroy CSR object
*/
void destroy_csr(CSR *csr)
{
    if(csr->ptr!=NULL)
    {
        free(csr->ptr);
        csr->ptr=NULL;
    }
} 
/*
Perform SPMV based on CSR on CPU
*/
float spmv_csr_serial(CSR * csr, float * y)
{
    int num_rows=csr->ptrlen-1;
    int i;
    float dot;
    float elapsedTime=0;
    int * ptr=csr->ptr;
    int * indices=csr->indices;
    float * data=csr->data;
    float * x=csr->X;
    int row;

#ifdef TIMING
    anonymouslib_timer ref_timer;
    ref_timer.start();
#endif
    for(row=0;row<num_rows;row++)
    {
        y[row]=0;
        dot=0;
        for(i=ptr[row];i<ptr[row+1];i++)
        {
            dot+=data[i]*x[indices[i]];
        }
        y[row]+=dot;
    }
#ifdef TIMING
    float ref_time = (float)ref_timer.stop() ;
    elapsedTime=ref_time;
#endif
    return elapsedTime;
}


/*
Perform SPMV based on CSR on CPU
*/
float spmv_csr_serial_double(CSR * csr, double * y)
{
    int num_rows=csr->ptrlen-1;
    int i;
    double dot;
    double elapsedTime=0;
    int * ptr=csr->ptr;
    int * indices=csr->indices;
    float * data=csr->data;
    float * x=csr->X;
    int row;

#ifdef TIMING
    anonymouslib_timer ref_timer;
    ref_timer.start();
#endif
    for(row=0;row<num_rows;row++)
    {
        y[row]=0;
        dot=0;
        for(i=ptr[row];i<ptr[row+1];i++)
        {
            dot+=data[i]*x[indices[i]];
        }
        y[row]+=dot;
    }
#ifdef TIMING
    float ref_time = (float)ref_timer.stop() ;
    elapsedTime=ref_time;
#endif
    return elapsedTime;
}
