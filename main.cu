/*********************************
Local segmented reduction based CSR
author : Lifeng Liu
**********************************/
#include<stdio.h>
#include <time.h>
#include "spmv.h"
#include "coo.h"
#include "csr.h"
#include "read_mtx.h"
#include "balance_csr.h"

float my_abs(float a)
{   
    return (a<0.0)?-a:a;
}

double my_abs_double(double a)
{   
    return (a<0.0)?-a:a;
}

float getB(int m, int nnz)
{   
    return (float)((m + 1 + nnz) * sizeof(int) + (2 * nnz + m) * sizeof(float));
}
float getB_double(int m, int nnz)
{   
    return (float)((m + 1 + nnz) * sizeof(int) + (2 * nnz + m) * sizeof(double));
}

float getFLOP(int nnz)
{   
    return (float)(2 * nnz);
}

/*
Calculate check sum between GPU result and CPU result
*/
float check_sum(int num_rows,
                float* y,
                float* true_result)
{   
    int i;
    float sum=0.0;
    for(i=0;i<num_rows;i++)
    {   
        sum+=my_abs(y[i]-true_result[i]);
#ifdef DEBUG
        if(my_abs((y[i]-true_result[i])/true_result[i])>THRESHOLD)
        {
            printf("@%d, expecting %f, get %f \n",
                i,true_result[i],y[i]);
            exit(0);
        }
#endif
    }
    return sum;
}

/*
Calculate check sum between GPU result and CPU result
*/
double check_sum_double(int num_rows,
                double* y,
                double* true_result)
{   
    int i;
    double sum=0.0;
    for(i=0;i<num_rows;i++)
    {   
        sum+=my_abs_double(y[i]-true_result[i]);
#ifdef DEBUG
        if(my_abs_double((y[i]-true_result[i])/true_result[i])>THRESHOLD)
        {
            printf("@%d, expecting %f, get %f \n",
                i,true_result[i],y[i]);
            exit(0);
        }
#endif
    }
    return sum;
}
int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}


int main(int argc, char * argv[])
{
    char * inputfile=(char *)calloc(MAX_FILE_NAME,sizeof(char));
    char * xfile=(char *)calloc(MAX_FILE_NAME,sizeof(char));
    float *Y;
	double *Y_double;
    float *cpu_result;
	double *cpu_result_double;
    int i;
    float gpu_time;

    COO coo;
    CSR csr;


    //default optioins
    strcpy(xfile,"randomx.data");


    if(argc==1)
    {
        printf("usage: ./spmv [options] -f input_file \n");
        printf("options:\n");
        printf("-x : set the file name of x vector (default randomx.data)\n");
        exit(0);
    }
    
    if((i=ArgPos((char*)"-f",argc,argv))>0) strcpy(inputfile,argv[i+1]);
    if((i=ArgPos((char*)"-x",argc,argv))>0) strcpy(xfile,argv[i+1]);

	read_coo(inputfile,xfile,&coo);
    coo2csr(&csr,&coo);

	printf("ptrlen=%d,num cols=%d,num nonzeros=%d\n",csr.ptrlen,csr.numcols,csr.nonzeros);

    Y=(float *)calloc(csr.ptrlen-1,sizeof(float));
    Y_double=(double *)calloc(csr.ptrlen-1,sizeof(double));
    cpu_result=(float *)calloc(csr.ptrlen-1,sizeof(float));
    cpu_result_double=(double *)calloc(csr.ptrlen-1,sizeof(double));

    //get gflops
    float gb=getB(csr.ptrlen-1,csr.nonzeros);
    float gb_double=getB_double(csr.ptrlen-1,csr.nonzeros);
    float gflp=getFLOP(csr.nonzeros);

    printf("-----------------------------------------\n");
    printf("LSRB CSR single:\n");
	spmv_csr_serial(&csr,cpu_result);
    gpu_time=spmv_lsrb_csr_cuda_v3(&csr,Y);
    printf("checksum=%f\n",check_sum(csr.ptrlen-1,Y,cpu_result));
#ifdef TIMING
    printf("gpu time= %f ms,bandwidth=%f GB/s,gflops=%f\n"
            ,gpu_time,gb/(1.0e+6*gpu_time),gflp/(1.0e+6*gpu_time));
#endif
    printf("-----------------------------------------\n");
    printf("LSRB CSR double:\n");
	spmv_csr_serial_double(&csr,cpu_result_double);
    gpu_time=spmv_lsrb_csr_cuda_v3_double(&csr,Y_double);
    printf("checksum=%f\n",check_sum_double(csr.ptrlen-1,Y_double,cpu_result_double));
#ifdef TIMING
    printf("gpu time= %f ms,bandwidth=%f GB/s,gflops=%f\n"
            ,gpu_time,gb_double/(1.0e+6*gpu_time),gflp/(1.0e+6*gpu_time));
#endif



    //clean
    destroy_coo(&coo);
    destroy_csr(&csr);
    //destroy_bsr(&bsr);
    free(Y);
    free(Y_double);
    free(cpu_result);
	free(cpu_result_double);
    free(inputfile);
    free(xfile);
    return 0;
}
