#ifndef __COO_H__ 
#define __COO_H__
#define UNIT_SIZE 512 
struct COO
{
    int numrows;
    int numcols;
    int nonzeros;
    int *rows;
    int *cols;
    float *data;
    float * X;
};
float spmv_coo_serial(COO * coo, float *y);

float spmv_coo_cuda(COO * csr,float * y,int kernel_type,
        int y_size=0,int unit_size=64);
void destroy_coo(COO * coo);
//for debug
void show_row(COO * coo,int row);
#endif 

