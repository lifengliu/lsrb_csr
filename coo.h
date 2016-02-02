/*********************************
Local segmented reduction based CSR
author : Lifeng Liu
**********************************/
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

void destroy_coo(COO * coo);
//for debug
void show_row(COO * coo,int row);
#endif 

