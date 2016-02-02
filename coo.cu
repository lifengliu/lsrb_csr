/*********************************
Local segmented reduction based CSR
author : Lifeng Liu
**********************************/
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"spmv.h"
#include"coo.h"
/*
Destroy COO object
*/
void destroy_coo(COO * coo)
{
    if(coo->rows!=NULL)
    {
        free(coo->rows);
        coo->rows=NULL;
    }
    if(coo->cols!=NULL)
    {
        free(coo->cols);
        coo->cols=NULL;
    }
    if(coo->data!=NULL)
    {
        free(coo->data);
        coo->data=NULL;
    }
    if(coo->X!=NULL)
    {
        free(coo->X);
        coo->X=NULL;
    }
}

void show_row(COO * coo,int r)
{
    printf("row %d:\n",r);
    float sum=0;
    for(int i=0;i<coo->nonzeros;i++)
    {
        int row=coo->rows[i];
        int col=coo->cols[i];
        float data=coo->data[i];
        if(row==r)
        {
            printf("(%d,%f)%f \n",col,data,coo->X[col]);
            sum+=data*coo->X[col];
        }
    }
    printf("sum=%f\n",sum);
}

