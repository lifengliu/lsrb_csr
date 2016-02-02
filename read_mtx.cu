#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "read_mtx.h"
#include "csr.h"
#include "coo.h"
#include "mmio.h"
#include "spmv.h"

void coo2csr(CSR * csr, COO * coo)
{
    csr->ptrlen=coo->numrows+1;
    csr->numcols=coo->numcols;
    csr->nonzeros=coo->nonzeros;
    csr->indices=coo->cols;
    csr->data=coo->data;
    csr->X=coo->X;
    csr->ptr=(int *)calloc(csr->ptrlen,sizeof(int));
    //hist
    for(int i=0;i<coo->nonzeros;i++)
    {
        csr->ptr[coo->rows[i]+1]++;
    }
    //prefix sum
    for(int i=1;i<csr->ptrlen;i++)
    {
        csr->ptr[i]+=csr->ptr[i-1];
    }
}

int compare(int a1,int a2,int b1,int b2)
{
    if(a1>b1)
        return 1;
    else if(a1<b1)
        return -1;
    else
    {   
        if(a2>b2)
            return 1;
        else if(a2<b2)
            return -1;
        else
            return 0;
    }
    return 0;
}
void swap_entry(COO * coo,int a,int b)
{
    int tmp;
    tmp=coo->rows[a];
    coo->rows[a]=coo->rows[b];
    coo->rows[b]=tmp;

    tmp=coo->cols[a];
    coo->cols[a]=coo->cols[b];
    coo->cols[b]=tmp;

    float tmpf;
    tmpf=coo->data[a];
    coo->data[a]=coo->data[b];
    coo->data[b]=tmpf;
}

void quick_sort(COO * coo,int start,int end)
{
    int rand_i=rand()%(end-start)+start;
    int povit_1=coo->rows[rand_i];
    int povit_2=coo->cols[rand_i];
    int p1=start;
    int p2=end;
    while(p2>=p1)
    {
        while(compare(coo->rows[p2],coo->cols[p2],povit_1,povit_2)==1)
            p2--;
        while(compare(coo->rows[p1],coo->cols[p1],povit_1,povit_2)==-1)
            p1++;
        if(p1<=p2)
        {
            swap_entry(coo,p2,p1);
            p2--;
            p1++;
        }
    }
    if(p2>start)
        quick_sort(coo,start,p2);
    if(p1<end)
        quick_sort(coo,p1,end);
}

bool check_sorted(COO * coo)
{   
    bool rlt=true;
    for(int i=1;i<coo->nonzeros;i++)
    {   
        if(coo->rows[i-1]>coo->rows[i])
        {   
            rlt=false;
            break;
        }
        else if(coo->rows[i-1]==coo->rows[i])
        {   
            if(coo->cols[i-1]>coo->cols[i])
            {   
                rlt=false;
                break;
            }
        }
    }
    return rlt;
}

void sort_by_row_col(COO * coo)
{
    quick_sort(coo,0,coo->nonzeros-1);
}

void read_coo(char * filename,char *xfilename,COO * coo)
{
    FILE *fp;
    MM_typecode matcode;
    if((fp=fopen(filename,"r"))==NULL)
    {
        printf("error opening file\n");
        exit(0);
    }
    if (mm_read_banner(fp, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }
    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }
    int report_nonzeros=0;
    int isPattern=0;
    int isReal=0;
    int isInteger=0;
    int isSymmetric=0;
    if (mm_read_mtx_crd_size(fp, &coo->numrows, &coo->numcols, &report_nonzeros) !=0)
        exit(1);
    if(mm_is_pattern(matcode)){isPattern=1;}
    if(mm_is_real(matcode)){isReal=1;}
    if(mm_is_integer(matcode)){isInteger=1;};
    if(mm_is_symmetric(matcode)||mm_is_hermitian(matcode)){isSymmetric=1;}
    int * rows_tmp=(int *)malloc(report_nonzeros*sizeof(int));
    int * cols_tmp=(int *)malloc(report_nonzeros*sizeof(int));
    float * data_tmp=(float *)malloc(report_nonzeros*sizeof(int));


    for(int i=0;i<report_nonzeros;i++)//&& i<MAX_CHUNK
    {
        int ival;
        int ret;
        if(isReal)
        {
            ret=fscanf(fp,"%d %d %f\n",&rows_tmp[i],&cols_tmp[i],&data_tmp[i]);
        }
        else if(isInteger)
        {
            ret=fscanf(fp,"%d %d %d\n",&rows_tmp[i],&cols_tmp[i],&ival);
            data_tmp[i]=(float)ival;
        }
        else if(isPattern)
        {
            ret=fscanf(fp,"%d %d\n",&rows_tmp[i],&rows_tmp[i]);
            data_tmp[i]=1.0;
        }
        if(ret==0)
        {
            printf("error reading\n");
            exit(0);
        }
        rows_tmp[i]--;
        cols_tmp[i]--;
    }
    //convert symmetric matrix  
    if(isSymmetric)
    {
        int num_not_diagnal=0;
        for(int i=0;i<report_nonzeros;i++)
        {
            if(rows_tmp[i]!=cols_tmp[i])
                num_not_diagnal++;
        }
        coo->nonzeros=report_nonzeros+num_not_diagnal;
    }
    else
    {
        coo->nonzeros=report_nonzeros;
    }
    coo->rows=(int *)malloc(coo->nonzeros*sizeof(int));
    coo->cols=(int *)malloc(coo->nonzeros*sizeof(int));
    coo->data=(float *)malloc(coo->nonzeros*sizeof(float));

    int curr_pos=0;
    for(int i=0;i<report_nonzeros;i++)
    {
        coo->rows[curr_pos]=rows_tmp[i];
        coo->cols[curr_pos]=cols_tmp[i];
        coo->data[curr_pos]=data_tmp[i];
        curr_pos++;
        if(isSymmetric && rows_tmp[i]!=cols_tmp[i])
        {
            coo->rows[curr_pos]=cols_tmp[i];
            coo->cols[curr_pos]=rows_tmp[i];
            coo->data[curr_pos]=data_tmp[i];
            curr_pos++;
        }
    }

    fclose(fp);
    int array_rows,array_cols;
    coo->X=(float *)malloc(coo->numcols*sizeof(float));
    if((fp=fopen(xfilename,"r"))==NULL)
        exit(0);
    if (mm_read_banner(fp, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }
    if(mm_read_mtx_array_size(fp,&array_rows,&array_cols)!=0)
        exit(1);
    if(array_rows<coo->numcols || array_cols!=1)
    {
        printf("Array size too small!\n");
        exit(1);
    }
    for(int i=0;i<coo->numcols;i++)
    {
        if(fscanf(fp,"%f\n",&coo->X[i])==0)
        {
            printf("reading error\n");
            exit(0);
        }
    }
    fclose(fp);
    sort_by_row_col(coo);
    assert(check_sorted(coo)==true);
    //free  
    free(rows_tmp);
    free(cols_tmp);
    free(data_tmp);
}

