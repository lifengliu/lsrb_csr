#ifndef __CSR_H__
#define __CSR_H__
struct CSR
{
    int ptrlen;
    int numcols;
    int nonzeros;
    int *ptr;
    int *indices;
    float *data;
    float *X;
};
float spmv_csr_serial(CSR * csr, float * y);
float spmv_csr_cuda(CSR * csr,float * y, int kernel_type);
float spmv_csr_cuda_double(CSR * csr,float * y, int kernel_type);
float spmv_csr_serial_double(CSR * csr, double * y);
void destroy_csr(CSR *csr);
#endif

