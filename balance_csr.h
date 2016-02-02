#ifndef __BALANCE_CSR_H__
#define __BALANCE_CSR_H__

float spmv_balance_csr_cpu(int unit_size,int num_data_blocks,CSR * csr, float * y,int * pptr);
float spmv_balance_csr_cuda(CSR * csr,float * y);
float spmv_balance_csr_cuda_v3(CSR * csr,float * y);
float spmv_balance_csr_cuda_v3_double(CSR * csr, double * y);
#endif 
