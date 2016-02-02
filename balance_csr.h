#ifndef __BALANCE_CSR_H__
#define __BALANCE_CSR_H__

float spmv_lsrb_csr_cuda_v3(CSR * csr,float * y);
float spmv_lsrb_csr_cuda_v3_double(CSR * csr, double * y);
#endif 
