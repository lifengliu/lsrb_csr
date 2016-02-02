#ifndef __READ_MTX_H__
#define __READ_MTX_H__
#include "csr.h"
#include "coo.h"
void coo2csr(CSR * csr, COO * coo);
void read_coo(char * filename,char *xfilename,COO * coo);
#endif

