/*********************************
Local segmented reduction based CSR
author : Lifeng Liu
**********************************/
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<assert.h>
#include"csr.h"
#include"spmv.h"
#include"balance_csr.h"

#define UNIT_SIZE_B 32 
//function declarations
/*
Intra warp prefix scan
in: Input value to be scanned
idInWarp: Thread id in warp
return value: Scanned result
*/
__device__ uint32_t inwarp_scan_v3(uint32_t in,int idInWarp)
{
    for(int i=1;i<=16;i=i*2)
    {
        uint32_t d=__shfl_up(in,i,32);
        if(idInWarp>=i) in+=d;
    }
    return in;
}
/*
Do SPMV on LSRB-CSR format
num_blocks_per_warp: number of data blocks assigned to each warp
num_warps: total number of warps
num_rows: number of rows in matrix
block_base,bit_map,seg_offset: Three auxiliary  arrays of LSRB-CSR
in: Input data
indices: Indices array
dev_x: Input X vector
out: output array
tmp_row,tmp_data: reserved for future
*/
__global__ void spmv_lsrb_csr_gpu_v3(int num_blocks_per_warp,
	int num_warps,int num_rows,
	const uint32_t * block_base, const uint32_t *bit_map, const int * seg_offset,
	const float * in, const int * indices,const float *dev_x,
	float *out,
	int * tmp_row, float * tmp_data)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int wid=tid/32;
	if(wid>=num_warps)
		return;
	int idInWarp=tid%32;
	uint32_t row_base=__ldg(&block_base[wid]);
	int block_index_base=wid*num_blocks_per_warp;
	int bit_maps=bit_map[block_index_base+idInWarp];
	float results=0.0;
	uint32_t rows=0;
	if((row_base&0x80000000)==0)
	{
		#pragma unroll
		for(int i=0;i<num_blocks_per_warp;i++)
		{
			uint32_t block_bit_map=__shfl(bit_maps,i,32);
			//carry
			uint32_t carry=block_bit_map&0x01;
			if(carry)
			{
				if(idInWarp==31)
					out[row_base+rows]=results;
				results=0.0;
			}
			else
			{
				results=__shfl(results,31,32);
				results=(idInWarp==0)?results:0.0;
			}
			//get results
			int data_block_index=(block_index_base+i)*32+idInWarp;
			float x=__ldg(&dev_x[indices[data_block_index]]);
			results+=in[data_block_index]*x;
			if(block_bit_map!=0)
			{
				//get rows
				rows=__shfl(rows,31,32);
				int move=31-idInWarp;
				block_bit_map=(block_bit_map<<move)>>move;
				uint32_t d;
				asm("popc.b32 %0,%1;":"=r"(d):"r"(block_bit_map));
				
				rows+=d;
				uint32_t rows_s=__shfl_down(rows,1,32);
				rows_s=(idInWarp==31)?rows:rows_s;
				//seg reduce
				#pragma unroll
				for(int j=1;j<=16;j=j*2)
				{
					float data_s=__shfl_up(results,j,32);
					uint32_t rows_t=__shfl_up(rows,j,32);
					if(idInWarp>=j && rows==rows_t) results+=data_s;
				}
				//write back
				if(rows!=rows_s)
					out[row_base+rows]=results;
				}
			else
			{
				#pragma unrool
				for(int r=16;r>0;r>>=1)
				{
					results+=__shfl_xor(results,r,32);
				}
			}
		}					
		if(idInWarp==31)
		{
	//		tmp_row[wid]=row_base+rows;
	//		tmp_data[wid]=results;
			atomicAdd(&out[row_base+rows],results);
		}
	}
	else
	{
		row_base=row_base&0x7fffffff;
		#pragma unroll
		for(int i=0;i<num_blocks_per_warp;i++)
		{
			uint32_t block_bit_map=__shfl(bit_maps,i,32);
			//carry
			uint32_t carry=block_bit_map&0x01;
			if(carry)
			{
				if(idInWarp==31)
					out[row_base+rows]=results;
				results=0.0;
			}
			else
			{
				results=__shfl(results,31,32);
				results=(idInWarp==0)?results:0.0;
			}
			//get results
			int data_block_index=(block_index_base+i)*32+idInWarp;
			float x=__ldg(&dev_x[indices[data_block_index]]);
			results+=in[data_block_index]*x;
			if(block_bit_map!=0)
			{
				//get rows
				rows=__shfl(rows,31,32);
				int move=31-idInWarp;
				block_bit_map=(block_bit_map<<move)>>move;
				uint32_t d;
				asm("popc.b32 %0,%1;":"=r"(d):"r"(block_bit_map));
				
				rows+=d;
				uint32_t rows_s=__shfl_down(rows,1,32);
				rows_s=(idInWarp==31)?rows:rows_s;
				//seg reduce
				#pragma unroll
				for(int j=1;j<=16;j=j*2)
				{
					float data_s=__shfl_up(results,j,32);
					uint32_t rows_t=__shfl_up(rows,j,32);
					if(idInWarp>=j && rows==rows_t) results+=data_s;
				}
				//write back
				if(rows!=rows_s)
					out[row_base+seg_offset[row_base+rows]]=results;
				}
			else
			{
				#pragma unrool
				for(int r=16;r>0;r>>=1)
				{
					results+=__shfl_xor(results,r,32);
				}
			}
		}					
		if(idInWarp==31)
		{
	//		tmp_row[wid]=row_base+rows;
	//		tmp_data[wid]=results;
			int add=row_base+seg_offset[row_base+rows];
			if(add<num_rows)
			atomicAdd(&out[add],results);
		}
	}
}
/*
Generate bitmap from ptr array
*/
__global__ void get_bit_map_seg_offset_gpu(
			int num_data_per_block,
			int ptrlen,
			int num_warps,
			int *dev_ptr,
			uint32_t *dev_block_base,
			uint32_t *dev_bit_map)
{
	extern __shared__ uint32_t s[];
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int wid=tid/32;
	int num_warps_in_block=blockDim.x/32;
	int wid_in_block=wid%num_warps_in_block;
	if(wid>=num_warps)
		return;
	int idInWarp=tid%32;
	uint32_t *tmp=s+wid_in_block*33;//tmp[0] as carry
	if(idInWarp==0)
		tmp[0]=0;
	uint32_t * has_empty_rows=s+num_warps_in_block*33+wid_in_block;	
	*has_empty_rows=0;
	uint32_t base=__ldg(&dev_block_base[wid]);
	uint32_t end=__ldg(&dev_block_base[wid+1]);
	//fast_track here
	for(int iteration=base;iteration<end;iteration+=32)//it will overlap automatically
	{
		int pos=idInWarp+iteration;
		int row=0xfffffff;
		if(pos<=end)
		{
			//get bit_map
			uint32_t bit=0;
			row=__ldg(&dev_ptr[pos]);
			if(row%num_data_per_block!=0)
			{
				int block=row/32;
				int offset=row%32;
				bit=1<<offset;
				atomicOr(&dev_bit_map[block],bit);
			}
		}
		int row_next=__shfl_down(row,1,32);
		if(pos<end)
		{
			if(idInWarp==31)
				row_next=__ldg(&dev_ptr[pos+1]);
			if(row==row_next)
			{
				*has_empty_rows=1;
			}
		}
	}
	if(*has_empty_rows!=0)
	{
	for(int iteration=base;iteration<end;iteration+=32)
	{
		int pos=idInWarp+iteration;
		//map
		if(pos<end)
		{
			tmp[idInWarp+1]=(__ldg(&dev_ptr[pos])!=__ldg(&dev_ptr[pos+1]))?1:0;
		}
		//scan
		for(int i=1;i<=16;i=i*2)
		{
			if(idInWarp>=i) tmp[idInWarp+1]+=tmp[idInWarp+1-i];
		}
		tmp[idInWarp+1]+=tmp[0];
		//compact
		if(pos<end)
		{
			if(tmp[idInWarp]!=tmp[idInWarp+1])
				dev_ptr[base+tmp[idInWarp]]=pos-base;
		}
		if(idInWarp==0)
			tmp[0]=tmp[32];
	}
	if(idInWarp==0)
		dev_block_base[wid]=dev_block_base[wid]|0x80000000;
	}
}
/*
Generate segment offset from ptr array
*/
__global__ void get_seg_offset_gpu(
			int num_warps,
			int *dev_ptr,
			uint32_t *dev_block_base)
{
	extern __shared__ uint32_t s[];
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int wid=tid/32;
	int num_warps_in_block=blockDim.x/32;
	int wid_in_block=wid%num_warps_in_block;
	if(wid>=num_warps)
		return;
	int idInWarp=tid%32;
	uint32_t *tmp=s+wid_in_block*33;//tmp[0] as carry
	if(idInWarp==0)
		tmp[0]=0;
	uint32_t * has_empty_rows=s+num_warps_in_block*33+wid_in_block;	
	*has_empty_rows=0;
	uint32_t base=__ldg(&dev_block_base[wid]);
	uint32_t end=__ldg(&dev_block_base[wid+1]);
	//fast_track here
	for(int iteration=base;iteration<end;iteration+=32)//it will overlap automatically
	{
		int pos=idInWarp+iteration;
		if(pos<end)
		{
			if(__ldg(&dev_ptr[pos])==__ldg(&dev_ptr[pos+1]))
			{
				*has_empty_rows=1;
			}
		}
		if(*has_empty_rows!=0)
			break;
	}
	if(*has_empty_rows!=0)
	{
	for(int iteration=base;iteration<end;iteration+=32)
	{
		int pos=idInWarp+iteration;
		//map
		if(pos<end)
		{
			tmp[idInWarp+1]=(__ldg(&dev_ptr[pos])!=__ldg(&dev_ptr[pos+1]))?1:0;
		}
		//scan
		for(int i=1;i<=16;i=i*2)
		{
			if(idInWarp>=i) tmp[idInWarp+1]+=tmp[idInWarp+1-i];
		}
		tmp[idInWarp+1]+=tmp[0];
		//compact
		if(pos<end)
		{
			if(tmp[idInWarp]!=tmp[idInWarp+1])
				dev_ptr[base+tmp[idInWarp]]=pos-base;
		}
		if(idInWarp==0)
			tmp[0]=tmp[32];
	}
	if(idInWarp==0)
		dev_block_base[wid]=dev_block_base[wid]|0x80000000;
	}
}
/*
Generate block base array from ptr array
*/

__global__ void get_block_base_gpu(int num_data_per_block,
			int ptrlen,int num_data_blocks,int *dev_ptr,
			uint32_t * dev_block_base)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	if(tid<num_data_blocks)
	{
		int data_base=tid*num_data_per_block;
		uint32_t a=0;
		uint32_t b=ptrlen;
		while(b>a+1)
		{
			uint32_t c=(a+b)>>1;
			if(__ldg(&dev_ptr[c])>data_base)
				b=c;
			else
				a=c;
		}
		dev_block_base[tid]=a; 	
	}
}


/*
Generate bit map array from ptr array
*/
__global__ void get_bit_map_gpu(int num_data_per_block,
		int ptrlen,int *dev_ptr,
		uint32_t *dev_bit_map)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	if(tid<ptrlen)
	{
		uint32_t bit=0;
		int row=__ldg(&dev_ptr[tid]);
		if(row%num_data_per_block!=0)
		{
			int block=row/32;
			int offset=row%32;
			bit=1<<offset;
			atomicOr(&dev_bit_map[block],bit);
		}
	}
}

/*
Get the statistics of how much rows each block has
result stores in bins size=32 [0--32),[32--64).....
*/
void get_rows_hist(int num_warps,uint32_t * bit_maps, int * bins)
{
	int num_blocks_per_warp=32;
	for(int i=0;i<32;i++)
		bins[i]=0;
	for(int i=0;i<num_warps;i++)
	{
		int base=i*num_blocks_per_warp;//32 should be num_blocks_per_warp
		int sum=0;
		for(int j=0;j<num_blocks_per_warp;j++)
		{
			uint32_t bit_map=bit_maps[base+j];
			for(int b=0;b<32;b++)
				if((bit_map>>b)&0x1!=0)
					sum++;
		}
		bins[sum/32]++;
	}	
}
/*
Wrapper function for SPMV on LSRB-CSR
y: Output vector
return value: time elapsed
*/
float spmv_lsrb_csr_cuda_v3(CSR * csr, float * y)
{
	//test
	//spmv_balance_csr_cuda_v3_test(csr,y);
	//return 0.0;

	float elapsedTime=0.0;
	int unit_size=UNIT_SIZE_B;
	
	int num_warps_per_block=8;
	int num_blocks_per_warp=32;
	int warp_data_size=unit_size*32;
	
	int padded_nonzeros=((csr->nonzeros-1)/warp_data_size+1)
		*warp_data_size;
	int num_data_blocks=padded_nonzeros/unit_size;
	int num_warps=padded_nonzeros/warp_data_size;
	int ptrlen=csr->ptrlen;
	int num_blocks=(num_warps-1)/num_warps_per_block+1;
		//number of thread blocks

	uint32_t *dev_block_base;
	uint32_t *dev_bit_map;
	int * dev_ptr;
	//format convert
	//step1 malloc
	cudaMalloc((void **)(&dev_block_base),(num_warps+1)*sizeof(uint32_t));
	cudaMalloc((void **)(&dev_bit_map)
		,num_data_blocks*sizeof(uint32_t));	
	//step2 get block_base
	float block_base_time=0.0;
	cudaMalloc((void **)(&dev_ptr),ptrlen*sizeof(int));
	cudaMemcpy(dev_ptr,csr->ptr,ptrlen*sizeof(int),
		cudaMemcpyHostToDevice);
	int thread_block_size_block_base=256;
	int num_thread_blocks_block_base=(num_warps+1-1)/thread_block_size_block_base+1;
	dim3 dimBlock_block_base(thread_block_size_block_base,1);
	dim3 dimGrid_block_base(num_thread_blocks_block_base,1);
#ifdef TIMING
	anonymouslib_timer block_base_timer;
	block_base_timer.start();
#endif
	get_block_base_gpu<<<dimGrid_block_base,dimBlock_block_base>>>
		(warp_data_size,ptrlen,num_warps+1,
		dev_ptr,dev_block_base);
#ifdef TIMING
	block_base_time=(float)block_base_timer.stop();
#endif
	printf("Get block_base time=%f ms\n",block_base_time);
	CUT_CHECK_ERROR("kernel error:");
	//step3 get bit_map
	int thread_block_size_bit_map=256;
	int num_thread_blocks_bit_map=(ptrlen-1)/thread_block_size_bit_map+1;
	dim3 dimBlock_bit_map(thread_block_size_bit_map,1);
	dim3 dimGrid_bit_map(num_thread_blocks_bit_map,1);

	float get_seg_offset_time=0.0;
	int thread_block_size_seg_offset=256;
	int num_warps_per_block_seg_offset=thread_block_size_seg_offset/32;
	int num_thread_blocks_seg_offset=(ptrlen-1)/thread_block_size_seg_offset+1;
	dim3 dimBlock_seg_offset(thread_block_size_seg_offset,1);
	dim3 dimGrid_seg_offset(num_thread_blocks_seg_offset,1);
	int shared_size=num_warps_per_block_seg_offset*(33+1)*sizeof(uint32_t);

	//warm up
	for(int i=0;i<NUM_RUN;i++)
	{
		cudaMemset(dev_bit_map,0,num_data_blocks*sizeof(uint32_t));
		get_bit_map_gpu<<<dimGrid_bit_map,dimBlock_bit_map>>>(
			warp_data_size,ptrlen,dev_ptr,dev_bit_map);
	}
	//step 3.5 get seg_offset
#ifdef TIMING
	anonymouslib_timer get_seg_offset_timer;
	get_seg_offset_timer.start();
#endif
	cudaMemset(dev_bit_map,0,num_data_blocks*sizeof(uint32_t));
	get_bit_map_seg_offset_gpu<<<dimGrid_seg_offset,
		dimBlock_seg_offset,shared_size>>>(
		warp_data_size,
		ptrlen,
		num_warps,
		dev_ptr,
		dev_block_base,
		dev_bit_map);
#ifdef TIMING
	get_seg_offset_time=(float)get_seg_offset_timer.stop();
#endif
	printf("Get bit_map seg_offset time=%f ms\n",get_seg_offset_time);
	printf("csr->csr_balance time=%f ms\n",block_base_time
		+get_seg_offset_time);
	CUT_CHECK_ERROR("kernel error format convert:");
	

	//calculate
	// gpu arrays
	//int * dev_ptr;
	int * dev_indices;
	float * dev_data;
	float * dev_x;
	float * dev_y;
	int  * tmp_row;
	float * tmp_data;
	int num_rows=csr->ptrlen-1;
	int num_cols=csr->numcols;
#ifdef TIMING
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif	
	//cudaMalloc((void **)(&dev_ptr),csr->ptrlen*sizeof(int));
	cudaMalloc((void **)(&dev_indices),padded_nonzeros*sizeof(int));
	cudaMemset(dev_indices+padded_nonzeros-warp_data_size,0
		,warp_data_size*sizeof(int));
	cudaMalloc((void **)(&dev_data),padded_nonzeros*sizeof(float));
	cudaMemset(dev_data+padded_nonzeros-warp_data_size,0
		,warp_data_size*sizeof(float));
	cudaMalloc((void **)(&dev_x),num_cols*sizeof(float));
	cudaMalloc((void **)(&dev_y),num_rows*sizeof(float));
	cudaMalloc((void **)(&tmp_row),num_warps*sizeof(int));
	cudaMalloc((void **)(&tmp_data),num_warps*sizeof(float));

	cudaMemcpy(dev_indices,csr->indices,csr->nonzeros*sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_data,csr->data,csr->nonzeros*sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x,csr->X,num_cols*sizeof(float),
		cudaMemcpyHostToDevice);

	//kernel
	dim3 dimBlock(num_warps_per_block*32,1);
	dim3 dimGrid(num_blocks,1);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#ifdef TIMING
	cudaEventRecord(start,0);
#endif
	for(int i=0;i<NUM_RUN;i++)
	{
		spmv_lsrb_csr_gpu_v3<<<dimGrid,dimBlock>>>(
			num_blocks_per_warp,
			num_warps,
			ptrlen-1,	
			dev_block_base,
			dev_bit_map,
			dev_ptr,
			dev_data,
			dev_indices,
			dev_x,
			dev_y,
			tmp_row,
			tmp_data);	
	}
#ifdef TIMING
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
#endif

#ifdef TIMING
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	elapsedTime/=NUM_RUN;
#endif
		//get the correct result
	
	cudaMemset(dev_y,0,num_rows*sizeof(float));
	spmv_lsrb_csr_gpu_v3<<<dimGrid,dimBlock>>>(
			num_blocks_per_warp,
			num_warps,
			ptrlen-1,	
			dev_block_base,
			dev_bit_map,
			dev_ptr,
			dev_data,
			dev_indices,
			dev_x,
			dev_y,
			tmp_row,
			tmp_data);	
	//get data
	int *h_tmp_row=(int *)calloc(num_warps,sizeof(int));
	float *h_tmp_data=(float *)calloc(num_warps,sizeof(float));
	cudaMemcpy(y,dev_y,num_rows*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tmp_row,tmp_row,num_warps*sizeof(int),
		cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tmp_data,tmp_data,num_warps*sizeof(float),
		cudaMemcpyDeviceToHost);
	
	//post scan
/*
	for(int i=0;i<num_warps;i++)
	{
		y[h_tmp_row[i]]+=h_tmp_data[i];
	}
*/
	CUT_CHECK_ERROR("kernel error:");
	

	//free
	cudaFree(dev_indices);
	cudaFree(dev_data);
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_bit_map);
	cudaFree(dev_block_base);
	cudaFree(tmp_row);
	cudaFree(tmp_data);	

	free(h_tmp_row);
	free(h_tmp_data);
	return elapsedTime;
}


/*
Atomic operation on double values
*/
__device__ double atomicAdd_double(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do{
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val +
            __longlong_as_double(assumed)));
        } while (assumed != old);
    return __longlong_as_double(old);
}

/*
Kernel for SPMV based on LSRB-CSR
*/
__global__ void spmv_lsrb_csr_gpu_v3_double(int num_blocks_per_warp,
	int num_warps,int num_rows,
	const uint32_t * block_base, const uint32_t *bit_map, const int * seg_offset,
	const double * in, const int * indices,const double *dev_x,
	double *out,
	int * tmp_row, double * tmp_data)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int wid=tid/32;
	if(wid>=num_warps)
		return;
	int idInWarp=tid%32;
	uint32_t row_base=__ldg(&block_base[wid]);
	int block_index_base=wid*num_blocks_per_warp;
	int bit_maps=bit_map[block_index_base+idInWarp];
	double results=0.0;
	uint32_t rows=0;
	if((row_base&0x80000000)==0)
	{
		#pragma unroll
		for(int i=0;i<num_blocks_per_warp;i++)
		{
			uint32_t block_bit_map=__shfl(bit_maps,i,32);
			//carry
			uint32_t carry=block_bit_map&0x01;
			if(carry)
			{
				if(idInWarp==31)
					out[row_base+rows]=results;
				results=0.0;
			}
			else
			{
				results=__shfl(results,31,32);
				results=(idInWarp==0)?results:0.0;
			}
			//get results
			int data_block_index=(block_index_base+i)*32+idInWarp;
			double x=__ldg(&dev_x[indices[data_block_index]]);
			results+=in[data_block_index]*x;
			if(block_bit_map!=0)
			{
				//get rows
				rows=__shfl(rows,31,32);
				int move=31-idInWarp;
				block_bit_map=(block_bit_map<<move)>>move;
				uint32_t d;
				asm("popc.b32 %0,%1;":"=r"(d):"r"(block_bit_map));
				
				rows+=d;
				uint32_t rows_s=__shfl_down(rows,1,32);
				rows_s=(idInWarp==31)?rows:rows_s;
				//seg reduce
				#pragma unroll
				for(int j=1;j<=16;j=j*2)
				{
					double data_s=__shfl_up(results,j,32);
					uint32_t rows_t=__shfl_up(rows,j,32);
					if(idInWarp>=j && rows==rows_t) results+=data_s;
				}
				//write back
				if(rows!=rows_s)
					out[row_base+rows]=results;
				}
			else
			{
				#pragma unrool
				for(int r=16;r>0;r>>=1)
				{
					results+=__shfl_xor(results,r,32);
				}
			}
		}					
		if(idInWarp==31)
		{
	//		tmp_row[wid]=row_base+rows;
	//		tmp_data[wid]=results;
			atomicAdd_double(&out[row_base+rows],results);
		}
	}
	else
	{
		row_base=row_base&0x7fffffff;
		#pragma unroll
		for(int i=0;i<num_blocks_per_warp;i++)
		{
			uint32_t block_bit_map=__shfl(bit_maps,i,32);
			//carry
			uint32_t carry=block_bit_map&0x01;
			if(carry)
			{
				if(idInWarp==31)
					out[row_base+rows]=results;
				results=0.0;
			}
			else
			{
				results=__shfl(results,31,32);
				results=(idInWarp==0)?results:0.0;
			}
			//get results
			int data_block_index=(block_index_base+i)*32+idInWarp;
			double x=__ldg(&dev_x[indices[data_block_index]]);
			results+=in[data_block_index]*x;
			if(block_bit_map!=0)
			{
				//get rows
				rows=__shfl(rows,31,32);
				int move=31-idInWarp;
				block_bit_map=(block_bit_map<<move)>>move;
				uint32_t d;
				asm("popc.b32 %0,%1;":"=r"(d):"r"(block_bit_map));
				
				rows+=d;
				uint32_t rows_s=__shfl_down(rows,1,32);
				rows_s=(idInWarp==31)?rows:rows_s;
				//seg reduce
				#pragma unroll
				for(int j=1;j<=16;j=j*2)
				{
					double data_s=__shfl_up(results,j,32);
					uint32_t rows_t=__shfl_up(rows,j,32);
					if(idInWarp>=j && rows==rows_t) results+=data_s;
				}
				//write back
				if(rows!=rows_s)
					out[row_base+seg_offset[row_base+rows]]=results;
				}
			else
			{
				#pragma unrool
				for(int r=16;r>0;r>>=1)
				{
					results+=__shfl_xor(results,r,32);
				}
			}
		}					
		if(idInWarp==31)
		{
	//		tmp_row[wid]=row_base+rows;
	//		tmp_data[wid]=results;
			int add=row_base+seg_offset[row_base+rows];
			if(add<num_rows)
			atomicAdd_double(&out[add],results);
		}
	}
}

/*
Wrapper function for SPMV on LSRB-CSR
y: Output vector
return value: time elapsed
*/
float spmv_lsrb_csr_cuda_v3_double(CSR * csr, double * y)
{
	//test
	//spmv_balance_csr_cuda_v3_test(csr,y);
	//return 0.0;

	float elapsedTime=0.0;
	int unit_size=UNIT_SIZE_B;
	
	int num_warps_per_block=8;
	int num_blocks_per_warp=32;
	int warp_data_size=unit_size*32;
	
	int padded_nonzeros=((csr->nonzeros-1)/warp_data_size+1)
		*warp_data_size;
	int num_data_blocks=padded_nonzeros/unit_size;
	int num_warps=padded_nonzeros/warp_data_size;
	int ptrlen=csr->ptrlen;
	int num_blocks=(num_warps-1)/num_warps_per_block+1;
		//number of thread blocks

	uint32_t *dev_block_base;
	uint32_t *dev_bit_map;
	int * dev_ptr;
	//format convert
	//step1 malloc
	cudaMalloc((void **)(&dev_block_base),(num_warps+1)*sizeof(uint32_t));
	cudaMalloc((void **)(&dev_bit_map)
		,num_data_blocks*sizeof(uint32_t));	
	//step2 get block_base
	float block_base_time=0.0;
	cudaMalloc((void **)(&dev_ptr),ptrlen*sizeof(int));
	cudaMemcpy(dev_ptr,csr->ptr,ptrlen*sizeof(int),
		cudaMemcpyHostToDevice);
	int thread_block_size_block_base=256;
	int num_thread_blocks_block_base=(num_warps+1-1)/thread_block_size_block_base+1;
	dim3 dimBlock_block_base(thread_block_size_block_base,1);
	dim3 dimGrid_block_base(num_thread_blocks_block_base,1);
#ifdef TIMING
	anonymouslib_timer block_base_timer;
	block_base_timer.start();
#endif
	get_block_base_gpu<<<dimGrid_block_base,dimBlock_block_base>>>
		(warp_data_size,ptrlen,num_warps+2,
		dev_ptr,dev_block_base);
#ifdef TIMING
	block_base_time=(float)block_base_timer.stop();
#endif
	printf("Get block_base time=%f ms\n",block_base_time);
	CUT_CHECK_ERROR("kernel error:");
	//step3 get bit_map
	int thread_block_size_bit_map=256;
	int num_thread_blocks_bit_map=(ptrlen-1)/thread_block_size_bit_map+1;
	dim3 dimBlock_bit_map(thread_block_size_bit_map,1);
	dim3 dimGrid_bit_map(num_thread_blocks_bit_map,1);

	float get_seg_offset_time=0.0;
	int thread_block_size_seg_offset=256;
	int num_warps_per_block_seg_offset=thread_block_size_seg_offset/32;
	int num_thread_blocks_seg_offset=(ptrlen-1)/thread_block_size_seg_offset+1;
	dim3 dimBlock_seg_offset(thread_block_size_seg_offset,1);
	dim3 dimGrid_seg_offset(num_thread_blocks_seg_offset,1);
	int shared_size=num_warps_per_block_seg_offset*(33+1)*sizeof(uint32_t);

	//warm up
	for(int i=0;i<NUM_RUN;i++)
	{
		cudaMemset(dev_bit_map,0,num_data_blocks*sizeof(uint32_t));
		get_bit_map_gpu<<<dimGrid_bit_map,dimBlock_bit_map>>>(
			warp_data_size,ptrlen,dev_ptr,dev_bit_map);
	}
	//step 3.5 get seg_offset
#ifdef TIMING
	anonymouslib_timer get_seg_offset_timer;
	get_seg_offset_timer.start();
#endif
	cudaMemset(dev_bit_map,0,num_data_blocks*sizeof(uint32_t));
	get_bit_map_seg_offset_gpu<<<dimGrid_seg_offset,
		dimBlock_seg_offset,shared_size>>>(
		warp_data_size,
		ptrlen,
		num_warps,
		dev_ptr,
		dev_block_base,
		dev_bit_map);
#ifdef TIMING
	get_seg_offset_time=(float)get_seg_offset_timer.stop();
#endif
	printf("Get bit_map seg_offset time=%f ms\n",get_seg_offset_time);
	printf("csr->csr_balance time=%f ms\n",block_base_time
		+get_seg_offset_time);
	CUT_CHECK_ERROR("kernel error format convert:");
	

	//calculate
	// gpu arrays
	//int * dev_ptr;
	int * dev_indices;
	double * dev_data;
	double * dev_x;
	double * dev_y;
	int  * tmp_row;
	double * tmp_data;
	int num_rows=csr->ptrlen-1;
	int num_cols=csr->numcols;
#ifdef TIMING
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif	
	//cudaMalloc((void **)(&dev_ptr),csr->ptrlen*sizeof(int));
	cudaMalloc((void **)(&dev_indices),padded_nonzeros*sizeof(int));
	cudaMemset(dev_indices+padded_nonzeros-warp_data_size,0
		,warp_data_size*sizeof(int));
	cudaMalloc((void **)(&dev_data),padded_nonzeros*sizeof(double));
	cudaMemset(dev_data+padded_nonzeros-warp_data_size,0
		,warp_data_size*sizeof(double));
	cudaMalloc((void **)(&dev_x),num_cols*sizeof(double));
	cudaMalloc((void **)(&dev_y),num_rows*sizeof(double));
	cudaMalloc((void **)(&tmp_row),num_warps*sizeof(int));
	cudaMalloc((void **)(&tmp_data),num_warps*sizeof(double));

	cudaMemcpy(dev_indices,csr->indices,csr->nonzeros*sizeof(int),
		cudaMemcpyHostToDevice);
	double * double_tmp_data=(double *)calloc(csr->nonzeros,sizeof(double));
	for(int i=0;i<csr->nonzeros;i++)
		double_tmp_data[i]=(double)csr->data[i];
	cudaMemcpy(dev_data,double_tmp_data,csr->nonzeros*sizeof(double),
		cudaMemcpyHostToDevice);
	free(double_tmp_data);
	double * tmp_x=(double *)calloc(num_cols,sizeof(double));
	for(int i=0;i<num_cols;i++)
		tmp_x[i]=(double)csr->X[i];
	cudaMemcpy(dev_x,tmp_x,num_cols*sizeof(double),
		cudaMemcpyHostToDevice);
	free(tmp_x);

	//kernel
	dim3 dimBlock(num_warps_per_block*32,1);
	dim3 dimGrid(num_blocks,1);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#ifdef TIMING
	cudaEventRecord(start,0);
#endif
	for(int i=0;i<NUM_RUN;i++)
	{
		spmv_lsrb_csr_gpu_v3_double<<<dimGrid,dimBlock>>>(
			num_blocks_per_warp,
			num_warps,
			ptrlen-1,	
			dev_block_base,
			dev_bit_map,
			dev_ptr,
			dev_data,
			dev_indices,
			dev_x,
			dev_y,
			tmp_row,
			tmp_data);	
	}
#ifdef TIMING
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
#endif

#ifdef TIMING
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	elapsedTime/=NUM_RUN;
#endif
		//get the correct result
	
	cudaMemset(dev_y,0,num_rows*sizeof(double));
	spmv_lsrb_csr_gpu_v3_double<<<dimGrid,dimBlock>>>(
			num_blocks_per_warp,
			num_warps,
			ptrlen-1,	
			dev_block_base,
			dev_bit_map,
			dev_ptr,
			dev_data,
			dev_indices,
			dev_x,
			dev_y,
			tmp_row,
			tmp_data);	
	//get data
	int *h_tmp_row=(int *)calloc(num_warps,sizeof(int));
	double *h_tmp_data=(double *)calloc(num_warps,sizeof(double));
	

	double *tmp_y=(double *)calloc(num_rows,sizeof(double));
	cudaMemcpy(tmp_y,dev_y,num_rows*sizeof(double),cudaMemcpyDeviceToHost);
	for(int i=0;i<num_rows;i++)
		y[i]=(double)tmp_y[i];
	free(tmp_y);
	
	cudaMemcpy(h_tmp_row,tmp_row,num_warps*sizeof(int),
		cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tmp_data,tmp_data,num_warps*sizeof(double),
		cudaMemcpyDeviceToHost);
	
	//post scan
/*
	for(int i=0;i<num_warps;i++)
	{
		y[h_tmp_row[i]]+=h_tmp_data[i];
	}
*/
	CUT_CHECK_ERROR("kernel error:");
	

	//free
	cudaFree(dev_indices);
	cudaFree(dev_data);
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_bit_map);
	cudaFree(dev_block_base);
	cudaFree(tmp_row);
	cudaFree(tmp_data);	

	free(h_tmp_row);
	free(h_tmp_data);
	return elapsedTime;
}
