#pragma once
#include <cuda_runtime.h>

template<typename T>
__global__ void spmv_csr_kernel(int n,
                                const int* __restrict__ rowptr,
                                const int* __restrict__ colind,
                                const T*  __restrict__ val,
                                const T*  __restrict__ x,
                                      T*  __restrict__ y)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x; // 0..31
    __shared__ T sdata[32*8];           // 8 = blockDim.y
    if (row < n) {
        T sum = 0;
        int row_start = rowptr[row];
        int row_end   = rowptr[row+1];
        for (int p = row_start + lane; p < row_end; p += 32)
            sum += val[p] * x[colind[p]];
        sdata[threadIdx.y*32 + lane] = sum;
        __syncthreads();
        // warp reduce
        if (lane < 16) sdata[threadIdx.y*32 + lane] += sdata[threadIdx.y*32 + lane+16];
        if (lane < 8 ) sdata[threadIdx.y*32 + lane] += sdata[threadIdx.y*32 + lane+8 ];
        if (lane < 4 ) sdata[threadIdx.y*32 + lane] += sdata[threadIdx.y*32 + lane+4 ];
        if (lane < 2 ) sdata[threadIdx.y*32 + lane] += sdata[threadIdx.y*32 + lane+2 ];
        if (lane == 0)  y[row] = sdata[threadIdx.y*32];
    }
}