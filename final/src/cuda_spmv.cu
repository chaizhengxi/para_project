#include "csr.hpp"
#include "cuda_kernel.cuh"
#include <vector>
#include <iostream>

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? std::stoul(argv[1]) : 1<<14;
    CSR hA = make_random_csr(n);
    std::vector<float> hx(n, 1.f), hy(n, 0.f);

    int nnz = hA.val.size();
    // device memory
    int  *d_rowptr, *d_colind;
    float *d_val, *d_x, *d_y;
    cudaMalloc(&d_rowptr,(n+1)*sizeof(int));
    cudaMalloc(&d_colind,nnz*sizeof(int));
    cudaMalloc(&d_val,   nnz*sizeof(float));
    cudaMalloc(&d_x,     n*sizeof(float));
    cudaMalloc(&d_y,     n*sizeof(float));

    cudaMemcpy(d_rowptr,hA.rowptr.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_colind,hA.colind.data(),nnz*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_val,   hA.val.data()   ,nnz*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,     hx.data()       ,n*sizeof(float)  ,cudaMemcpyHostToDevice);

    dim3 block(32,8);                       // 32 lanes × 8 rows
    dim3 grid ((n+block.y-1)/block.y);

    cudaEvent_t t0,t1;  cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    spmv_csr_kernel<<<grid,block>>>(n,d_rowptr,d_colind,d_val,d_x,d_y);
    cudaEventRecord(t1);  cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms,t0,t1);

    double gflops = 2.0*nnz/ms/1e6;
    std::cout << "cuda_spmv  n="<<n<<"  GFLOPs="<<gflops<<"\n";

    cudaMemcpy(hy.data(),d_y,n*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_rowptr); cudaFree(d_colind); cudaFree(d_val);
    cudaFree(d_x); cudaFree(d_y);
}