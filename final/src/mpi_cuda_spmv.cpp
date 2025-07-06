#include "csr.hpp"
#include "cuda_kernel.cuh"
#include <vector>
#include <iostream>

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);

    size_t n_global = (argc>1)?std::stoul(argv[1]):1<<14;
    size_t n_local  = (n_global+size-1)/size;
    size_t begin = rank*n_local, end = std::min(n_global, begin+n_local);
    size_t n = end-begin;

    CSR hA = make_random_csr(n);
    std::vector<float> hx(n,1.f), hy(n,0.f);

    cudaSetDevice(rank%cudaGetDeviceCount());

    // device alloc & copy (与 cuda_spmv.cu 相同，略)
    int nnz = hA.val.size();
    int *d_rowptr,*d_colind; float*d_val,*d_x,*d_y;
    cudaMalloc(&d_rowptr,(n+1)*sizeof(int));
    cudaMalloc(&d_colind,nnz*sizeof(int));
    cudaMalloc(&d_val,nnz*sizeof(float));
    cudaMalloc(&d_x,n*sizeof(float));
    cudaMalloc(&d_y,n*sizeof(float));
    cudaMemcpy(d_rowptr,hA.rowptr.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_colind,hA.colind.data(),nnz*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_val,hA.val.data(),nnz*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,hx.data(),n*sizeof(float),cudaMemcpyHostToDevice);

    dim3 block(32,8); dim3 grid((n+block.y-1)/block.y);
    spmv_csr_kernel<<<grid,block>>>(n,d_rowptr,d_colind,d_val,d_x,d_y);

    double local_sum = 0, global_sum = 0;
    cudaMemcpy(hy.data(),d_y,n*sizeof(float),cudaMemcpyDeviceToHost);
    for(float v:hy) local_sum += v;
    MPI_Reduce(&local_sum,&global_sum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    if(rank==0) std::cout<<"mpi_cuda_spmv OK, result checksum="<<global_sum<<"\n";

    MPI_Finalize();
    return 0;
}