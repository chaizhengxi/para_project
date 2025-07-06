#include "csr.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#ifdef USE_OPENMP
#endif

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? std::stoul(argv[1]) : 1 << 14;
    CSR A = make_random_csr(n);
    std::vector<float> x(n, 1.f), y(n, 0.f);

    auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic,64)
    for (size_t i = 0; i < n; ++i) {
        float sum = 0.f;
        for (int p = A.rowptr[i]; p < A.rowptr[i + 1]; ++p)
            sum += A.val[p] * x[A.colind[p]];
        y[i] = sum;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    size_t nnz = A.val.size();
    std::cout << "omp_spmv  threads=" 
#ifdef USE_OPENMP
              << omp_get_max_threads()
#else
              << 1
#endif
              << "  GFLOPs=" << 2.0 * nnz / ms / 1e6 << "\n";
}