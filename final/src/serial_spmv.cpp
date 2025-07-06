#include "csr.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? std::stoul(argv[1]) : 1 << 14;
    CSR A = make_random_csr(n);

    std::vector<float> x(n, 1.f), y(n, 0.f);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i) {
        float sum = 0.f;
        for (int p = A.rowptr[i]; p < A.rowptr[i + 1]; ++p)
            sum += A.val[p] * x[A.colind[p]];
        y[i] = sum;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    size_t nnz = A.val.size();
    double gflops = 2.0 * nnz / ms / 1e6;
    std::cout << "serial_spmv  n=" << n << "  GFLOPs=" << gflops << "\n";
    return 0;
}