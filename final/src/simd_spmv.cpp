#include "csr.hpp"
#include <iostream>
#include <vector>
#include <chrono>

static inline bool has_avx512() {
#if defined(__AVX512F__)
    return true;
#else
    return false;
#endif
}

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? std::stoul(argv[1]) : 1 << 14;
    CSR A = make_random_csr(n);
    std::vector<float> x(n, 1.f), y(n, 0.f);

    auto t0 = std::chrono::high_resolution_clock::now();
    if (has_avx512()) {
        for (size_t i = 0; i < n; ++i) {
            __m512 sumv = _mm512_setzero_ps();
            int p = A.rowptr[i], e = A.rowptr[i + 1];
            for (; p + 15 < e; p += 16) {
                __m512 val = _mm512_loadu_ps(&A.val[p]);
                __m512 idx = _mm512_loadu_ps(reinterpret_cast<const float*>(&A.colind[p]));
                __m512 vec = _mm512_i32gather_ps(_mm512_cvtsepi32_ps(idx), x.data(), 4);
                sumv = _mm512_fmadd_ps(val, vec, sumv);
            }
            float sum = _mm512_reduce_add_ps(sumv);
            for (; p < e; ++p) sum += A.val[p] * x[A.colind[p]];
            y[i] = sum;
        }
    } else {  // fallback
        for (size_t i = 0; i < n; ++i) {
            float sum = 0.f;
            for (int p = A.rowptr[i]; p < A.rowptr[i + 1]; ++p)
                sum += A.val[p] * x[A.colind[p]];
            y[i] = sum;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    size_t nnz = A.val.size();
    std::cout << "simd_spmv GFLOPs=" << 2.0 * nnz / ms / 1e6 << "\n";
}