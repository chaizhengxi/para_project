#pragma once
#include <vector>
#include <random>
#include <cstddef>
#include <cstdio>

struct CSR {
    size_t n;                  // 方阵维度
    std::vector<int>   rowptr; // n+1
    std::vector<int>   colind; // nnz
    std::vector<float> val;    // nnz
};

// 生成一份随机稀疏矩阵（便于快速测试）
inline CSR make_random_csr(size_t n, int avg_nnz_per_row = 20) {
    CSR A;
    A.n = n;
    A.rowptr.resize(n + 1, 0);
    std::default_random_engine eng(123);
    std::uniform_int_distribution<int> col_dist(0, (int)n - 1);
    std::uniform_real_distribution<float> val_dist(-1.f, 1.f);

    for (size_t i = 0; i < n; ++i) {
        int row_nnz = avg_nnz_per_row;
        A.rowptr[i + 1] = A.rowptr[i] + row_nnz;
        for (int k = 0; k < row_nnz; ++k) {
            A.colind.push_back(col_dist(eng));
            A.val.push_back(val_dist(eng));
        }
    }
    return A;
}