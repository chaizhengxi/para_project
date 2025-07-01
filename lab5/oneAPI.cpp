#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

using namespace sycl;

static const int MATRIX_SIZE = 4000;
typedef float ElementType;
ElementType matrix[MATRIX_SIZE * MATRIX_SIZE];  // 一维数组存储矩阵

// 矩阵初始化函数
void init_matrix(ElementType* A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = 0;
        }
        A[i * n + i] = 1.0;
        for (int j = i + 1; j < n; ++j)
            A[i * n + j] = rand() % 100;
    }

    for (int i = 0; i < n; ++i) {
        int k1 = rand() % n;
        int k2 = rand() % n;
        for (int j = 0; j < n; ++j) {
            A[i * n + j] += A[0 * n + j];
            A[k1 * n + j] += A[k2 * n + j];
        }
    }
}

// GPU版本高斯消元
void gauss_GPU(ElementType* matrix, int n) {
    sycl::queue q{sycl::gpu_selector{}};
    std::cout << "设备: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // 分配USM共享内存
    ElementType* new_matrix = sycl::malloc_shared<ElementType>(n * n, q);
    std::memcpy(new_matrix, matrix, sizeof(ElementType) * n * n);

    for (int i = 0; i < n; ++i) {
        q.parallel_for(sycl::range<1>(n - i - 1), [=](sycl::id<1> row_idx) {
            int j = row_idx[0] + i + 1;
            ElementType div_factor = new_matrix[j * n + i] / new_matrix[i * n + i];
            for (int k = i; k < n; ++k) {
                new_matrix[j * n + k] -= new_matrix[i * n + k] * div_factor;
            }
        }).wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;

    std::cout << "GPU 执行时间: " << elapsed_time.count() << " 秒" << std::endl;
    sycl::free(new_matrix, q);
}

// CPU版本高斯消元
void gauss_CPU(ElementType* matrix, int n) {
    sycl::queue q{sycl::cpu_selector{}};
    std::cout << "设备: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // 分配USM共享内存
    ElementType* new_matrix = sycl::malloc_shared<ElementType>(n * n, q);
    std::memcpy(new_matrix, matrix, sizeof(ElementType) * n * n);

    for (int i = 0; i < n; ++i) {
        q.parallel_for(sycl::range<1>(n - i - 1), [=](sycl::id<1> row_idx) {
            int j = row_idx[0] + i + 1;
            ElementType div_factor = new_matrix[j * n + i] / new_matrix[i * n + i];
            for (int k = i; k < n; ++k) {
                new_matrix[j * n + k] -= new_matrix[i * n + k] * div_factor;
            }
        }).wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;

    std::cout << "CPU 执行时间: " << elapsed_time.count() << " 秒" << std::endl;
    sycl::free(new_matrix, q);
}

int main() {
    init_matrix(matrix, MATRIX_SIZE);
    gauss_CPU(matrix, MATRIX_SIZE);
    gauss_GPU(matrix, MATRIX_SIZE);
    return 0;
}