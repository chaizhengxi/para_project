#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h> // AVX、AVX2
#include <iostream>
#include <chrono>

// 定义矩阵大小
const int MATRIX_SIZE = 3000;

// 定义矩阵
float matrix[MATRIX_SIZE][MATRIX_SIZE];

// 初始化矩阵
void initializeMatrix() {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            matrix[i][j] = 0;
        }
        matrix[i][i] = 1.0;
        for (int j = i + 1; j < MATRIX_SIZE; ++j) {
            matrix[i][j] = static_cast<float>(rand());
        }
    }

    for (int k = 0; k < MATRIX_SIZE; ++k) {
        for (int i = k + 1; i < MATRIX_SIZE; ++i) {
            for (int j = 0; j < MATRIX_SIZE; ++j) {
                matrix[i][j] += matrix[k][j];
            }
        }
    }
}

// 普通高斯消元算法
void ordinaryGaussianElimination() {
    for (int k = 0; k < MATRIX_SIZE; ++k) {
        for (int j = k + 1; j < MATRIX_SIZE; ++j) {
            matrix[k][j] /= matrix[k][k];
        }
        matrix[k][k] = 1.0;

        for (int i = k + 1; i < MATRIX_SIZE; ++i) {
            for (int j = k + 1; j < MATRIX_SIZE; ++j) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// SSE 优化的高斯消元算法
void sseOptimizedGaussianElimination() {
    for (int k = 0; k < MATRIX_SIZE; ++k) {
        __m128 diagonalElement = _mm_set1_ps(matrix[k][k]);

        int j;
        for (j = k + 1; j + 4 <= MATRIX_SIZE; j += 4) {
            __m128 currentElements = _mm_loadu_ps(&matrix[k][j]);
            currentElements = _mm_div_ps(currentElements, diagonalElement);
            _mm_storeu_ps(&matrix[k][j], currentElements);
        }

        for (; j < MATRIX_SIZE; ++j) {
            matrix[k][j] /= matrix[k][k];
        }

        matrix[k][k] = 1.0;

        for (int i = k + 1; i < MATRIX_SIZE; ++i) {
            __m128 multiplier = _mm_set1_ps(matrix[i][k]);

            for (j = k + 1; j + 4 <= MATRIX_SIZE; j += 4) {
                __m128 elementsKj = _mm_loadu_ps(&matrix[k][j]);
                __m128 elementsIj = _mm_loadu_ps(&matrix[i][j]);
                __m128 product = _mm_mul_ps(elementsKj, multiplier);
                elementsIj = _mm_sub_ps(elementsIj, product);
                _mm_storeu_ps(&matrix[i][j], elementsIj);
            }

            for (; j < MATRIX_SIZE; ++j) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }

            matrix[i][k] = 0;
        }
    }
}

// AVX256 优化的高斯消元算法
void avx256OptimizedGaussianElimination() {
    for (int k = 0; k < MATRIX_SIZE; ++k) {
        float inverseDiagonal = 1.0f / matrix[k][k];
        matrix[k][k] = 1.0f;

        for (int j = k + 1; j < MATRIX_SIZE; ++j) {
            matrix[k][j] *= inverseDiagonal;
        }

        for (int i = k + 1; i < MATRIX_SIZE; ++i) {
            float multiplierValue = matrix[i][k];
            __m256 multiplier = _mm256_set1_ps(multiplierValue);

            int j;
            for (j = k + 1; j + 8 <= MATRIX_SIZE; j += 8) {
                __m256 elementsKj = _mm256_loadu_ps(&matrix[k][j]);
                __m256 elementsIj = _mm256_loadu_ps(&matrix[i][j]);
                __m256 product = _mm256_mul_ps(elementsKj, multiplier);
                __m256 result = _mm256_sub_ps(elementsIj, product);
                _mm256_storeu_ps(&matrix[i][j], result);
            }

            for (; j < MATRIX_SIZE; ++j) {
                matrix[i][j] -= multiplierValue * matrix[k][j];
            }

            matrix[i][k] = 0.0f;
        }
    }
}

// 测量函数执行时间
template<typename Func>
long long measureTime(Func func) {
    auto startTime = std::chrono::high_resolution_clock::now();
    func();
    auto endTime = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}

int main() {
    // 初始化矩阵
    initializeMatrix();

    // 普通算法执行时间
    long long ordinaryTime = measureTime(ordinaryGaussianElimination);
    std::cout << "Ordinary Gaussian Elimination time: " << ordinaryTime << " microseconds" << std::endl;

    // 重新初始化矩阵
    initializeMatrix();

    // SSE 优化算法执行时间
    long long sseTime = measureTime(sseOptimizedGaussianElimination);
    std::cout << "SSE Optimized Gaussian Elimination time: " << sseTime << " microseconds" << std::endl;

    // 重新初始化矩阵
    initializeMatrix();

    // AVX256 优化算法执行时间
    long long avx256Time = measureTime(avx256OptimizedGaussianElimination);
    std::cout << "AVX256 Optimized Gaussian Elimination time: " << avx256Time << " microseconds" << std::endl;

    return 0;
}    