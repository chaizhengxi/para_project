#include <omp.h>
#include <iostream>
#include <windows.h>
#include <immintrin.h> // AVX支持
using namespace std;

// 矩阵规模
const int n = 1000;
float arr[n][n];  // 原始矩阵（用于重置）
float A[n][n];     // 工作矩阵

// 线程数及测试参数
const int NUM_THREADS = 7; // 工作线程数
const int CYCLE = 5;       // 循环测试次数

// 初始化矩阵（生成非奇异矩阵）
void init() {
    srand(time(nullptr));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            arr[i][j] = (i == j) ? 1.0f : static_cast<float>(rand() % 100);
        }
    }
    // 确保矩阵非奇异（添加行操作）
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            arr[i][j] += arr[0][j]; // 简单非奇异处理
        }
    }
}

// 重置矩阵到初始状态
void ReStart() {
    for (int i = 0; i < n; i++) {
        memcpy(A[i], arr[i], n * sizeof(float));
    }
}

// 普通串行高斯消去
void f_ordinary() {
    for (int k = 0; k < n; k++) {
        float tmp = A[k][k];
        // 归一化当前行
        for (int j = k + 1; j < n; j++) {
            A[k][j] /= tmp;
        }
        A[k][k] = 1.0f;
        
        // 消去下方行
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}

// OpenMP静态调度（无向量化）
void f_omp_static() {
#pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
        // 串行归一化当前行
#pragma omp single
        {
            float tmp = A[k][k];
            for (int j = k + 1; j < n; j++) {
                A[k][j] /= tmp;
            }
            A[k][k] = 1.0f;
        }
        
        // 并行消去下方行（静态调度）
#pragma omp for schedule(static)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}

// OpenMP静态调度 + AVX向量化
void f_omp_static_avx() {
#pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
        // 串行归一化（AVX向量化）
#pragma omp single
        {
            float tmp = A[k][k];
            __m256 tmp_vec = _mm256_set1_ps(tmp); // 广播标量到向量
            int j = k + 1;
            
            // 处理对齐的8元素块
            for (; j <= n - 8; j += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[k][j]);
                a_vec = _mm256_div_ps(a_vec, tmp_vec);
                _mm256_storeu_ps(&A[k][j], a_vec);
            }
            
            // 处理剩余元素
            for (; j < n; j++) {
                A[k][j] /= tmp;
            }
            A[k][k] = 1.0f;
        }
        
        // 并行消去（AVX向量化 + 静态调度）
#pragma omp for schedule(static)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            __m256 factor_vec = _mm256_set1_ps(factor);
            int j = k + 1;
            
            // 处理对齐的8元素块
            for (; j <= n - 8; j += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i][j]);
                __m256 k_vec = _mm256_loadu_ps(&A[k][j]);
                k_vec = _mm256_mul_ps(k_vec, factor_vec);
                a_vec = _mm256_sub_ps(a_vec, k_vec);
                _mm256_storeu_ps(&A[i][j], a_vec);
            }
            
            // 处理剩余元素
            for (; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}

// OpenMP静态调度 + barrier同步
void f_omp_static_avx_barrier() {
#pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
        // 主线程执行归一化
#pragma omp master
        {
            float tmp = A[k][k];
            __m256 tmp_vec = _mm256_set1_ps(tmp);
            int j = k + 1;
            
            for (; j <= n - 8; j += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[k][j]);
                a_vec = _mm256_div_ps(a_vec, tmp_vec);
                _mm256_storeu_ps(&A[k][j], a_vec);
            }
            for (; j < n; j++) A[k][j] /= tmp;
            A[k][k] = 1.0f;
        }
        
        // 所有线程等待归一化完成
#pragma omp barrier
        
        // 并行消去（静态调度）
#pragma omp for schedule(static)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            __m256 factor_vec = _mm256_set1_ps(factor);
            int j = k + 1;
            
            for (; j <= n - 8; j += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i][j]);
                __m256 k_vec = _mm256_loadu_ps(&A[k][j]);
                k_vec = _mm256_mul_ps(k_vec, factor_vec);
                a_vec = _mm256_sub_ps(a_vec, k_vec);
                _mm256_storeu_ps(&A[i][j], a_vec);
            }
            for (; j < n; j++) A[i][j] -= factor * A[k][j];
            A[i][k] = 0.0f;
        }
    }
}

// 除法阶段并行化（实验性，可能存在竞争）
void f_omp_static_avx_division() {
#pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
        float tmp = A[k][k];
        __m256 tmp_vec = _mm256_set1_ps(tmp);
        
        // 尝试并行化归一化（风险：行内元素竞争）
#pragma omp for schedule(static)
        for (int j = k + 1; j < n; j++) {
            if (j % 8 == 0) { // 对齐处理
                int jj = j;
                __m256 a_vec = _mm256_loadu_ps(&A[k][jj]);
                a_vec = _mm256_div_ps(a_vec, tmp_vec);
                _mm256_storeu_ps(&A[k][jj], a_vec);
            } else { // 标量处理
                A[k][j] /= tmp;
            }
        }
        A[k][k] = 1.0f;
        
        // 并行消去（静态调度）
#pragma omp for schedule(static)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            __m256 factor_vec = _mm256_set1_ps(factor);
            int j = k + 1;
            
            for (; j <= n - 8; j += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i][j]);
                __m256 k_vec = _mm256_loadu_ps(&A[k][j]);
                k_vec = _mm256_mul_ps(k_vec, factor_vec);
                a_vec = _mm256_sub_ps(a_vec, k_vec);
                _mm256_storeu_ps(&A[i][j], a_vec);
            }
            for (; j < n; j++) A[i][j] -= factor * A[k][j];
            A[i][k] = 0.0f;
        }
    }
}

// OpenMP自动向量化
void f_omp_static_simd() {
#pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
        // 串行归一化（自动向量化）
#pragma omp single
        {
            float tmp = A[k][k];
#pragma omp simd
            for (int j = k + 1; j < n; j++) {
                A[k][j] /= tmp;
            }
            A[k][k] = 1.0f;
        }
        
        // 并行消去（自动向量化）
#pragma omp for schedule(static)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
#pragma omp simd
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}

// 动态调度版本
void f_omp_dynamic() {
#pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
#pragma omp single
        {
            float tmp = A[k][k];
            for (int j = k + 1; j < n; j++) {
                A[k][j] /= tmp;
            }
            A[k][k] = 1.0f;
        }
        
        // 动态调度消去
#pragma omp for schedule(dynamic, 80)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}

// 动态调度 + AVX
void f_omp_dynamic_avx() {
#pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
#pragma omp single
        {
            float tmp = A[k][k];
            __m256 tmp_vec = _mm256_set1_ps(tmp);
            int j = k + 1;
            
            for (; j <= n - 8; j += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[k][j]);
                a_vec = _mm256_div_ps(a_vec, tmp_vec);
                _mm256_storeu_ps(&A[k][j], a_vec);
            }
            for (; j < n; j++) A[k][j] /= tmp;
            A[k][k] = 1.0f;
        }
        
        // 动态调度消去（AVX）
#pragma omp for schedule(dynamic, 200)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            __m256 factor_vec = _mm256_set1_ps(factor);
            int j = k + 1;
            
            for (; j <= n - 8; j += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i][j]);
                __m256 k_vec = _mm256_loadu_ps(&A[k][j]);
                k_vec = _mm256_mul_ps(k_vec, factor_vec);
                a_vec = _mm256_sub_ps(a_vec, k_vec);
                _mm256_storeu_ps(&A[i][j], a_vec);
            }
            for (; j < n; j++) A[i][j] -= factor * A[k][j];
            A[i][k] = 0.0f;
        }
    }
}

// 指导式调度版本
void f_omp_guided() {
#pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
#pragma omp single
        {
            float tmp = A[k][k];
            for (int j = k + 1; j < n; j++) {
                A[k][j] /= tmp;
            }
            A[k][k] = 1.0f;
        }
        
        // 指导式调度消去
#pragma omp for schedule(guided, 80)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}

// 指导式调度 + AVX
void f_omp_guided_avx() {
#pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
#pragma omp single
        {
            float tmp = A[k][k];
            __m256 tmp_vec = _mm256_set1_ps(tmp);
            int j = k + 1;
            
            for (; j <= n - 8; j += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[k][j]);
                a_vec = _mm256_div_ps(a_vec, tmp_vec);
                _mm256_storeu_ps(&A[k][j], a_vec);
            }
            for (; j < n; j++) A[k][j] /= tmp;
            A[k][k] = 1.0f;
        }
        
        // 指导式调度消去（AVX）
#pragma omp for schedule(guided, 80)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            __m256 factor_vec = _mm256_set1_ps(factor);
            int j = k + 1;
            
            for (; j <= n - 8; j += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i][j]);
                __m256 k_vec = _mm256_loadu_ps(&A[k][j]);
                k_vec = _mm256_mul_ps(k_vec, factor_vec);
                a_vec = _mm256_sub_ps(a_vec, k_vec);
                _mm256_storeu_ps(&A[i][j], a_vec);
            }
            for (; j < n; j++) A[i][j] -= factor * A[k][j];
            A[i][k] = 0.0f;
        }
    }
}

// 性能测试函数
void test_performance(const char* name, void (*func)()) {
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    double total = 0.0;

    for (int i = 0; i < CYCLE; i++) {
        ReStart();
        QueryPerformanceCounter(&start);
        func();
        QueryPerformanceCounter(&end);
        total += (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
    }
    cout << name << ": " << total / CYCLE << " ms" << endl;
}

int main() {
    init(); // 初始化矩阵
    
    // 执行各版本测试
    test_performance("f_ordinary", f_ordinary);
    test_performance("f_omp_static", f_omp_static);
    test_performance("f_omp_dynamic", f_omp_dynamic);
    test_performance("f_omp_guided", f_omp_guided);
    test_performance("f_omp_static_avx", f_omp_static_avx);
    test_performance("f_omp_dynamic_avx", f_omp_dynamic_avx);
    test_performance("f_omp_guided_avx", f_omp_guided_avx);
    test_performance("f_omp_static_avx_barrier", f_omp_static_avx_barrier);
    test_performance("f_omp_static_avx_division", f_omp_static_avx_division);
    test_performance("f_omp_static_simd", f_omp_static_simd);

    return 0;
}