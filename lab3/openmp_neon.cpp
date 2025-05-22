#include <omp.h>
#include <iostream>
#include <sys/time.h>
#include <arm_neon.h> // NEON向量化支持
using namespace std;

// 矩阵规模和线程数
const int n = 500;
float arr[n][n];
float A[n][n];
const int NUM_THREADS = 7;

// 初始化矩阵（生成非奇异矩阵）
void init() {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            arr[i][j] = 0;
        }
        arr[i][i] = 1.0f;
        for (int j = i + 1; j < n; j++) {
            arr[i][j] = static_cast<float>(rand() % 100);
        }
    }
    
    // 确保矩阵非奇异
    for (int i = 0; i < n; i++) {
        int k1 = rand() % n;
        int k2 = rand() % n;
        for (int j = 0; j < n; j++) {
            arr[i][j] += arr[0][j];
            arr[k1][j] += arr[k2][j];
        }
    }
}

// 重置矩阵到初始状态
void ReStart() {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = arr[i][j];
        }
    }
}

// 普通串行版本
void f_ordinary() {
    for (int k = 0; k < n; k++) {
        float pivot = A[k][k];
        // 归一化当前行
        for (int j = k + 1; j < n; j++) {
            A[k][j] /= pivot;
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

// OpenMP静态调度 + NEON向量化（修复版本）
void f_omp_static_neon() {
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
        // 串行部分：归一化当前行
        #pragma omp single
        {
            float pivot = A[k][k];
            float32x4_t v_pivot = vmovq_n_f32(pivot);
            
            int j;
            // 向量化处理
            for (j = k + 1; j <= n - 4; j += 4) {
                float32x4_t v_row = vld1q_f32(&A[k][j]);
                v_row = vdivq_f32(v_row, v_pivot);
                vst1q_f32(&A[k][j], v_row);
            }
            // 处理剩余元素
            for (; j < n; j++) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0f;
        }
        
        // 并行部分：消去下方行
        #pragma omp for schedule(static)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            float32x4_t v_factor = vmovq_n_f32(factor);
            
            int j;
            // 向量化处理
            for (j = k + 1; j <= n - 4; j += 4) {
                float32x4_t v_elim = vld1q_f32(&A[k][j]);
                float32x4_t v_target = vld1q_f32(&A[i][j]);
                float32x4_t v_result = vsubq_f32(v_target, vmulq_f32(v_elim, v_factor));
                vst1q_f32(&A[i][j], v_result);
            }
            // 处理剩余元素
            for (; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}

// OpenMP动态调度 + NEON向量化
void f_omp_dynamic_neon() {
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
        #pragma omp single
        {
            float pivot = A[k][k];
            float32x4_t v_pivot = vmovq_n_f32(pivot);
            
            int j;
            for (j = k + 1; j <= n - 4; j += 4) {
                float32x4_t v_row = vld1q_f32(&A[k][j]);
                v_row = vdivq_f32(v_row, v_pivot);
                vst1q_f32(&A[k][j], v_row);
            }
            for (; j < n; j++) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0f;
        }
        
        #pragma omp for schedule(dynamic, 14)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            float32x4_t v_factor = vmovq_n_f32(factor);
            
            int j;
            for (j = k + 1; j <= n - 4; j += 4) {
                float32x4_t v_elim = vld1q_f32(&A[k][j]);
                float32x4_t v_target = vld1q_f32(&A[i][j]);
                float32x4_t v_result = vsubq_f32(v_target, vmulq_f32(v_elim, v_factor));
                vst1q_f32(&A[i][j], v_result);
            }
            for (; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}

// OpenMP指导式调度 + NEON向量化
void f_omp_guide_neon() {
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
        #pragma omp single
        {
            float pivot = A[k][k];
            float32x4_t v_pivot = vmovq_n_f32(pivot);
            
            int j;
            for (j = k + 1; j <= n - 4; j += 4) {
                float32x4_t v_row = vld1q_f32(&A[k][j]);
                v_row = vdivq_f32(v_row, v_pivot);
                vst1q_f32(&A[k][j], v_row);
            }
            for (; j < n; j++) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0f;
        }
        
        #pragma omp for schedule(guided, 1)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            float32x4_t v_factor = vmovq_n_f32(factor);
            
            int j;
            for (j = k + 1; j <= n - 4; j += 4) {
                float32x4_t v_elim = vld1q_f32(&A[k][j]);
                float32x4_t v_target = vld1q_f32(&A[i][j]);
                float32x4_t v_result = vsubq_f32(v_target, vmulq_f32(v_elim, v_factor));
                vst1q_f32(&A[i][j], v_result);
            }
            for (; j < n; j++) {
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
        #pragma omp single
        {
            float pivot = A[k][k];
            for (int j = k + 1; j < n; j++) {
                A[k][j] /= pivot;
            }
            A[k][k] = 1.0f;
        }
        
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

// 测试函数性能并返回执行时间（毫秒）
double test_function(void (*func)()) {
    ReStart();
    timeval start, end;
    gettimeofday(&start, nullptr);
    func();
    gettimeofday(&end, nullptr);
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
}

int main() {
    srand(time(nullptr));
    init();
    
    // 测试各版本性能
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Number of threads: " << NUM_THREADS << endl;
    cout << "----------------------------------------" << endl;
    
    cout << "f_ordinary: " << test_function(f_ordinary) << " ms" << endl;
    cout << "f_omp_static: " << test_function(f_omp_static) << " ms" << endl;
    cout << "f_omp_static_neon: " << test_function(f_omp_static_neon) << " ms" << endl;
    cout << "f_omp_dynamic_neon: " << test_function(f_omp_dynamic_neon) << " ms" << endl;
    cout << "f_omp_guide_neon: " << test_function(f_omp_guide_neon) << " ms" << endl;
    
    return 0;
}