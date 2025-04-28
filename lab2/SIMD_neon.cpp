#include <arm_neon.h>
#include <sys/time.h>
#include <iostream>

// 定义矩阵大小
const int MATRIX_SIZE = 1000;

// 定义矩阵 A 和辅助矩阵 B
float matrixA[MATRIX_SIZE][MATRIX_SIZE];
float matrixB[MATRIX_SIZE][MATRIX_SIZE];

// 初始化矩阵 A
void initializeMatrix() {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            matrixA[i][j] = 0;
        }
        matrixA[i][i] = 1.0;
        for (int j = i + 1; j < MATRIX_SIZE; ++j) {
            matrixA[i][j] = static_cast<float>(rand());
        }
    }

    for (int k = 0; k < MATRIX_SIZE; ++k) {
        for (int i = k + 1; i < MATRIX_SIZE; ++i) {
            for (int j = 0; j < MATRIX_SIZE; ++j) {
                matrixA[i][j] += matrixA[k][j];
            }
        }
    }
}

// 普通算法
void ordinaryAlgorithm() {
    for (int k = 0; k < MATRIX_SIZE; ++k) {
        for (int j = k + 1; j < MATRIX_SIZE; ++j) {
            matrixA[k][j] /= matrixA[k][k];
        }
        matrixA[k][k] = 1.0;

        for (int i = k + 1; i < MATRIX_SIZE; ++i) {
            for (int j = k + 1; j < MATRIX_SIZE; ++j) {
                matrixA[i][j] -= matrixA[i][k] * matrixA[k][j];
            }
            matrixA[i][k] = 0;
        }
    }
}

// 普通算法（缓存优化）
void ordinaryAlgorithmWithCache() {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < i; ++j) {
            matrixB[j][i] = matrixA[i][j];
            matrixA[i][j] = 0;
        }
    }

    for (int k = 0; k < MATRIX_SIZE; ++k) {
        for (int j = k + 1; j < MATRIX_SIZE; ++j) {
            matrixA[k][j] /= matrixA[k][k];
        }
        matrixA[k][k] = 1.0;

        for (int i = k + 1; i < MATRIX_SIZE; ++i) {
            for (int j = k + 1; j < MATRIX_SIZE; ++j) {
                matrixA[i][j] -= matrixB[k][i] * matrixA[k][j];
            }
        }
    }
}

// 优化算法（使用 NEON 指令）
void optimizedAlgorithm() {
    for (int k = 0; k < MATRIX_SIZE; ++k) {
        float32x4_t divisor = vmovq_n_f32(matrixA[k][k]);

        for (int j = k + 1; j + 3 < MATRIX_SIZE; j += 4) {
            float32x4_t vec = vld1q_f32(&matrixA[k][j]);
            vec = vdivq_f32(vec, divisor);
            vst1q_f32(&matrixA[k][j], vec);
        }

        for (int j = (MATRIX_SIZE & ~3); j < MATRIX_SIZE; ++j) {
            matrixA[k][j] /= matrixA[k][k];
        }
        matrixA[k][k] = 1.0;

        for (int i = k + 1; i < MATRIX_SIZE; ++i) {
            float32x4_t factor = vmovq_n_f32(matrixA[i][k]);

            for (int j = k + 1; j + 3 < MATRIX_SIZE; j += 4) {
                float32x4_t vecKj = vld1q_f32(&matrixA[k][j]);
                float32x4_t vecIj = vld1q_f32(&matrixA[i][j]);
                float32x4_t product = vmulq_f32(vecKj, factor);
                vecIj = vsubq_f32(vecIj, product);
                vst1q_f32(&matrixA[i][j], vecIj);
            }

            for (int j = (MATRIX_SIZE & ~3); j < MATRIX_SIZE; ++j) {
                matrixA[i][j] -= matrixA[i][k] * matrixA[k][j];
            }
            matrixA[i][k] = 0;
        }
    }
}

// 优化算法（缓存和 NEON 指令）
void optimizedAlgorithmWithCache() {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < i; ++j) {
            matrixB[j][i] = matrixA[i][j];
            matrixA[i][j] = 0;
        }
    }

    for (int k = 0; k < MATRIX_SIZE; ++k) {
        float32x4_t divisor = vmovq_n_f32(matrixA[k][k]);

        for (int j = k + 1; j + 3 < MATRIX_SIZE; j += 4) {
            float32x4_t vec = vld1q_f32(&matrixA[k][j]);
            vec = vdivq_f32(vec, divisor);
            vst1q_f32(&matrixA[k][j], vec);
        }

        for (int j = (MATRIX_SIZE & ~3); j < MATRIX_SIZE; ++j) {
            matrixA[k][j] /= matrixA[k][k];
        }
        matrixA[k][k] = 1.0;

        for (int i = k + 1; i < MATRIX_SIZE; ++i) {
            float32x4_t factor = vmovq_n_f32(matrixB[k][i]);

            for (int j = k + 1; j + 3 < MATRIX_SIZE; j += 4) {
                float32x4_t vecKj = vld1q_f32(&matrixA[k][j]);
                float32x4_t vecIj = vld1q_f32(&matrixA[i][j]);
                float32x4_t product = vmulq_f32(vecKj, factor);
                vecIj = vsubq_f32(vecIj, product);
                vst1q_f32(&matrixA[i][j], vecIj);
            }

            for (int j = (MATRIX_SIZE & ~3); j < MATRIX_SIZE; ++j) {
                matrixA[i][j] -= matrixA[i][k] * matrixA[k][j];
            }
        }
    }
}

// 优化算法（内存对齐和 NEON 指令）
void optimizedAlgorithmWithAlignment() {
    for (int k = 0; k < MATRIX_SIZE; ++k) {
        float32x4_t divisor = vmovq_n_f32(matrixA[k][k]);
        int j = k + 1;

        // 对齐内存访问
        while ((k * MATRIX_SIZE + j) % 4 != 0) {
            matrixA[k][j] /= matrixA[k][k];
            ++j;
        }

        // 循环展开和 SIMD 优化
        for (; j + 3 < MATRIX_SIZE; j += 4) {
            float32x4_t vec = vld1q_f32(&matrixA[k][j]);
            vec = vdivq_f32(vec, divisor);
            vst1q_f32(&matrixA[k][j], vec);
        }

        // 处理剩余部分
        for (; j < MATRIX_SIZE; ++j) {
            matrixA[k][j] /= matrixA[k][k];
        }
        matrixA[k][k] = 1.0;

        for (int i = k + 1; i < MATRIX_SIZE; ++i) {
            float32x4_t factor = vmovq_n_f32(matrixA[i][k]);
            int j = k + 1;

            // 对齐内存访问
            while ((i * MATRIX_SIZE + j) % 4 != 0) {
                matrixA[i][j] -= matrixA[k][j] * matrixA[i][k];
                ++j;
            }

            // 循环展开和 SIMD 优化
            for (; j + 3 < MATRIX_SIZE; j += 4) {
                float32x4_t vecKj = vld1q_f32(&matrixA[k][j]);
                float32x4_t vecIj = vld1q_f32(&matrixA[i][j]);
                float32x4_t product = vmulq_f32(vecKj, factor);
                vecIj = vsubq_f32(vecIj, product);
                vst1q_f32(&matrixA[i][j], vecIj);
            }

            // 处理剩余部分
            for (; j < MATRIX_SIZE; ++j) {
                matrixA[i][j] -= matrixA[k][j] * matrixA[i][k];
            }
            matrixA[i][k] = 0;
        }
    }
}

// 输出矩阵结果
void printMatrix() {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            std::cout << matrixA[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// 测量函数执行时间
double measureTime(void (*func)()) {
    struct timeval start, end;
    gettimeofday(&start, nullptr);
    func();
    gettimeofday(&end, nullptr);
    return ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) / 1000.0;
}

int main() {
    // 普通算法
    initializeMatrix();
    double ordinaryTime = measureTime(ordinaryAlgorithm);
    std::cout << "Ordinary algorithm time: " << ordinaryTime << " ms" << std::endl;

    // 普通算法（缓存优化）
    initializeMatrix();
    double ordinaryCacheTime = measureTime(ordinaryAlgorithmWithCache);
    std::cout << "Ordinary algorithm with cache time: " << ordinaryCacheTime << " ms" << std::endl;

    // 优化算法（使用 NEON 指令）
    initializeMatrix();
    double optimizedTime = measureTime(optimizedAlgorithm);
    std::cout << "Optimized algorithm time: " << optimizedTime << " ms" << std::endl;

    // 优化算法（缓存和 NEON 指令）
    initializeMatrix();
    double optimizedCacheTime = measureTime(optimizedAlgorithmWithCache);
    std::cout << "Optimized algorithm with cache time: " << optimizedCacheTime << " ms" << std::endl;

    // 优化算法（内存对齐和 NEON 指令）
    initializeMatrix();
    double optimizedAlignmentTime = measureTime(optimizedAlgorithmWithAlignment);
    std::cout << "Optimized algorithm with alignment time: " << optimizedAlignmentTime << " ms" << std::endl;

    return 0;
}    