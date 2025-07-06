/*****************************************************************
 *  Dense Gaussian Elimination – ARM-Neon single-core benchmark  *
 *  串行基线 / Neon SIMD / Neon SIMD + Cache-block (B=64)        *
 *****************************************************************/
#include <arm_neon.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

#ifndef N          // 可在命令行 -DN=3000 指定
#define N 1000     // 默认阶数，测试时请调成 50–3000 多组验证
#endif

using ms        = std::chrono::duration<double, std::milli>;
using steady_ck = std::chrono::steady_clock;

static float A[N][N] __attribute__((aligned(16)));   // 主矩阵
static float B[N][N] __attribute__((aligned(16)));   // 备份(重复测试用)

/*********************  数据初始化  ******************************/
static void init_matrix(int n)
{
    std::srand(2025);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) A[i][j] = 0.f;
        A[i][i] = 1.f;
        for (int j = i + 1; j < n; ++j) A[i][j] = std::rand() % 1000 + 1;
    }
    /* 使矩阵“非退化”且保持上三角结构 */
    for (int k = 0; k < n; ++k)
        for (int i = k + 1; i < n; ++i)
            for (int j = 0; j < n; ++j) A[i][j] += A[k][j];
    std::memcpy(B, A, sizeof(A));
}

static inline void restore(int n) { std::memcpy(A, B, sizeof(float) * n * n); }

/*********************  串行基线版本  ****************************/
static void gauss_serial(int n)
{
    for (int k = 0; k < n; ++k) {
        float pivot = A[k][k];
        for (int j = k + 1; j < n; ++j) A[k][j] /= pivot;
        A[k][k] = 1.f;

        for (int i = k + 1; i < n; ++i) {
            float fac = A[i][k];
            for (int j = k + 1; j < n; ++j)
                A[i][j] -= fac * A[k][j];
            A[i][k] = 0.f;
        }
    }
}

/*********************  Neon SIMD 版本  **************************/
static void gauss_neon(int n)
{
    for (int k = 0; k < n; ++k) {
        float32x4_t v_pivot = vdupq_n_f32(A[k][k]);

        int j = k + 1;
        for (; j + 3 < n; j += 4) {
            float32x4_t v = vld1q_f32(&A[k][j]);
            v = vdivq_f32(v, v_pivot);
            vst1q_f32(&A[k][j], v);
        }
        for (; j < n; ++j) A[k][j] /= A[k][k];
        A[k][k] = 1.f;

        for (int i = k + 1; i < n; ++i) {
            float32x4_t v_fac = vdupq_n_f32(A[i][k]);
            int jj = k + 1;
            for (; jj + 3 < n; jj += 4) {
                float32x4_t v_k = vld1q_f32(&A[k][jj]);
                float32x4_t v_i = vld1q_f32(&A[i][jj]);
                v_i = vsubq_f32(v_i, vmulq_f32(v_k, v_fac));
                vst1q_f32(&A[i][jj], v_i);
            }
            for (; jj < n; ++jj) A[i][jj] -= A[i][k] * A[k][jj];
            A[i][k] = 0.f;
        }
    }
}

/*********************  Neon + Block64  *************************/
constexpr int Bsize = 64;          // 行块大小

static void gauss_neon_block64(int n)
{
    for (int kk = 0; kk < n; kk += Bsize) {
        int kend = std::min(kk + Bsize, n);

        /* ---- 块内消元 ---- */
        for (int k = kk; k < kend; ++k) {
            float32x4_t v_pivot = vdupq_n_f32(A[k][k]);

            int j = k + 1;
            for (; j + 3 < n; j += 4) {
                float32x4_t v = vld1q_f32(&A[k][j]);
                v = vdivq_f32(v, v_pivot);
                vst1q_f32(&A[k][j], v);
            }
            for (; j < n; ++j) A[k][j] /= A[k][k];
            A[k][k] = 1.f;

            /* k 行以下进行消元，只在本块内的 i 做“完全”消元，
               块外行照常处理。 */
            for (int i = k + 1; i < n; ++i) {
                float32x4_t v_fac = vdupq_n_f32(A[i][k]);
                int jj = k + 1;
                for (; jj + 3 < n; jj += 4) {
                    float32x4_t v_k = vld1q_f32(&A[k][jj]);
                    float32x4_t v_i = vld1q_f32(&A[i][jj]);
                    v_i = vsubq_f32(v_i, vmulq_f32(v_k, v_fac));
                    vst1q_f32(&A[i][jj], v_i);
                }
                for (; jj < n; ++jj) A[i][jj] -= A[i][k] * A[k][jj];
                A[i][k] = 0.f;
            }
            /* 预取下一行，降低 miss 延迟 */
            __builtin_prefetch(&A[k + 2][k], 0, 1);
        }
    }
}

/************************  计时器  *******************************/
template <typename F>
static double run_and_time(F func, int n)
{
    auto t0 = steady_ck::now();
    func(n);
    return ms(steady_ck::now() - t0).count();
}

/**************************  main  *******************************/
int main()
{
    init_matrix(N);

    restore(N);
    double t_serial = run_and_time(gauss_serial, N);
    std::cout << "Serial      : " << std::fixed << std::setprecision(2)
              << t_serial << " ms\n";

    restore(N);
    double t_neon = run_and_time(gauss_neon, N);
    std::cout << "Neon SIMD   : " << t_neon << " ms  (×"
              << t_serial / t_neon << ")\n";

    restore(N);
    double t_block = run_and_time(gauss_neon_block64, N);
    std::cout << "Neon+Block64: " << t_block << " ms  (×"
              << t_serial / t_block << ")\n";
    return 0;
}