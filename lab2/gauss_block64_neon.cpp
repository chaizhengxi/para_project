/********************************************************************
 *  高斯消元（单线程）+ NEON 微优化 demo
 *  ——————————————————————————————————————————————————————
 *  本文件演示四种 kernel：
 *  ① gauss_serial             : 纯标量基线
 *  ② gauss_serial_block64     : 行分块 + 预取
 *  ③ gauss_neon               : NEON128 向量化
 *  ④ gauss_neon_block64       : NEON128 + 行分块
 *  
 *  代码已插入大量中文注释，主要讨论：
 *    • 数据布局与访存模型
 *    • SIMD 浮点指令用法
 *    • 为什么这些优化提速不明显
 *******************************************************************/

#include <arm_neon.h>      // NEON intrinsics
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

constexpr int  BLOCK  = 64;     // 行块(block)大小；只在“受消去行”维度分块
constexpr int  REPEAT = 5;      // 计时取多次均值
constexpr uint32_t SEED = 2025; // RNG 种子，保证结果可复现

/********************************************************************
 * make_matrix()
 * ---------------------------------------------------------------
 * 构造一个必定非奇异的随机矩阵：
 *   先写上三角随机数，保证对角线非零 → 再把每行累加到上一行，
 *   这样可以避免后续高斯消元出现“选主元 / 行交换”分支。
 *******************************************************************/
void make_matrix(std::vector<float>& a, int n, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dis(1.f, 10.f);

    /* 生成上三角 */
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            a[i * n + j] = (j >= i ? dis(rng) : 0.f);

    /* 行累加，使得对角线元素更大、矩阵条件数更好 */
    for (int i = n - 2; i >= 0; --i)
        for (int j = 0; j < n; ++j)
            a[i * n + j] += a[(i + 1) * n + j];
}

/********************************************************************
 * 1) gauss_serial : 纯标量实现
 *******************************************************************/
void gauss_serial(float* A, int n)
{
    for (int k = 0; k < n; ++k) {             // k = 主元(pivot) 行/列
        float pivot_inv = 1.0f / A[k * n + k]; // 把 pivot 取倒数
        for (int j = k; j < n; ++j)            // ★ 这里横向扫描 (k,k..n)
            A[k * n + j] *= pivot_inv;         // 主元行归一化

        /* 消去主元列下方所有行 */
        for (int i = k + 1; i < n; ++i) {
            float factor = A[i * n + k];       // 该行的系数
            for (int j = k; j < n; ++j)        // ★ 横向扫描造成大量带宽消耗
                A[i * n + j] -= factor * A[k * n + j];
            A[i * n + k] = 0.f;                // 填 0 只是视觉整洁
        }
    }
}

/********************************************************************
 * 2) gauss_serial_block64 : 行分块 + 预取
 * ---------------------------------------------------------------
 * 思路：一次只处理 BLOCK (=64) 行，期望把它们一起留在缓存中。
 * 问题：列方向依旧 J=k..n 顺序扫，pivot 行在每轮都要重新进缓存，
 *       带宽瓶颈没有解除 → 提速很有限。
 *******************************************************************/
void gauss_serial_block64(float* A, int n)
{
    for (int k = 0; k < n; ++k) {
        /* --- pivot 行归一化：与基线相同 --- */
        float pivot_inv = 1.0f / A[k * n + k];
        for (int j = k; j < n; ++j)
            A[k * n + j] *= pivot_inv;

        /* --- 下三角分块消去 --- */
        for (int ii = k + 1; ii < n; ii += BLOCK) {
            int i_end = std::min(ii + BLOCK, n);

            for (int i = ii; i < i_end; ++i) {
                /* 仅预取列 k 的下 8 行；真正热点是整行 → 作用很小 */
                __builtin_prefetch(&A[(i + 8) * n + k], 0, 1);

                float factor = A[i * n + k];
                for (int j = k; j < n; ++j)      // ★ 列方向仍全扫
                    A[i * n + j] -= factor * A[k * n + j];
                A[i * n + k] = 0.f;
            }
        }
    }
}

/********************************************************************
 * 3) gauss_neon : NEON128 向量化基线
 * ---------------------------------------------------------------
 * 利用 128-bit 向量 (4×float) 将“mul & sub”打包计算。
 * 计算-访存比依旧 ≈0.25 FLOP/B，带宽仍是主瓶颈，
 * 因此 NEON 提升 ~1.5–2× 后就触顶。
 *******************************************************************/
void gauss_neon(float* A, int n)
{
    for (int k = 0; k < n; ++k) {
        /* ---------- 归一化主元行 ----------- */
        float pivot_inv  = 1.0f / A[k * n + k];
        float32x4_t piv4 = vdupq_n_f32(pivot_inv);   // 向量化 pivot 倒数

        int j = k;
        for (; j + 4 <= n; j += 4) {                 // 4×float 对齐循环
            float32x4_t row = vld1q_f32(&A[k * n + j]);
            row = vmulq_f32(row, piv4);
            vst1q_f32(&A[k * n + j], row);
        }
        for (; j < n; ++j)                           // 处理尾数
            A[k * n + j] *= pivot_inv;

        /* ---------- 消去主元列以下的行 ---------- */
        for (int i = k + 1; i < n; ++i) {
            float factor     = A[i * n + k];
            float32x4_t f4   = vdupq_n_f32(factor);

            j = k;
            for (; j + 4 <= n; j += 4) {
                float32x4_t dst = vld1q_f32(&A[i * n + j]);
                float32x4_t src = vld1q_f32(&A[k * n + j]);
                dst = vsubq_f32(dst, vmulq_f32(f4, src)); // dst -= f*src
                vst1q_f32(&A[i * n + j], dst);
            }
            for (; j < n; ++j)
                A[i * n + j] -= factor * A[k * n + j];

            A[i * n + k] = 0.f;
        }
    }
}

/********************************************************************
 * 4) gauss_neon_block64 : NEON + 行分块
 * ---------------------------------------------------------------
 * 先用 NEON 做向量化，再在 i 维度加 BLOCK=64
 * 由于带宽早已饱和，“再行分块”几乎看不到增益。
 *******************************************************************/
void gauss_neon_block64(float* A, int n)
{
    for (int k = 0; k < n; ++k) {
        /* ----- 归一化 pivot 行 ----- */
        float pivot_inv  = 1.0f / A[k * n + k];
        float32x4_t piv4 = vdupq_n_f32(pivot_inv);

        int j = k;
        for (; j + 4 <= n; j += 4) {
            float32x4_t row = vld1q_f32(&A[k * n + j]);
            row = vmulq_f32(row, piv4);
            vst1q_f32(&A[k * n + j], row);
        }
        for (; j < n; ++j) A[k * n + j] *= pivot_inv;

        /* ----- 分块消去下三角 ----- */
        for (int ii = k + 1; ii < n; ii += BLOCK) {
            int i_end = std::min(ii + BLOCK, n);

            for (int i = ii; i < i_end; ++i) {
                __builtin_prefetch(&A[(i + 8) * n + k], 0, 1); // 同样作用有限

                float factor   = A[i * n + k];
                float32x4_t f4 = vdupq_n_f32(factor);

                j = k;
                for (; j + 4 <= n; j += 4) {
                    float32x4_t dst = vld1q_f32(&A[i * n + j]);
                    float32x4_t src = vld1q_f32(&A[k * n + j]);
                    dst = vsubq_f32(dst, vmulq_f32(f4, src));
                    vst1q_f32(&A[i * n + j], dst);
                }
                for (; j < n; ++j)
                    A[i * n + j] -= factor * A[k * n + j];
                A[i * n + k] = 0.f;
            }
        }
    }
}

/********************************************************************
 * bench() : 计时代码
 * ---------------------------------------------------------------
 * 每次都 copy 一份矩阵 (约 n²*4 Byte)，n=3000 时拷贝 ~36 MB；
 * 这部分开销也会被统计进去，导致“观察到的”加速比被稀释。
 *******************************************************************/
template <typename Fn>
double bench(Fn func, const std::vector<float>& src, int n)
{
    std::vector<float> a(src);          // 深拷贝，确保输入一致
    auto t0 = Clock::now();
    func(a.data(), n);                  // 调 kernel
    auto t1 = Clock::now();
    return Ms(t1 - t0).count();
}

/********************************************************************
 * main() : 跑 500~3000 的矩阵尺寸，打印两张性能对比表
 *******************************************************************/
int main()
{
    std::mt19937 rng(SEED);
    const int sizes[] = {500, 1000, 1500, 2000, 2500, 3000};

    std::cout << std::fixed << std::setprecision(1);

    //================ 表 2：串行 vs. 行分块 =========================
    std::cout << "\n表 2：串行 + Block64 运行时间 (ms)\n";
    std::cout << "  n     基线串行   Block64   提速(%)\n";

    for (int n : sizes) {
        std::vector<float> A(n * n);
        make_matrix(A, n, rng);

        double t_serial = 0, t_blk = 0;
        for (int r = 0; r < REPEAT; ++r) t_serial += bench(gauss_serial,          A, n);
        for (int r = 0; r < REPEAT; ++r) t_blk    += bench(gauss_serial_block64, A, n);
        t_serial /= REPEAT;
        t_blk    /= REPEAT;

        double speed = (t_serial - t_blk) / t_serial * 100.0;
        std::cout << std::setw(5) << n << std::setw(11) << t_serial
                  << std::setw(10) << t_blk << std::setw(9) << speed << "\n";
    }

    //================ 表 3：NEON vs. NEON+Block =====================
    std::cout << "\n表 3：Neon SIMD + Block64 运行时间 (ms)\n";
    std::cout << "  n     基线Neon  Neon+Blk64 提速(%)\n";

    for (int n : sizes) {
        std::vector<float> A(n * n);
        make_matrix(A, n, rng);

        double t_neon = 0, t_blk = 0;
        for (int r = 0; r < REPEAT; ++r) t_neon += bench(gauss_neon,          A, n);
        for (int r = 0; r < REPEAT; ++r) t_blk  += bench(gauss_neon_block64, A, n);
        t_neon /= REPEAT;
        t_blk  /= REPEAT;

        double speed = (t_neon - t_blk) / t_neon * 100.0;
        std::cout << std::setw(5) << n << std::setw(11) << t_neon
                  << std::setw(10) << t_blk << std::setw(9) << speed << "\n";
    }
}

