#include <immintrin.h> // 包含SIMD指令头文件
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
using namespace std;

// 矩阵规模定义（以最大测试用例为例）
const int COL = 113045;
const int mat_L = 32; // 每个unsigned int存储32位
const int ROW = 47381; // 被消元行数量
const int ELE_ROW = COL; // 消元子行数等于列数

unsigned int Act[ELE_ROW][COL/mat_L + 1] = {0}; // 消元子矩阵
unsigned int Pas[ROW][COL/mat_L + 1] = {0};      // 被消元行矩阵

// 消元子初始化（从文件读取数据）
void init_A() {
    ifstream infile("act2.txt");
    string line;
    int index, a;
    while (getline(infile, line)) {
        istringstream iss(line);
        iss >> index;
        while (iss >> a) {
            int block = a / mat_L;           // 计算32位块索引
            int bit = a % mat_L;             // 计算块内位位置
            Act[index][block] |= (1 << bit); // 设置对应位
        }
        Act[index][COL/mat_L] = 1; // 标记非空消元子
    }
}

// 被消元行初始化（从文件读取数据）
void init_P() {
    ifstream infile("pas2.txt");
    string line;
    int index = 0, a;
    while (getline(infile, line) && index < ROW) {
        istringstream iss(line);
        iss >> Pas[index][COL/mat_L]; // 存储首项位置
        while (iss >> a) {
            int block = a / mat_L;
            int bit = a % mat_L;
            Pas[index][block] |= (1 << bit);
        }
        index++;
    }
}

// 串行算法实现
void f_ordinary() {
    for (int i = 0; i < ROW; i++) { // 遍历被消元行
        for (int j = COL; j >= 0; j--) { // 从高列到低列寻找首项
            int block = j / mat_L;
            int bit = j % mat_L;
            if (Pas[i][block] & (1 << bit)) { // 当前行在j列有元素
                if (Act[j][block] & (1 << bit)) { // 存在对应消元子
                    // 逐块异或
                    for (int p = COL/mat_L; p >= 0; p--) {
                        Pas[i][p] ^= Act[j][p];
                    }
                } else { // 升格为消元子
                    for (int p = 0; p <= COL/mat_L; p++) {
                        Act[j][p] = Pas[i][p];
                    }
                    Act[j][COL/mat_L] = 1; // 标记为非空消元子
                    break; // 处理完当前行
                }
            }
        }
    }
}

// SSE并行优化（128位，4路uint32_t）
void f_sse() {
    for (int i = 0; i < ROW; i++) {
        for (int j = COL; j >= 0; j--) {
            int block = j / mat_L;
            int bit = j % mat_L;
            if (Pas[i][block] & (1 << bit)) {
                if (Act[j][block] & (1 << bit)) {
                    // 向量化异或处理前4n个块
                    for (int p = 0; p <= COL/mat_L; p += 4) {
                        // 加载4个uint32_t块（128位）
                        __m128i va_pas = _mm_loadu_si128((__m128i*)(&Pas[i][p]));
                        __m128i va_act = _mm_loadu_si128((__m128i*)(&Act[j][p]));
                        // 整数异或操作
                        __m128i vx = _mm_xor_si128(va_pas, va_act);
                        // 存储结果
                        _mm_storeu_si128((__m128i*)(&Pas[i][p]), vx);
                    }
                    // 处理剩余块
                    for (int p = (COL/mat_L/4)*4; p <= COL/mat_L; p++) {
                        Pas[i][p] ^= Act[j][p];
                    }
                } else {
                    // 升格处理（串行）
                    for (int p = 0; p <= COL/mat_L; p++) {
                        Act[j][p] = Pas[i][p];
                    }
                    Act[j][COL/mat_L] = 1;
                    break;
                }
            }
        }
    }
}

// AVX256并行优化（256位，8路uint32_t）
void f_avx256() {
    for (int i = 0; i < ROW; i++) {
        for (int j = COL; j >= 0; j--) {
            int block = j / mat_L;
            int bit = j % mat_L;
            if (Pas[i][block] & (1 << bit)) {
                if (Act[j][block] & (1 << bit)) {
                    // 向量化异或处理前8n个块
                    for (int p = 0; p <= COL/mat_L; p += 8) {
                        __m256i va_pas = _mm256_loadu_si256((__m256i*)(&Pas[i][p]));
                        __m256i va_act = _mm256_loadu_si256((__m256i*)(&Act[j][p]));
                        __m256i vx = _mm256_xor_si256(va_pas, va_act);
                        _mm256_storeu_si256((__m256i*)(&Pas[i][p]), vx);
                    }
                    // 处理剩余块
                    for (int p = (COL/mat_L/8)*8; p <= COL/mat_L; p++) {
                        Pas[i][p] ^= Act[j][p];
                    }
                } else {
                    // 升格处理（串行）
                    for (int p = 0; p <= COL/mat_L; p++) {
                        Act[j][p] = Pas[i][p];
                    }
                    Act[j][COL/mat_L] = 1;
                    break;
                }
            }
        }
    }
}

// AVX512并行优化（512位，16路uint32_t）
void f_avx512() {
    for (int i = 0; i < ROW; i++) {
        for (int j = COL; j >= 0; j--) {
            int block = j / mat_L;
            int bit = j % mat_L;
            if (Pas[i][block] & (1 << bit)) {
                if (Act[j][block] & (1 << bit)) {
                    // 向量化异或处理前16n个块
                    for (int p = 0; p <= COL/mat_L; p += 16) {
                        __m512i va_pas = _mm512_loadu_si512((__m512i*)(&Pas[i][p]));
                        __m512i va_act = _mm512_loadu_si512((__m512i*)(&Act[j][p]));
                        __m512i vx = _mm512_xor_epi32(va_pas, va_act);
                        _mm512_storeu_si512((__m512i*)(&Pas[i][p]), vx);
                    }
                    // 处理剩余块
                    for (int p = (COL/mat_L/16)*16; p <= COL/mat_L; p++) {
                        Pas[i][p] ^= Act[j][p];
                    }
                } else {
                    // 升格处理（串行）
                    for (int p = 0; p <= COL/mat_L; p++) {
                        Act[j][p] = Pas[i][p];
                    }
                    Act[j][COL/mat_L] = 1;
                    break;
                }
            }
        }
    }
}

// 性能测试框架
template<typename Func>
void test_perf(Func func, const string& name) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
    cout << name << " time: " << duration << " μs" << endl;
}

int main() {
    init_A();
    init_P();

    test_perf(f_ordinary, "Ordinary");
    test_perf(f_sse, "SSE");
    test_perf(f_avx256, "AVX256");
    test_perf(f_avx512, "AVX512");

    return 0;
}