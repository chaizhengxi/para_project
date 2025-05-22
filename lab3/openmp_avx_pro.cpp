#include <omp.h>
#include <iostream>
#include <fstream>
#include <windows.h>
#include <immintrin.h>
using namespace std;

// 定义矩阵尺寸和线程数
const int Num = 1362;          // 矩阵列数（消元子/行数据长度）
const int pasNum = 54274;      // 被消元行数量
const int lieNum = 43577;      // 消元子数量
const int NUM_THREADS = 7;     // 线程数量
const int CYCLE = 5;           // 测试循环次数

// 全局数据结构
unsigned int Act[43577][1363] = {0};  // 消元子矩阵
unsigned int Pas[54274][1363] = {0};  // 被消元行矩阵

// 初始化消元子数据
void init_A() {
    ifstream infile("act3.txt");
    char line[10000] = {0};
    int index, val;
    
    while (infile.getline(line, sizeof(line))) {
        istringstream iss(line);
        iss >> index; // 首项位置
        while (iss >> val) {
            int k = val % 32;
            int j = val / 32;
            Act[index][Num - 1 - j] |= (1 << k);
        }
        Act[index][Num] = 1; // 标记消元子有效
    }
}

// 初始化被消元行数据
void init_P() {
    ifstream infile("pas3.txt");
    char line[10000] = {0};
    int index = 0, val;
    
    while (infile.getline(line, sizeof(line)) && index < pasNum) {
        istringstream iss(line);
        iss >> val; // 首项位置
        Pas[index][Num] = val;
        while (iss >> val) {
            int k = val % 32;
            int j = val / 32;
            Pas[index][Num - 1 - j] |= (1 << k);
        }
        index++;
    }
}

// 普通高斯消去（串行）
void f_ordinary() {
    for (int i = lieNum - 1; i >= 0; --i) {
        for (int j = 0; j < pasNum; ++j) {
            while (Pas[j][Num] == i) {
                if (Act[i][Num]) { // 存在有效消元子
                    // 标量异或操作
                    for (int k = 0; k < Num; ++k)
                        Pas[j][k] ^= Act[i][k];
                    
                    // 更新首项位置
                    int num = 0, S_num = 0;
                    for (; num < Num; ++num) {
                        if (Pas[j][num]) {
                            S_num = num * 32 + __builtin_ctz(Pas[j][num]);
                            break;
                        }
                    }
                    Pas[j][Num] = (num < Num) ? S_num - 1 : -1;
                } else { // 升格为消元子
                    for (int k = 0; k < Num; ++k)
                        Act[i][k] = Pas[j][k];
                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }
}

// AVX向量化优化版本
void f_avx() {
    __m256i va_Pas, va_Act; // 使用整数向量化指令（更适合异或操作）
    
    for (int i = lieNum - 1; i >= 0; --i) {
        for (int j = 0; j < pasNum; ++j) {
            while (Pas[j][Num] == i) {
                if (Act[i][Num]) {
                    // AVX向量化异或（每次处理8个uint32_t）
                    for (int k = 0; k <= Num - 8; k += 8) {
                        va_Pas = _mm256_loadu_si256((__m256i*)&Pas[j][k]);
                        va_Act = _mm256_loadu_si256((__m256i*)&Act[i][k]);
                        va_Pas = _mm256_xor_si256(va_Pas, va_Act);
                        _mm256_storeu_si256((__m256i*)&Pas[j][k], va_Pas);
                    }
                    
                    // 处理剩余元素
                    for (int k = (Num / 8) * 8; k < Num; ++k)
                        Pas[j][k] ^= Act[i][k];
                    
                    // 更新首项位置（使用内置函数优化）
                    int num = 0;
                    for (; num < Num; ++num) {
                        if (Pas[j][num]) {
                            Pas[j][Num] = num * 32 + __builtin_ctz(Pas[j][num]) - 1;
                            break;
                        }
                    }
                    if (num == Num) Pas[j][Num] = -1;
                } else {
                    // 标量赋值升格
                    for (int k = 0; k < Num; ++k)
                        Act[i][k] = Pas[j][k];
                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }
}

// OpenMP+AVX优化版本（动态调度）
void f_omp_avx() {
    __m256i va_Pas, va_Act;
#pragma omp parallel num_threads(NUM_THREADS)
    {
        for (int i = lieNum - 1; i >= 0; --i) {
            // 并行处理被消元行
#pragma omp for schedule(dynamic, 64)
            for (int j = 0; j < pasNum; ++j) {
                while (Pas[j][Num] == i) {
                    if (Act[i][Num]) {
                        // AVX向量化异或
                        for (int k = 0; k <= Num - 8; k += 8) {
                            va_Pas = _mm256_loadu_si256((__m256i*)&Pas[j][k]);
                            va_Act = _mm256_loadu_si256((__m256i*)&Act[i][k]);
                            va_Pas = _mm256_xor_si256(va_Pas, va_Act);
                            _mm256_storeu_si256((__m256i*)&Pas[j][k], va_Pas);
                        }
                        
                        for (int k = (Num / 8) * 8; k < Num; ++k)
                            Pas[j][k] ^= Act[i][k];
                        
                        // 快速计算首项位置
                        int num = 0;
                        for (; num < Num; ++num) {
                            if (Pas[j][num]) {
                                Pas[j][Num] = num * 32 + __builtin_ctz(Pas[j][num]) - 1;
                                break;
                            }
                        }
                        if (num == Num) Pas[j][Num] = -1;
                    } else {
                        // 临界区保护升格操作（避免竞争）
#pragma omp critical
                        {
                            if (!Act[i][Num]) {
                                for (int k = 0; k < Num; ++k)
                                    Act[i][k] = Pas[j][k];
                                Act[i][Num] = 1;
                            }
                        }
                        break;
                    }
                }
            }
        }
    }
}

// 性能测试函数
void test_performance(const char* name, void (*func)()) {
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    double total = 0.0;

    for (int i = 0; i < CYCLE; ++i) {
        QueryPerformanceCounter(&start);
        func();
        QueryPerformanceCounter(&end);
        total += (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
    }
    cout << name << ": " << total / CYCLE << " ms" << endl;
}

int main() {
    init_A();
    init_P();

    test_performance("f_ordinary", f_ordinary);
    test_performance("f_avx", f_avx);
    test_performance("f_omp_avx", f_omp_avx);

    return 0;
}