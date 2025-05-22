#include <omp.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <arm_neon.h> // NEON向量化支持

using namespace std;

// 定义矩阵尺寸和线程数
const int Num = 1362;          // 矩阵列数（每个消元子/被消元行的长度）
const int pasNum = 54274;      // 被消元行数量
const int lieNum = 43577;      // 消元子数量
const int NUM_THREADS = 7;     // 工作线程数

// 全局数据结构（消元子矩阵和被消元行矩阵）
unsigned int Act[43577][1363] = {0};
unsigned int Pas[54274][1363] = {0};

// 全局同步标志：是否需要继续下一轮升格
bool sign = false;

// 初始化消元子数据（从文件读取）
void init_A() {
    ifstream infile("act3.txt");
    char line[10000] = {0};
    int index, value;
    
    while (infile.getline(line, sizeof(line))) {
        istringstream iss(line);
        iss >> index; // 首项位置（消元子所在行号）
        
        int pos;
        while (iss >> pos) {
            int bit = pos % 32;        // 位位置（0-31）
            int word = pos / 32;       // 字位置（对应unsigned int索引）
            Act[index][Num - 1 - word] |= (1 << bit); // 存储位数据（倒序存储）
        }
        Act[index][Num] = 1; // 标记消元子有效
    }
}

// 初始化被消元行数据（从文件读取）
void init_P() {
    ifstream infile("pas3.txt");
    char line[10000] = {0};
    int index = 0, value;
    
    while (infile.getline(line, sizeof(line)) && index < pasNum) {
        istringstream iss(line);
        iss >> value; // 首项位置
        Pas[index][Num] = value; // 存储首项位置（初始值）
        
        int pos;
        while (iss >> pos) {
            int bit = pos % 32;
            int word = pos / 32;
            Pas[index][Num - 1 - word] |= (1 << bit); // 存储位数据（倒序存储）
        }
        index++;
    }
}

// 普通串行版本（标量运算）
void f_ordinary() {
    for (int i = lieNum - 1; i >= 0; --i) { // 按首项位置从高到低处理
        for (int j = 0; j < pasNum; ++j) {
            while (Pas[j][Num] == i) { // 处理首项等于当前i的被消元行
                if (Act[i][Num]) { // 存在有效消元子
                    // 标量异或操作
                    for (int k = 0; k < Num; ++k) {
                        Pas[j][k] ^= Act[i][k];
                    }
                    
                    // 更新首项位置
                    int num = 0, S_num = 0;
                    for (; num < Num; ++num) {
                        if (Pas[j][num]) {
                            S_num = num * 32 + __builtin_ctz(Pas[j][num]); // 计算最低有效位
                            break;
                        }
                    }
                    Pas[j][Num] = (num < Num) ? S_num - 1 : -1; // 首项位置减1（存储格式）
                } else { // 升格为消元子
                    for (int k = 0; k < Num; ++k) {
                        Act[i][k] = Pas[j][k];
                    }
                    Act[i][Num] = 1; // 标记消元子有效
                    break; // 升格后跳出循环
                }
            }
        }
    }
}

// OpenMP并行版本1（NEON向量化 + 静态调度 + 集中升格）
void f_omp1() {
    uint32x4_t va_Pas, va_Act; // NEON向量寄存器（4个uint32_t）
    
#pragma omp parallel num_threads(NUM_THREADS) private(va_Pas, va_Act)
    {
        bool local_sign; // 线程本地标志
        
        do {
            local_sign = false; // 初始化本地标志
            
            // 处理大块首项区间（每次处理8个首项）
            for (int i = lieNum - 1; i >= 8; i -= 8) {
#pragma omp for schedule(static)
                for (int j = 0; j < pasNum; ++j) {
                    while (Pas[j][Num] >= i - 7 && Pas[j][Num] <= i) { // 首项在[i-7, i]区间
                        int index = Pas[j][Num];
                        if (Act[index][Num]) {
                            // NEON向量化异或（每次处理4个uint32_t）
                            for (int k = 0; k <= Num - 4; k += 4) {
                                va_Pas = vld1q_u32(&Pas[j][k]);
                                va_Act = vld1q_u32(&Act[index][k]);
                                va_Pas = veorq_u32(va_Pas, va_Act);
                                vst1q_u32(&Pas[j][k], va_Pas);
                            }
                            // 处理剩余元素
                            for (int k = (Num / 4) * 4; k < Num; ++k) {
                                Pas[j][k] ^= Act[index][k];
                            }
                            
                            // 更新首项位置
                            int num = 0, S_num = 0;
                            for (; num < Num; ++num) {
                                if (Pas[j][num]) {
                                    S_num = num * 32 + __builtin_ctz(Pas[j][num]);
                                    break;
                                }
                            }
                            Pas[j][Num] = (num < Num) ? S_num - 1 : -1;
                        } else {
                            // 升格操作（需同步，仅单线程执行）
#pragma omp critical
                            {
                                if (!Act[index][Num]) {
                                    for (int k = 0; k < Num; ++k) {
                                        Act[index][k] = Pas[j][k];
                                    }
                                    Act[index][Num] = 1;
                                }
                            }
                            break;
                        }
                    }
                }
            }
            
            // 处理剩余首项（单个处理）
            for (int i = lieNum % 8 - 1; i >= 0; --i) {
#pragma omp for schedule(static)
                for (int j = 0; j < pasNum; ++j) {
                    while (Pas[j][Num] == i) {
                        if (Act[i][Num]) {
                            // NEON向量化异或
                            for (int k = 0; k <= Num - 4; k += 4) {
                                va_Pas = vld1q_u32(&Pas[j][k]);
                                va_Act = vld1q_u32(&Act[i][k]);
                                va_Pas = veorq_u32(va_Pas, va_Act);
                                vst1q_u32(&Pas[j][k], va_Pas);
                            }
                            for (int k = (Num / 4) * 4; k < Num; ++k) {
                                Pas[j][k] ^= Act[i][k];
                            }
                            
                            int num = 0, S_num = 0;
                            for (; num < Num; ++num) {
                                if (Pas[j][num]) {
                                    S_num = num * 32 + __builtin_ctz(Pas[j][num]);
                                    break;
                                }
                            }
                            Pas[j][Num] = (num < Num) ? S_num - 1 : -1;
                        } else {
#pragma omp critical
                            {
                                if (!Act[i][Num]) {
                                    for (int k = 0; k < Num; ++k) {
                                        Act[i][k] = Pas[j][k];
                                    }
                                    Act[i][Num] = 1;
                                }
                            }
                            break;
                        }
                    }
                }
            }
            
            // 集中检查升格需求（仅主线程执行）
#pragma omp single
            {
                sign = false;
                for (int i = 0; i < pasNum; ++i) {
                    int temp = Pas[i][Num];
                    if (temp == -1) continue; // 已处理的行跳过
                    if (!Act[temp][Num]) { // 需要升格
                        for (int k = 0; k < Num; ++k) {
                            Act[temp][k] = Pas[i][k];
                        }
                        Pas[i][Num] = -1;
                        sign = true; // 标记需要继续循环
                    }
                }
            }
            
            // 线程同步
#pragma omp barrier
            local_sign = sign; // 同步全局标志到本地
        } while (local_sign); // 所有线程根据本地标志决定是否继续
    }
}

// OpenMP并行版本2（仅并行消元，无向量化）
void f_omp2() {
#pragma omp parallel num_threads(NUM_THREADS)
    for (int i = lieNum - 1; i >= 0; --i) { // 按首项位置从高到低处理
        // 静态调度并行处理被消元行
#pragma omp for schedule(static)
        for (int j = 0; j < pasNum; ++j) {
            while (Pas[j][Num] == i) {
                if (Act[i][Num]) {
                    // 标量异或（无向量化）
                    for (int k = 0; k < Num; ++k) {
                        Pas[j][k] ^= Act[i][k];
                    }
                    
                    int num = 0, S_num = 0;
                    for (; num < Num; ++num) {
                        if (Pas[j][num]) {
                            S_num = num * 32 + __builtin_ctz(Pas[j][num]);
                            break;
                        }
                    }
                    Pas[j][Num] = (num < Num) ? S_num - 1 : -1;
                } else {
                    // 升格操作（临界区保护）
#pragma omp critical
                    {
                        if (!Act[i][Num]) {
                            for (int k = 0; k < Num; ++k) {
                                Act[i][k] = Pas[j][k];
                            }
                            Act[i][Num] = 1;
                        }
                    }
                    break;
                }
            }
        }
    }
}

// 性能测试函数
double measure_time(void (*func)()) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    func();
    gettimeofday(&end, NULL);
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
}

int main() {
    init_A();       // 初始化消元子
    init_P();       // 初始化被消元行
    
    // 测试串行版本
    double ordinary_time = measure_time(f_ordinary);
    cout << "Ordinary (serial): " << ordinary_time << " ms" << endl;
    
    // 测试OpenMP+NEON版本
    double omp1_time = measure_time(f_omp1);
    cout << "OpenMP+NEON (vectorized): " << omp1_time << " ms" << endl;
    
    // 测试OpenMP标量版本
    double omp2_time = measure_time(f_omp2);
    cout << "OpenMP (scalar): " << omp2_time << " ms" << endl;
    
    return 0;
}