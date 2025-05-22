#include <pthread.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <arm_neon.h>
#include <cstring>
using namespace std;

// 矩阵常量定义
const int Num = 1362;          // 向量维度
const int pasNum = 54274;      // 被消元行数量
const int lieNum = 43577;      // 消元子数量
const int NUM_THREADS = 7;     // 线程数量

// 全局数据结构
unsigned int Act[43577][1363] = {0};  // 消元子矩阵
unsigned int Pas[54274][1363] = {0};  // 被消元行矩阵

// 线程同步机制
sem_t sem_leader;              // 主线程同步信号量
sem_t sem_next[NUM_THREADS-1]; // 工作线程同步信号量
bool sign = false;             // 全局标志：是否需要下一轮迭代

// 线程参数结构
struct threadParam_t {
    int t_id;  // 线程ID
};

// 初始化消元子矩阵
void init_A() {
    ifstream infile("act3.txt");
    if (!infile.is_open()) {
        cerr << "无法打开文件 act3.txt" << endl;
        exit(1);
    }
    
    string line;
    while (getline(infile, line)) {
        istringstream iss(line);
        int index;
        iss >> index;  // 读取行索引
        
        int value;
        while (iss >> value) {
            int bit = value % 32;
            int word = value / 32;
            Act[index][Num - 1 - word] |= (1 << bit);
        }
        Act[index][Num] = 1;  // 标记该行有效
    }
    infile.close();
}

// 初始化被消元行矩阵
void init_P() {
    ifstream infile("pas3.txt");
    if (!infile.is_open()) {
        cerr << "无法打开文件 pas3.txt" << endl;
        exit(1);
    }
    
    string line;
    int index = 0;
    while (getline(infile, line) && index < pasNum) {
        istringstream iss(line);
        int first;
        iss >> first;  // 读取首项值
        
        Pas[index][Num] = first;
        
        int value;
        while (iss >> value) {
            int bit = value % 32;
            int word = value / 32;
            Pas[index][Num - 1 - word] |= (1 << bit);
        }
        index++;
    }
    infile.close();
}

// 计算无符号整数的最高位位置
int find_msb(unsigned int x) {
    if (x == 0) return -1;
    return 31 - __builtin_clz(x);
}

// 线程主函数
void* threadFunc(void* param) {
    threadParam_t* p = static_cast<threadParam_t*>(param);
    int t_id = p->t_id;
    uint32x4_t va_Pas, va_Act;  // NEON寄存器
    
    do {
        // 阶段1：并行处理被消元行（不进行升格）
        for (int i = lieNum - 1; i >= 0; i--) {
            // 分块处理，每次处理8个消元子
            int block_size = (i >= 8) ? 8 : i + 1;
            int block_start = i - block_size + 1;
            
            for (int j = t_id; j < pasNum; j += NUM_THREADS) {
                int pivot = Pas[j][Num];
                if (pivot < block_start || pivot > i) continue;
                
                while (pivot >= block_start && pivot <= i) {
                    if (Act[pivot][Num]) {  // 消元子存在
                        // NEON向量化异或操作
                        for (int k = 0; k <= Num - 4; k += 4) {
                            va_Pas = vld1q_u32(&Pas[j][k]);
                            va_Act = vld1q_u32(&Act[pivot][k]);
                            va_Pas = veorq_u32(va_Pas, va_Act);
                            vst1q_u32(&Pas[j][k], va_Pas);
                        }
                        // 处理剩余元素
                        for (int k = (Num / 4) * 4; k < Num; k++) {
                            Pas[j][k] ^= Act[pivot][k];
                        }
                        
                        // 更新首项位置
                        int new_pivot = -1;
                        for (int k = 0; k < Num; k++) {
                            if (Pas[j][k] != 0) {
                                new_pivot = k * 32 + find_msb(Pas[j][k]);
                                break;
                            }
                        }
                        Pas[j][Num] = new_pivot;
                        pivot = new_pivot;
                    } else {
                        break;  // 消元子不存在，退出循环
                    }
                }
            }
            
            i -= block_size - 1;  // 跳到下一个块
        }
        
        // 线程同步：等待所有线程完成处理
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_wait(&sem_leader);  // 等待工作线程完成
            }
            
            // 阶段2：主线程进行消元子升格
            sign = false;
            for (int i = 0; i < pasNum; i++) {
                int pivot = Pas[i][Num];
                if (pivot == -1) continue;  // 已处理
                
                if (!Act[pivot][Num]) {  // 需要升格
                    memcpy(Act[pivot], Pas[i], sizeof(unsigned int) * Num);
                    Act[pivot][Num] = 1;
                    Pas[i][Num] = -1;
                    sign = true;
                }
            }
            
            // 通知所有工作线程继续
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_next[i]);
            }
        } else {
            // 工作线程通知主线程已完成
            sem_post(&sem_leader);
            // 等待主线程通知继续
            sem_wait(&sem_next[t_id - 1]);
        }
        
    } while (sign);  // 检查是否需要下一轮迭代
    
    pthread_exit(NULL);
}

int main() {
    // 初始化数据
    init_A();
    init_P();
    
    // 初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; i++) {
        sem_init(&sem_next[i], 0, 0);
    }
    
    // 开始计时
    timeval start, end;
    gettimeofday(&start, nullptr);
    
    // 创建线程
    pthread_t threads[NUM_THREADS];
    threadParam_t params[NUM_THREADS];
    
    for (int i = 0; i < NUM_THREADS; i++) {
        params[i].t_id = i;
        pthread_create(&threads[i], nullptr, threadFunc, &params[i]);
    }
    
    // 等待所有线程完成
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }
    
    // 停止计时
    gettimeofday(&end, nullptr);
    double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 + 
                     (end.tv_usec - start.tv_usec) / 1000.0;
    
    cout << "处理时间: " << time_ms << " ms" << endl;
    
    // 清理资源
    sem_destroy(&sem_leader);
    for (int i = 0; i < NUM_THREADS - 1; i++) {
        sem_destroy(&sem_next[i]);
    }
    
    return 0;
}