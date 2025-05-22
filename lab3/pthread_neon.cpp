#include <iostream>
#include <pthread.h>
#include <sys/time.h>
#include <arm_neon.h>
using namespace std;

// 矩阵规模和线程数
const int n = 1000;
float A[n][n];
const int NUM_THREADS = 7; // 工作线程数量

// 线程参数结构体
struct ThreadParam {
    int t_id; // 线程ID
};

// 信号量和屏障定义
sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS], sem_workerend[NUM_THREADS];
pthread_barrier_t barrier_div, barrier_elim;

// 初始化矩阵（生成非奇异矩阵）
void init() {
    srand(time(nullptr));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (i == j) ? 1.0f : rand() % 100;
        }
    }
    // 确保矩阵非奇异
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] += A[0][j];
        }
    }
}

// 普通串行高斯消去
void f_ordinary() {
    for (int k = 0; k < n; k++) {
        float pivot = A[k][k];
        for (int j = k + 1; j < n; j++) {
            A[k][j] /= pivot;
        }
        A[k][k] = 1.0f;
        
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            A[i][k] = 0.0f;
        }
    }
}

// 动态线程 + NEON向量化
void* thread_dynamic(void* arg) {
    ThreadParam* p = (ThreadParam*)arg;
    int t_id = p->t_id;
    float32x4_t vaik, vakj, vaij, vx;
    
    for (int k = 0; k < n; k++) {
        // 等待主线程完成除法
        pthread_barrier_wait(&barrier_div);
        
        // 分配任务：每个线程处理固定步长的行
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            vaik = vmovq_n_f32(A[i][k]);
            int j;
            for (j = k + 1; j <= n - 4; j += 4) {
                vakj = vld1q_f32(&A[k][j]);
                vaij = vld1q_f32(&A[i][j]);
                vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            // 处理剩余元素
            for (; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0f;
        }
        
        pthread_barrier_wait(&barrier_elim);
    }
    return nullptr;
}

// 静态线程 + 信号量同步
void pthread_static_sem() {
    // 初始化信号量
    sem_init(&sem_main, 0, 0);
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    
    pthread_t threads[NUM_THREADS];
    ThreadParam params[NUM_THREADS];
    
    // 创建线程
    for (int i = 0; i < NUM_THREADS; i++) {
        params[i].t_id = i;
        pthread_create(&threads[i], nullptr, thread_dynamic, &params[i]);
    }
    
    float32x4_t vt, va;
    for (int k = 0; k < n; k++) {
        // 主线程执行除法（NEON向量化）
        vt = vmovq_n_f32(A[k][k]);
        int j;
        for (j = k + 1; j <= n - 4; j += 4) {
            va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k][j], va);
        }
        for (; j < n; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0f;
        
        // 唤醒工作线程
        for (int i = 0; i < NUM_THREADS; i++) {
            sem_post(&sem_workerstart[i]);
        }
        
        // 等待所有工作线程完成
        for (int i = 0; i < NUM_THREADS; i++) {
            sem_wait(&sem_main);
        }
        
        // 通知进入下一轮
        for (int i = 0; i < NUM_THREADS; i++) {
            sem_post(&sem_workerend[i]);
        }
    }
    
    // 清理资源
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], nullptr);
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }
    sem_destroy(&sem_main);
}

// 静态线程 + 屏障同步
void pthread_static_barrier() {
    // 初始化屏障
    pthread_barrier_init(&barrier_div, nullptr, NUM_THREADS);
    pthread_barrier_init(&barrier_elim, nullptr, NUM_THREADS);
    
    pthread_t threads[NUM_THREADS];
    ThreadParam params[NUM_THREADS];
    
    for (int i = 0; i < NUM_THREADS; i++) {
        params[i].t_id = i;
        pthread_create(&threads[i], nullptr, thread_dynamic, &params[i]);
    }
    
    float32x4_t vt, va;
    for (int k = 0; k < n; k++) {
        // 主线程执行除法
        if (params[0].t_id == 0) {
            vt = vmovq_n_f32(A[k][k]);
            int j;
            for (j = k + 1; j <= n - 4; j += 4) {
                va = vld1q_f32(&A[k][j]);
                va = vdivq_f32(va, vt);
                vst1q_f32(&A[k][j], va);
            }
            for (; j < n; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0f;
        }
        
        // 同步点：等待所有线程完成除法
        pthread_barrier_wait(&barrier_div);
        
        // 同步点：等待所有线程完成消去
        pthread_barrier_wait(&barrier_elim);
    }
    
    // 清理屏障
    pthread_barrier_destroy(&barrier_div);
    pthread_barrier_destroy(&barrier_elim);
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }
}

// 性能测试函数
double measure_time(void (*func)()) {
    timeval start, end;
    gettimeofday(&start, nullptr);
    func();
    gettimeofday(&end, nullptr);
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
}

int main() {
    init();
    
    // 测试串行版本（注释掉以避免干扰多线程测试）
    // cout << "Serial: " << measure_time(f_ordinary) << " ms" << endl;
    
    cout << "Dynamic Threads: " << measure_time(pthread_static_sem) << " ms" << endl;
    cout << "Static + Semaphore: " << measure_time(pthread_static_sem) << " ms" << endl;
    cout << "Static + Barrier: " << measure_time(pthread_static_barrier) << " ms" << endl;
    
    return 0;
}