#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

// Cache 优化算法：逐行访问矩阵元素，计算内积
void row_access_algorithm(int** matrix, int* vector, int n, int* result) {
    for (int i = 0; i < n; i++) {
        result[i] = 0;
        for (int j = 0; j < n; j++) {
            result[i] += matrix[j][i] * vector[j];
        }
    }
}

int main() {
    long long head, tail, freq;
    QueryPerformanceCounter((LARGE_INTEGER*)&freq);

    int n = 0;
    double times = 1;

    for (n = 20000; n <= 50000; n += 10000) {
        // 初始化矩阵 b 和向量 a
        int** b = new int*[n];
        for (int i = 0; i < n; i++) {
            b[i] = new int[n];
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i][j] = i + j;
            }
        } // 初始化 b[i][j]

        int* a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        } // 初始化 a

        int* sum = new int[n];
        long long total_time = 0;

        for (int num = 0; num < times; num++) {
            QueryPerformanceCounter((LARGE_INTEGER*)&head);

            // 调用 Cache 优化算法计算内积
            row_access_algorithm(b, a, n, sum);

            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            total_time += tail - head;
        }

        // 输出平均时间
        cout << (total_time) / times * 1000.0 / freq << endl;

        // 释放内存
        for (int i = 0; i < n; i++) {
            delete[] b[i];
        }
        delete[] b;
        delete[] a;
        delete[] sum;
    }

    return 0;
}