#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

// 平凡算法：逐个累加数组元素
void naive_sum(int* numbers, int size, int* result) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += numbers[i];
    }
    *result = sum;
}

int main() {
    long long head, tail, freq;
    QueryPerformanceCounter((LARGE_INTEGER*)&freq);

    int n = 524288;
    int times = 100;

    for (; n <= 536870912; n *= 2) {
        int* a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }

        long long total_time = 0;
        int result;

        for (int num = 0; num < times; num++) {
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            naive_sum(a, n, &result);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            total_time += (tail - head);
        }

        cout << (total_time) / times * 1000.0 / freq << endl;

        delete[] a;
    }

    return 0;
}