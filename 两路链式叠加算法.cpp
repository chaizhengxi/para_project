#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

// 两路链式叠加算法：分两部分累加数组元素
void two_way_chain_sum(int* numbers, int size, int* result) {
    int sum1 = 0;
    int sum2 = 0;
    for (int i = 0; i < size - 1; i += 2) {
        sum1 += numbers[i];
        sum2 += numbers[i + 1];
    }
    *result = sum1 + sum2;
}

int main() {
    long long head, tail, freq;
    QueryPerformanceCounter((LARGE_INTEGER*)&freq);

    int n = 1024;
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
            two_way_chain_sum(a, n, &result);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            total_time += (tail - head);
        }

        cout << (total_time) / times * 1000.0 / freq << endl;

        delete[] a;
    }

    return 0;
}