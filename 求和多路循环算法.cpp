#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

// 多路循环算法：通过循环将数组中的元素逐步合并
void multi_loop_sum(int* numbers, int n) {
    for (; n > 1; n /= 2) {
        for (int i = 0; i < n / 2; i++) {
            numbers[i] = numbers[i * 2] + numbers[i * 2 + 1];
        }
    }
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

        for (int num = 0; num < times; num++) {
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            multi_loop_sum(a, n);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            total_time += (tail - head);
        }

        cout << (total_time) / times * 1000.0 / freq << endl;

        delete[] a;
    }

    return 0;
}