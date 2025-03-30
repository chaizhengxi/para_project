#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

// 递归算法：分治策略求和
void recursive_sum(int* array, int n, int* result) {
    if (n == 1) {
        *result = array[0];
        return;
    }
    int mid = n / 2;
    int left_sum, right_sum;
    recursive_sum(array, mid, &left_sum);
    recursive_sum(array + mid, n - mid, &right_sum);
    *result = left_sum + right_sum;
}

int main() {
    long long head, tail, freq;
    QueryPerformanceCounter((LARGE_INTEGER*)&freq);

    int n = 1024;
    int times = 100;

    for (; n <= 67108864; n *= 2) {
        int* a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }

        long long total_time = 0;
        int result;

        for (int num = 0; num < times; num++) {
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            recursive_sum(a, n, &result);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            total_time += (tail - head);
        }

        cout << (total_time) / times * 1000.0 / freq << endl;

        delete[] a;
    }

    return 0;
}