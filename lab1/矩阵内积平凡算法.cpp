#include <iostream>
#include <windows.h>
#include <stdlib.h>
using namespace std;

void column_access_algorithm(int** matrix, int* vector, int n, int* result) {
    for (int j = 0; j < n; j++) {
        result[j] = 0;
        for (int i = 0; i < n; i++) {
            result[j] += matrix[i][j] * vector[i];
        }
    }
}

int main() {
    long long head, tail, freq;
    QueryPerformanceCounter((LARGE_INTEGER*)&freq);

    int n = 0;
    double times = 100;

    for (; n <= 1000; n += 100) {
        // ��ʼ������ b ������ a
        int** b = new int*[n];
        for (int i = 0; i < n; i++) {
            b[i] = new int[n];
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i][j] = i + j;
            }
        } // ��ʼ�� b[i][j]

        int* a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        } // ��ʼ�� a

        int* sum = new int[n];
        long long total_time = 0;

        for (int num = 0; num < times; num++) {
            QueryPerformanceCounter((LARGE_INTEGER*)&head);

            // ����ƽ���㷨�����ڻ�
            column_access_algorithm(b, a, n, sum);

            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            total_time += tail - head;
        }

        // ���ƽ��ʱ��
        cout << (total_time) / times * 1000.0 / freq << endl;

        // �ͷ��ڴ�
        for (int i = 0; i < n; i++) {
            delete[] b[i];
        }
        delete[] b;
        delete[] a;
        delete[] sum;
    }

    return 0;
}