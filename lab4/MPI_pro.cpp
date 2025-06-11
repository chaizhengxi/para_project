#include <windows.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <mpi.h>
#include <pmmintrin.h>
#include <omp.h>
using namespace std;

static const int thread_count = 4;

unsigned int Act[43577][1363] = {0};
unsigned int Pas[54274][1363] = {0};

const int Num = 1362;
const int pasNum = 54274;
const int lieNum = 43577;

void init_A() {
    unsigned int a;
    ifstream infile("act3.txt");
    char fin[10000] = {0};
    int index;
    while (infile.getline(fin, sizeof(fin))) {
        stringstream line(fin);
        int biaoji = 0;
        while (line >> a) {
            if (biaoji == 0) {
                index = a;
                biaoji = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            Act[index][Num - 1 - j] += temp;
            Act[index][Num] = 1;
        }
    }
}

void init_P() {
    unsigned int a;
    ifstream infile("pas3.txt");
    char fin[10000] = {0};
    int index = 0;
    while (infile.getline(fin, sizeof(fin))) {
        stringstream line(fin);
        int biaoji = 0;
        while (line >> a) {
            if (biaoji == 0) {
                Pas[index][Num] = a;
                biaoji = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            Pas[index][Num - 1 - j] += temp;
        }
        index++;
    }
}

void f_ordinary() {
    double seconds = 0;
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    int i;
    for (i = lieNum - 1; i - 8 >= -1; i -= 8) {
        for (int j = 0; j < pasNum; j++) {
            while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7) {
                int index = Pas[j][Num];
                if (Act[index][Num] == 1) {
                    for (int k = 0; k < Num; k++) {
                        Pas[j][k] = Pas[j][k] ^ Act[index][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++) {
                        if (Pas[j][num] != 0) {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0) {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;
                } else {
                    for (int k = 0; k < Num; k++)
                        Act[index][k] = Pas[j][k];
                    Act[index][Num] = 1;
                    break;
                }
            }
        }
    }

    for (int i = lieNum % 8 - 1; i >= 0; i--) {
        for (int j = 0; j < pasNum; j++) {
            while (Pas[j][Num] == i) {
                if (Act[i][Num] == 1) {
                    for (int k = 0; k < Num; k++) {
                        Pas[j][k] = Pas[j][k] ^ Act[i][k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < Num; num++) {
                        if (Pas[j][num] != 0) {
                            unsigned int temp = Pas[j][num];
                            while (temp != 0) {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    Pas[j][Num] = S_num - 1;
                } else {
                    for (int k = 0; k < Num; k++)
                        Act[i][k] = Pas[j][k];
                    Act[i][Num] = 1;
                    break;
                }
            }
        }
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    seconds = (tail - head) * 1000.0 / freq;
    cout << "f_ordinary_pro: " << seconds << " ms" << endl;
}

void LU_pro(int rank, int num_proc) {
    int i;
    for (i = lieNum - 1; i - 8 >= -1; i -= 8) {
        for (int j = 0; j < pasNum; j++) {
            if (j % num_proc == rank) {
                while (Pas[j][Num] <= i && Pas[j][Num] >= i - 7) {
                    int index = Pas[j][Num];
                    if (Act[index][Num] == 1) {
                        __m128 va_Pas, va_Act;
                        for (int k = 0; k + 4 <= Num; k += 4) {
                            va_Pas = _mm_loadu_ps((float*)&Pas[j][k]);
                            va_Act = _mm_loadu_ps((float*)&Act[index][k]);
                            va_Pas = _mm_xor_ps(va_Pas, va_Act);
                            _mm_store_ss((float*)&Pas[j][k], va_Pas);
                        }
                        for (; k < Num; k++) {
                            Pas[j][k] = Pas[j][k] ^ Act[index][k];
                        }
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++) {
                            if (Pas[j][num] != 0) {
                                unsigned int temp = Pas[j][num];
                                while (temp != 0) {
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Pas[j][Num] = S_num - 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    for (int i = lieNum % 8 - 1; i >= 0; i--) {
        for (int j = 0; j < pasNum; j++) {
            if (j % num_proc == rank) {
                while (Pas[j][Num] == i) {
                    if (Act[i][Num] == 1) {
                        __m128 va_Pas, va_Act;
                        for (int k = 0; k + 4 <= Num; k += 4) {
                            va_Pas = _mm_loadu_ps((float*)&Pas[j][k]);
                            va_Act = _mm_loadu_ps((float*)&Act[i][k]);
                            va_Pas = _mm_xor_ps(va_Pas, va_Act);
                            _mm_store_ss((float*)&Pas[j][k], va_Pas);
                        }
                        for (; k < Num; k++) {
                            Pas[j][k] = Pas[j][k] ^ Act[i][k];
                        }
                        int num = 0, S_num = 0;
                        for (num = 0; num < Num; num++) {
                            if (Pas[j][num] != 0) {
                                unsigned int temp = Pas[j][num];
                                while (temp != 0) {
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Pas[j][Num] = S_num - 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }
}

void f_mpi_pro() {
    int num_proc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double seconds = 0;
    long long head, tail, freq;

    if (rank == 0) {
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        int sign;
        do {
            for (int i = 0; i < pasNum; i++) {
                int flag = i % num_proc;
                if (flag == rank) continue;
                MPI_Send(&Pas[i], Num + 1, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
            }
            LU_pro(rank, num_proc);
            for (int i = 0; i < pasNum; i++) {
                int flag = i % num_proc;
                if (flag == rank) continue;
                MPI_Recv(&Pas[i], Num + 1, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            sign = 0;
            for (int i = 0; i < pasNum; i++) {
                int temp = Pas[i][Num];
                if (temp == -1) continue;
                if (Act[temp][Num] == 0) {
                    for (int k = 0; k < Num; k++) Act[temp][k] = Pas[i][k];
                    Pas[i][Num] = -1;
                    sign = 1;
                }
            }
            for (int r = 1; r < num_proc; r++) {
                MPI_Send(&sign, 1, MPI_INT, r, 2, MPI_COMM_WORLD);
            }
        } while (sign == 1);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        seconds = (tail - head) * 1000.0 / freq;
        cout << "f_mpi_pro: " << seconds << " ms" << endl;
    } else {
        int sign;
        do {
            for (int i = rank; i < pasNum; i += num_proc) {
                MPI_Recv(&Pas[i], Num + 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            LU_pro(rank, num_proc);
            for (int i = rank; i < pasNum; i += num_proc) {
                MPI_Send(&Pas[i], Num + 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            }
            MPI_Recv(&sign, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } while (sign == 1);
    }
}

int main() {
    init_A();
    init_P();
    f_ordinary();

    init_A();
    init_P();
    MPI_Init(NULL, NULL);
    f_mpi_pro();
    MPI_Finalize();
}