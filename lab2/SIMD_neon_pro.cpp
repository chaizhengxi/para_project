#include <arm_neon.h>
#include <sys/time.h>
#include <iostream>
#include <sstream>
#include <fstream>

// 定义矩阵大小
const int ROW_COUNT = 37960;
const int COL_COUNT = 1188;
const int PAS_ROW_COUNT = 14921;

// 消元子矩阵和被消元行矩阵
unsigned int eliminators[ROW_COUNT][COL_COUNT] = { 0 };
unsigned int rows_to_eliminate[PAS_ROW_COUNT][COL_COUNT] = { 0 };

// 初始化消元子矩阵
void initialize_eliminators() {
    std::ifstream file("act.txt");
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int row_index;
        iss >> row_index;
        int num;
        while (iss >> num) {
            int bit_index = num % 32;
            int col_index = num / 32;
            eliminators[row_index][COL_COUNT - 2 - col_index] |= (1 << bit_index);
            eliminators[row_index][COL_COUNT - 1] = 1;
        }
    }
}

// 初始化被消元行矩阵
void initialize_rows_to_eliminate() {
    std::ifstream file("pas.txt");
    std::string line;
    int row_index = 0;
    while (std::getline(file, line) && row_index < PAS_ROW_COUNT) {
        std::istringstream iss(line);
        int first_num;
        iss >> first_num;
        rows_to_eliminate[row_index][COL_COUNT - 1] = first_num;
        int num;
        while (iss >> num) {
            int bit_index = num % 32;
            int col_index = num / 32;
            rows_to_eliminate[row_index][COL_COUNT - 2 - col_index] |= (1 << bit_index);
        }
        row_index++;
    }
}

// 普通串行消元函数
void serial_elimination() {
    for (int i = ROW_COUNT - 1; i >= 0; i -= 8) {
        for (int j = 0; j < PAS_ROW_COUNT; j++) {
            int leading_term = rows_to_eliminate[j][COL_COUNT - 1];
            while (leading_term <= i && leading_term >= i - 7) {
                if (eliminators[leading_term][COL_COUNT - 1] == 1) {
                    for (int k = 0; k < COL_COUNT - 1; k++) {
                        rows_to_eliminate[j][k] ^= eliminators[leading_term][k];
                    }
                    int new_leading_term = 0;
                    for (int k = 0; k < COL_COUNT - 1; k++) {
                        if (rows_to_eliminate[j][k] != 0) {
                            unsigned int temp = rows_to_eliminate[j][k];
                            int bit_count = 0;
                            while (temp) {
                                temp >>= 1;
                                bit_count++;
                            }
                            new_leading_term = k * 32 + bit_count - 1;
                            break;
                        }
                    }
                    rows_to_eliminate[j][COL_COUNT - 1] = new_leading_term;
                } else {
                    for (int k = 0; k < COL_COUNT; k++) {
                        eliminators[leading_term][k] = rows_to_eliminate[j][k];
                    }
                    eliminators[leading_term][COL_COUNT - 1] = 1;
                    break;
                }
                leading_term = rows_to_eliminate[j][COL_COUNT - 1];
            }
        }
    }
}

// 使用NEON指令并行消元函数
void neon_parallel_elimination() {
    for (int i = ROW_COUNT - 1; i >= 0; i -= 8) {
        for (int j = 0; j < PAS_ROW_COUNT; j++) {
            int leading_term = rows_to_eliminate[j][COL_COUNT - 1];
            while (leading_term <= i && leading_term >= i - 7) {
                if (eliminators[leading_term][COL_COUNT - 1] == 1) {
                    int k;
                    for (k = 0; k + 4 <= COL_COUNT - 1; k += 4) {
                        uint32x4_t vec_rows = vld1q_u32(&rows_to_eliminate[j][k]);
                        uint32x4_t vec_elim = vld1q_u32(&eliminators[leading_term][k]);
                        vec_rows = veorq_u32(vec_rows, vec_elim);
                        vst1q_u32(&rows_to_eliminate[j][k], vec_rows);
                    }
                    for (; k < COL_COUNT - 1; k++) {
                        rows_to_eliminate[j][k] ^= eliminators[leading_term][k];
                    }
                    int new_leading_term = 0;
                    for (int k = 0; k < COL_COUNT - 1; k++) {
                        if (rows_to_eliminate[j][k] != 0) {
                            unsigned int temp = rows_to_eliminate[j][k];
                            int bit_count = 0;
                            while (temp) {
                                temp >>= 1;
                                bit_count++;
                            }
                            new_leading_term = k * 32 + bit_count - 1;
                            break;
                        }
                    }
                    rows_to_eliminate[j][COL_COUNT - 1] = new_leading_term;
                } else {
                    for (int k = 0; k < COL_COUNT; k++) {
                        eliminators[leading_term][k] = rows_to_eliminate[j][k];
                    }
                    eliminators[leading_term][COL_COUNT - 1] = 1;
                    break;
                }
                leading_term = rows_to_eliminate[j][COL_COUNT - 1];
            }
        }
    }
}

// 测量函数执行时间
double measure_time(void (*func)()) {
    struct timeval start, end;
    gettimeofday(&start, nullptr);
    func();
    gettimeofday(&end, nullptr);
    return ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) / 1000.0;
}

int main() {
    initialize_eliminators();
    initialize_rows_to_eliminate();
    double serial_time = measure_time(serial_elimination);
    std::cout << "Serial elimination time: " << serial_time << " ms" << std::endl;

    initialize_eliminators();
    initialize_rows_to_eliminate();
    double parallel_time = measure_time(neon_parallel_elimination);
    std::cout << "Parallel elimination time: " << parallel_time << " ms" << std::endl;

    return 0;
}    