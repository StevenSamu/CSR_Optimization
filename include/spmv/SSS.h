#include <string>
#include <vector>
#include <iostream>
#include <immintrin.h> // Include header for SIMD instructions (AVX)
#include <thread>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <stdexcept>

struct SSSMatrix {
    std::vector<double> dval;         // Diagonal values
    std::vector<double> val;          // Non-diagonal values
    std::vector<int> col_idx;         // Column indices for the lower triangular part
    std::vector<int> row_ptr;         // Row pointers for the lower triangular part
};

SSSMatrix readMTXtoSSS(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) { // Skip comments
        if (line[0] != '%') break; 
    }

    int numRows, numCols, numEntries;
    std::istringstream dimensions(line); // dimensions of vector 
    dimensions >> numRows >> numCols >> numEntries;

    SSSMatrix A;
    A.dval.resize(numRows, 0.0);
    A.row_ptr.resize(numRows + 1, 0);
    std::vector<std::vector<std::pair<int, double>>> temp(numRows);

    int row, col;
    double value;
    while (file >> row >> col >> value) {
        row--; // Adjust for 1-based indexing in MTX format
        col--;
        if (row >= col) { // lower triangular 
            if (row == col) { // diagnol element
                A.dval[row] = value;
            } else { // non-diagnol element
                temp[row].push_back(std::make_pair(col, value));
            }
        }
    }

    // Flatten temp into val and col_idx, and update row_ptr
    int nnz = 0;
    for (int i = 0; i < numRows; ++i) {
        A.row_ptr[i] = nnz;
        for (const auto& pair : temp[i]) {
            A.val.push_back(pair.second);
            A.col_idx.push_back(pair.first);
            nnz++;
        }
    }
    A.row_ptr[numRows] = nnz; // Last element of row_ptr points to the end of val/col_idx
    return A;   
}


inline double SSSmultiplyAndSum(const std::vector<double>& A, const std::vector<double>& x) { 
    // Load the vectors into AVX registers
    __m256d avx_vec1 = _mm256_loadu_pd(A.data());
    __m256d avx_vec2 = _mm256_loadu_pd(x.data());

    // Perform element-wise multiplication
    __m256d avx_result = _mm256_mul_pd(avx_vec1, avx_vec2);

    // Perform horizontal addition to accumulate the sum
    __m256d hsum = _mm256_hadd_pd(avx_result, avx_result);

    // Extract the result from the AVX register
    __m128d low  = _mm256_extractf128_pd(hsum, 0);
    __m128d high = _mm256_extractf128_pd(hsum, 1);
    __m128d sum = _mm_add_pd(low, high);

    // Extract the final sum from the SSE register
    double result;
    _mm_store_sd(&result, sum);
    return result;
}

void SSSprocessRow(const SSSMatrix& A, const std::vector<double>& x, std::vector<double>& y, int startRow, int endRow) {
    constexpr int VectorSize = 4; 
    std::vector<double> tempVecA(VectorSize), tempVecX(VectorSize), tempVecAX(VectorSize);

    for (int current_row = startRow; current_row < endRow; ++current_row) {
        double result = A.dval[current_row] * x[current_row]; // Initialize result with diagonal multiplication 
        
        int j = A.row_ptr[current_row];
        for (int j = A.row_ptr[current_row]; j < A.row_ptr[current_row + 1]; j += VectorSize) { // for each element in the row 
            for (int i = 0; i < VectorSize; ++i) {
                if (j + i < A.row_ptr[current_row + 1]) { // if elements dont exceed the 4 float limit
                    tempVecA[i] = A.val[j + i];
                    int colIdx = A.col_idx[j + i];
                    tempVecX[i] = x[colIdx];
                    // For symmetric update, accumulate directly to corresponding y[colIdx]
                    y[colIdx] += A.val[j + i] * x[current_row];
                } else {
                    // Padding with zeros for the remaining elements
                    tempVecA[i] = 0.0;
                    tempVecX[i] = 0.0;
                    break;
                }
            }
            result += SSSmultiplyAndSum(tempVecA, tempVecX); // update results of current row 
        }
        y[current_row] += result; // Atomic addition is not necessary here since each thread works on distinct rows
    }
}
std::vector<double> SSSspmv(const SSSMatrix& A, const std::vector<double>& x) { // same as CSR 
    const int numThreads = omp_get_max_threads();
    const int numRows = A.row_ptr.size() - 1;
    std::vector<double> y(numRows, 0.0);

    const int rowsPerThread = std::ceil(static_cast<double>(numRows) / numThreads);
    
    #pragma omp parallel for
    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = std::min((i + 1) * rowsPerThread, numRows);
        SSSprocessRow(A, x, y, startRow, endRow);
    }

    return y;
}

