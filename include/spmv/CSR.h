#include <string>
#include <vector>
#include <iostream> 
#include <immintrin.h> // Include header for SIMD instructions (AVX)
#include <thread>
#include <omp.h>

struct CSRMatrix {
    std::vector<double> val;       // Values array
    std::vector<int> row_ptr;      // Row pointer array
    std::vector<int> col_idx;      // Column indices array
};

inline double multiplyAndSum(const double* A, const double* x, int size) {
    double result = 0.0;
    int i = 0;
    for (; i < size - 4; i += 4) { // process elements in row as batches of 4, as thats what SIMD allows for double's
        __m256d vecA = _mm256_loadu_pd(&A[i]);
        __m256d vecX = _mm256_loadu_pd(&x[i]);
        __m256d vecResult = _mm256_mul_pd(vecA, vecX);
        __m256d hsum = _mm256_hadd_pd(vecResult, vecResult);
        __m128d low = _mm256_extractf128_pd(hsum, 0);
        __m128d high = _mm256_extractf128_pd(hsum, 1);
        __m128d sum128 = _mm_add_pd(low, high);
        double sumScalar;
        _mm_store_sd(&sumScalar, sum128);
        result += sumScalar;
    }
    for (; i < size; ++i) {  // Process remaining elements in row 
        result += A[i] * x[i];
    }
    return result;
}

void processRow(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y, int startRow, int endRow) {
    constexpr int VectorSize = 4; 

    for (int current_row = startRow; current_row < endRow; ++current_row) {
        double result = 0.0;

        int j = A.row_ptr[current_row];
        int elements = A.row_ptr[current_row + 1] - j;
        if (elements > VectorSize) { // if there enough elements for SIMD 
            result = multiplyAndSum(&A.val[j], &x[A.col_idx[j]], elements); // Directly pass pointers to the segments of the original vectors

        } else {
            for (; j < A.row_ptr[current_row + 1]; ++j) { // Handle fewer elements directly is faster then computing using SIMD and padding with zeros 
                result += A.val[j] * x[A.col_idx[j]];
            }
        }
        y[current_row] = result;
    }
}

std::vector<double> spmv(const CSRMatrix& A, const std::vector<double>& x) {
    const int numThreads = omp_get_max_threads(); // Get the number of available threads
    const int numRows = A.row_ptr.size() - 1;  // Number of rows in the matrix
    std::vector<double> y(numRows, 0.0); // Initialize result vector y

    const int rowsPerThread = std::ceil(static_cast<double>(numRows) / numThreads);
    
    #pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < numThreads; i++) {
        int startRow = i * rowsPerThread; 
        int endRow = std::min((i + 1) * rowsPerThread, numRows); // grab either the last row or row below it 
        processRow(A, x, y, startRow, endRow);
    }

    return y;
}

std::vector<double> benchmark_spmv(const CSRMatrix& A, const std::vector<double>& x) {
    const int numRows = A.row_ptr.size();  // Number of rows in the matrix
    std::vector<double> y(numRows, 0.0); // Initialize result vector y

    // Iterate over each row of the matrix
    for (int i = 0; i < numRows; ++i) {
        // Loop over non-zero elements in the current row
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            // Multiply the non-zero element with the corresponding element in x
            y[i] += A.val[j] * x[A.col_idx[j]];
        }
    }

    return y;
}


 