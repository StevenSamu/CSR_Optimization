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

void print_matrix(const CSRMatrix& A) {
    // Extract matrix dimensions
    int numRows = A.row_ptr.size() - 1;
    int numCols = 0;
    for (int i = 0; i < numRows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            numCols = std::max(numCols, A.col_idx[j] + 1);
        }
    }

    // Initialize matrix with zeros
    std::vector<std::vector<double>> matrix(numRows, std::vector<double>(numCols, 0.0));

    // Fill in non-zero values
    for (int i = 0; i < numRows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            matrix[i][A.col_idx[j]] = A.val[j];
        }
    }

    // Print the matrix
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

double multiplyAndSum(const std::vector<double>& A, const std::vector<double>& x) {
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

void processRow(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y, int startRow, int endRow) { //performs matrix multiplication on a single row of the CSR matrix
    // current row is not only the row we are working with, but also the index in y we want to insert the final value
    constexpr int VectorSize = 4; // AVX2 can work with 4 double values each iteration
     // Iterate over the rows in the specified range and perform SpMV    std::cout << "HELLO TEST" << std::endl; 
    std::vector<double> tempVecA (VectorSize);
    std::vector<double> tempVecX (VectorSize);
    for (int current_row = startRow; current_row < endRow; current_row++) { // Less then endRow as endRow will be start row for next thread
        double result = 0.0;

        for (int j = A.row_ptr[current_row]; j < A.row_ptr[current_row + 1]; j += VectorSize) {  // Loop over non-zero elements in the current row
            for (int i = 0; i < VectorSize; i++) {
                if (j + i < A.row_ptr[current_row + 1]) {
                    tempVecA[i] = A.val[j + i];
                    tempVecX[i] = x[A.col_idx[j + i]];
                } else {
                    // Padding with zeros for the remaining elements
                    tempVecA[i] = 0.0;
                    tempVecX[i] = 0.0;
                }
            }
            result += multiplyAndSum(tempVecA, tempVecX);

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
        int endRow = std::min((i + 1) * rowsPerThread, numRows); 
        processRow(A, x, y, startRow, endRow);
    }

    return y;
}

// Function to perform SpMV using CSR format
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


 