#include <string>
#include <vector>
#include <iostream> 
#include <immintrin.h>
#include <thread>
#include <omp.h>
#include <cmath>
#include <numeric>
#include <cstring>
#include <iomanip>

#define std_width 25

struct CSRMatrix {
    std::vector<double> val;       // Values array
    std::vector<int> row_ptr;      // Row pointer array
    std::vector<int> col_idx;      // Column indices array
};

// simple CSR matrix format benchmark to test against
std::vector<double> benchmark_spmv(const CSRMatrix& A, const std::vector<double>& x) {
    const int numRows = A.row_ptr.size() - 1;
    std::vector<double> y(numRows, 0.0);

    for (int i = 0; i < numRows; ++i) {
        int row_start = A.row_ptr[i] - 1;      // Convert 1-based to 0-based for each index, i.e [1,5,9] -> [0,4,8] 
        int row_end = A.row_ptr[i + 1] - 1;    
        
        double sum = 0.0;
        for (int j = row_start; j < row_end; ++j) {
            sum += A.val[j] * x[A.col_idx[j] - 1];  
        }
        y[i] = sum;
    }

    return y;
}

// Simplest possible optimization - remove redundant operations
std::vector<double> spmv_simple_opt(const CSRMatrix& A, const std::vector<double>& x) {
    const int numRows = A.row_ptr.size() - 1;
    std::vector<double> y(numRows, 0.0);
    
    // Cache pointers to avoid vector overhead, store pointer of first array element
    const double* val_ptr = A.val.data();
    const int* col_idx_ptr = A.col_idx.data();
    const int* row_ptr_ptr = A.row_ptr.data();
    const double* x_ptr = x.data();
    double* y_ptr = y.data();

    for (int i = 0; i < numRows; ++i) {
        int row_start = row_ptr_ptr[i] - 1;
        int row_end = row_ptr_ptr[i + 1] - 1;
        
        double sum = 0.0;
        for (int j = row_start; j < row_end; ++j) {
            sum += val_ptr[j] * x_ptr[col_idx_ptr[j] - 1];
        }
        y_ptr[i] = sum;
    }

    return y;
}

/*
Convert indices once at the beginning instead of in every access

My hypothisis on why this sometimes preforms worse then simple case:

- Branch prediction misses due to the if (!preprocessed) check)
- Cache locality doesnt keep the preprocessed variable in register mem calls must be made

*/
std::vector<double> spmv_preprocess(const CSRMatrix& A, const std::vector<double>& x) {    
    const int numRows = A.row_ptr.size() - 1;
    std::vector<double> y(numRows, 0.0);
    
    // Pre-compute 0-based indices, store in static/data segment of mem rather then stack 
    static std::vector<int> row_ptr_0based;
    static std::vector<int> col_idx_0based;
    static bool preprocessed = false;
    
    if (!preprocessed) { // on first use, function computes the indencie difference only once rather then n times per call
        row_ptr_0based.resize(A.row_ptr.size());
        col_idx_0based.resize(A.col_idx.size());
        
        // Convert row pointers to 0-based
        for (size_t i = 0; i < A.row_ptr.size(); ++i) {
            row_ptr_0based[i] = A.row_ptr[i] - 1;
        }
        
        // Convert column indices to 0-based
        for (size_t i = 0; i < A.col_idx.size(); ++i) {
            col_idx_0based[i] = A.col_idx[i] - 1;
        }
        
        preprocessed = true;
    }

    // same as simple optimization by caching the data 
    const double* val_ptr = A.val.data();
    const int* col_idx_ptr = col_idx_0based.data();
    const int* row_ptr_ptr = row_ptr_0based.data();
    const double* x_ptr = x.data();

    for (int i = 0; i < numRows; ++i) {
        int row_start = row_ptr_ptr[i];
        int row_end = row_ptr_ptr[i + 1];
        
        double sum = 0.0;
        for (int j = row_start; j < row_end; ++j) {
            sum += val_ptr[j] * x_ptr[col_idx_ptr[j]];
        }
        y[i] = sum;
    }

    return y;
}

/*
Selective paralleization to process multiplications in parallel 

My hypothosis on why this preforms worse then the simple optimization:

- Parallelization suffers from thread overhead thread initlization and setup
- No inherint load balancing/splitting data optimally between thread
- Suffers from more cache misses since threads share a memory    

*/
std::vector<double> spmv_selective_parallel(const CSRMatrix& A, const std::vector<double>& x) {
    const int numRows = A.row_ptr.size() - 1;
    std::vector<double> y(numRows, 0.0);
    
    // Only use parallelism if there is enough work
    const int min_rows_per_thread = 100;
    // Choose the max between the number of rows to threads and 1, then take the min between that number and max threads on the system. 
    const int num_threads = std::min(omp_get_max_threads(), std::max(1, numRows / min_rows_per_thread));
    if (num_threads > 1) {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int rows_per_thread = (numRows + num_threads - 1) / num_threads;
            int start = tid * rows_per_thread;
            int end = std::min(start + rows_per_thread, numRows); // in case of noneven split of rows per thread
            
            for (int i = start; i < end; ++i) {
                int row_start = A.row_ptr[i] - 1;
                int row_end = A.row_ptr[i + 1] - 1;
                
                double sum = 0.0;
                for (int j = row_start; j < row_end; ++j) {
                    sum += A.val[j] * x[A.col_idx[j] - 1];
                }
                y[i] = sum;
            }
        }
    } else {
        // Just run serially
        return spmv_simple_opt(A, x);
    }
    
    return y;
}

/*
SIMD Row Batch Processing Approach 

My hypothosis on why this preforms worse then the simple optimization:

- Vector operator overhead for bounds checking, when raw pointers are faster
- SIMD overhead v.s. benefits trade-off: irregular mem access kills SIMD, setup costs > computation benefits
- GCC Compiler is very tuned to handle auto vectorization without manual overhead 

*/
std::vector<double> spmv_simd_rows(const CSRMatrix& A, const std::vector<double>& x) {
    const int numRows = A.row_ptr.size() - 1;
    std::vector<double> y(numRows, 0.0);
    
    for (int i = 0; i < numRows; ++i) { // iterate through rows and process data in vecotrized batches
        int row_start = A.row_ptr[i] - 1;
        int row_end = A.row_ptr[i + 1] - 1;
        int row_nnz = row_end - row_start;
        
        __m256d sum_vec = _mm256_setzero_pd();
        int j = row_start;
        
        // 4 elements at a time using SIMD
        for (; j <= row_end - 4; j += 4) {
            // Load 4 matrix values into 256bit AVX registers
            __m256d vals = _mm256_loadu_pd(&A.val[j]);
            
            // Gather 4 corresponding vector values using column indices
            __m256d x_vals = _mm256_set_pd(
                x[A.col_idx[j + 3] - 1],
                x[A.col_idx[j + 2] - 1], 
                x[A.col_idx[j + 1] - 1],
                x[A.col_idx[j] - 1]
            );
            
            // matmul: sum_vec += vals * x_vals -> sum_vec = [sum0, sum1, sum2, sum3]
            sum_vec = _mm256_fmadd_pd(vals, x_vals, sum_vec);

        }
        
        alignas(32) double temp[4]; // Force temp array to start at 32B divisible mem location
        _mm256_store_pd(temp, sum_vec); // Copy SIMD register to array
        double partial_sum = temp[0] + temp[1] + temp[2] + temp[3];
        
        // Handle remaining elements
        for (; j < row_end; ++j) {
            partial_sum += A.val[j] * x[A.col_idx[j] - 1];
        }
        
        y[i] = partial_sum;
    }
    
    return y;
}

// verify benchmark data to computed data of specific versions out vec
bool verify_results(const std::vector<double>& y1, const std::vector<double>& y2, double tolerance) {
    
    if (y1.size() != y2.size()) {
        std::cout << "ERROR: Result vectors have different sizes!" << std::endl;
        return false;
    }
    
    double max_diff = 0.0;
    int diff_count = 0;
    
    for (size_t i = 0; i < y1.size(); ++i) {
        double diff = std::abs(y1[i] - y2[i]);
        if (diff > tolerance) {
            if (diff_count < 5) {  // Print first 5 differences
                std::cout << "Difference at index " << i << ": " 
                        << y1[i] << " vs " << y2[i] << " (diff: " << diff << ")" << std::endl;
            }
            diff_count++;
            max_diff = std::max(max_diff, diff);
        }
    }
    
    if (diff_count > 0) {
        std::cout << "Total differences: " << diff_count << ", Max difference: " << max_diff << std::endl;
        return false;
    }
    std::cout << std::left << std::fixed;
    std::cout << std::setw(std_width)<< "Run Status:" << "results match BM" << std::endl;
    return true;
}