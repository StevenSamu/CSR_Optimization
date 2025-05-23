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

std::vector<double> benchmark_spmv(const CSRMatrix& A, const std::vector<double>& x) {
    const int numRows = A.row_ptr.size() - 1;
    std::vector<double> y(numRows, 0.0);

    // This is the baseline - no optimizations at all
    for (int i = 0; i < numRows; ++i) {
        int row_start = A.row_ptr[i] - 1;      // Convert 1-based to 0-based
        int row_end = A.row_ptr[i + 1] - 1;    // Convert 1-based to 0-based
        
        double sum = 0.0;
        for (int j = row_start; j < row_end; ++j) {
            sum += A.val[j] * x[A.col_idx[j] - 1];  // Convert 1-based to 0-based
        }
        y[i] = sum;
    }

    return y;
}

// First, let's try the simplest possible optimization - just remove redundant operations
std::vector<double> spmv_simple_opt(const CSRMatrix& A, const std::vector<double>& x) {
    const int numRows = A.row_ptr.size() - 1;
    std::vector<double> y(numRows, 0.0);
    
    // Cache pointers to avoid vector overhead
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

// Convert indices once at the beginning instead of in every access
std::vector<double> spmv_preprocess(const CSRMatrix& A, const std::vector<double>& x) {
    const int numRows = A.row_ptr.size() - 1;
    std::vector<double> y(numRows, 0.0);
    
    // Pre-compute 0-based indices
    static std::vector<int> row_ptr_0based;
    static std::vector<int> col_idx_0based;
    static bool preprocessed = false;
    
    if (!preprocessed) {
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

// Try minimal threading - only if it really helps
std::vector<double> spmv_selective_parallel(const CSRMatrix& A, const std::vector<double>& x) {
    const int numRows = A.row_ptr.size() - 1;
    std::vector<double> y(numRows, 0.0);
    
    // Only use parallelism if we have enough work
    const int min_rows_per_thread = 100;
    const int num_threads = std::min(omp_get_max_threads(), 
                                     std::max(1, numRows / min_rows_per_thread));
    
    if (num_threads > 1) {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int rows_per_thread = (numRows + num_threads - 1) / num_threads;
            int start = tid * rows_per_thread;
            int end = std::min(start + rows_per_thread, numRows);
            
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

// verify bencmark data to computed data of specific versions out vec
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

// Function to analyze the matrix structure
void analyze_matrix(const CSRMatrix& A) {
    const int numRows = A.row_ptr.size() - 1;
    int empty_rows = 0;
    int max_row_nnz = 0;
    int min_row_nnz = INT_MAX;
    double avg_row_nnz = 0.0;
    
    std::vector<int> nnz_histogram(11, 0);  // 0, 1-10, 11-20, ..., 91-100, >100
    
    for (int i = 0; i < numRows; ++i) {
        int nnz = A.row_ptr[i + 1] - A.row_ptr[i];
        if (nnz == 0) empty_rows++;
        max_row_nnz = std::max(max_row_nnz, nnz);
        min_row_nnz = std::min(min_row_nnz, nnz);
        avg_row_nnz += nnz;
        
        int bucket = std::min(nnz / 10, 10);
        nnz_histogram[bucket]++;
    }
    
    avg_row_nnz /= numRows;
    
    std::cout << "\n=== Matrix Structure Analysis ===" << std::endl;
    std::cout << "Empty rows: " << empty_rows << std::endl;
    std::cout << "Max non-zeros in a row: " << max_row_nnz << std::endl;
    std::cout << "Min non-zeros in a row: " << min_row_nnz << std::endl;
    std::cout << "Avg non-zeros per row: " << avg_row_nnz << std::endl;
    
    std::cout << "\nRow density histogram:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "  " << (i*10) << "-" << ((i+1)*10-1) << " nnz: " << nnz_histogram[i] << " rows" << std::endl;
    }
    std::cout << "  >100 nnz: " << nnz_histogram[10] << " rows" << std::endl;
}