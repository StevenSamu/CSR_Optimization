#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>

#include "spmv/SSS.h"
#include "spmv/CSR.h"
#include "spmv/csr_formatter.h"


int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <path_to_matrix_file.mtx> <num_runs> <tolerance>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    const int num_runs = std::stoi(argv[2]);
    const float tolerance = std::stof(argv[3]);
    if (num_runs <= 0 || tolerance <= 0) {
        std::cerr << "Error: num_runs & tolerance must be a positive integer." << std::endl;
        return 1;
    }


    CSRMatrix A;
    CSR sim = assemble_csr_matrix(filename);
    A.col_idx = sim.col_ind;
    A.row_ptr = sim.row_ptr;
    A.val = sim.val;

    // Process user matrix information to console
    std::cout << "\n=== Matrix Information ===" << std::endl;
    std::cout << std::left << std::fixed << std::setprecision(2);
    std::cout << std::setw(std_width) << "Matrix size:"        << sim.R << "x" << sim.C << std::endl;
    std::cout << std::setw(std_width) << "Non-zero elements:"  << sim.val.size() << std::endl;
    std::cout << std::setw(std_width) << "Total elements:"     << static_cast<long long>(sim.R) * sim.C << std::endl;
    std::cout << std::setw(std_width) << "Sparsity (%):"       << (1.0 - (double)sim.val.size() / ((long long)sim.R * sim.C)) * 100 << "%" << std::endl;

    std::vector<double> x(sim.C, 1.0);
    std::cout << "\n=== Vector Info ===" << std::endl;
    std::cout << std::setw(std_width) << "Input vector size:"  << x.size() << std::endl;
    std::cout << std::setw(std_width) << "Input tolerance:"  << std::setprecision(6) << tolerance << std::endl;

    // Naive CSR version
    std::cout << "\n=== Naive/Benchmark CSR ===" << std::endl;
    std::vector<double> y_naive;
    double naive_time = 0.0;
    for (int run = 0; run < num_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        y_naive = benchmark_spmv(A, x);
        auto end = std::chrono::high_resolution_clock::now();
        naive_time += std::chrono::duration<double>(end - start).count();
    }
    naive_time /= num_runs;
    std::cout << std::setw(std_width) << "Benchmark Status:" << "Completed" << std::endl;

    std::cout << "\n=== Testing Different Optimizations ===" << std::endl;

    // simple pointer optimization
    std::cout << "\n1. Simple pointer optimization:" << std::endl;
    double simple_opt_time = 0.0;
    std::vector<double> y_simple;
    for (int run = 0; run < num_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        y_simple = spmv_simple_opt(A, x);
        auto end = std::chrono::high_resolution_clock::now();
        simple_opt_time += std::chrono::duration<double>(end - start).count();
    }
    simple_opt_time /= num_runs;
    verify_results(y_naive, y_simple, tolerance);
    std::cout << std::setw(std_width) << "Speedup:" << naive_time / simple_opt_time << "x" << std::endl;

    // preprocessed indicies
    std::cout << "\n2. Preprocessed indices:" << std::endl;
    double preprocess_time = 0.0;
    std::vector<double> y_preprocess;
    for (int run = 0; run < num_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        y_preprocess = spmv_preprocess(A, x);
        auto end = std::chrono::high_resolution_clock::now();
        preprocess_time += std::chrono::duration<double>(end - start).count();
    }
    preprocess_time /= num_runs;
    verify_results(y_naive, y_preprocess, tolerance);
    std::cout << std::setw(std_width) << "Speedup:" << naive_time / preprocess_time << "x" << std::endl;

    // selective parallel
    std::cout << "\n3. Selective parallel:" << std::endl;
    double parallel_time = 0.0;
    std::vector<double> y_parallel;
    for (int run = 0; run < num_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        y_parallel = spmv_selective_parallel(A, x);
        auto end = std::chrono::high_resolution_clock::now();
        parallel_time += std::chrono::duration<double>(end - start).count();
    }
    parallel_time /= num_runs;
    verify_results(y_naive, y_parallel, tolerance);
    std::cout << std::setw(std_width) << "Speedup:" << naive_time / parallel_time << "x" << std::endl;


    std::cout << "\n4. SIMD rows (4 at a time):" << std::endl;
    double simd_time = 0.0;
    std::vector<double> y_simd;
    for (int run = 0; run < num_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        y_simd = spmv_simd_rows(A, x);
        auto end = std::chrono::high_resolution_clock::now();
        simd_time += std::chrono::duration<double>(end - start).count();
    }
    simd_time /= num_runs;
    verify_results(y_naive, y_simd, tolerance);
    std::cout << std::setw(std_width) << "Speedup: " << naive_time / simd_time << "x" << std::endl;

    // Final results summary
    std::cout << "\n=== FINAL TIMING RESULTS (average of " << num_runs << " runs) ===" << std::endl;

    std::cout << std::left << std::setw(std_width) << "Version"
            << std::right << std::setw(std_width) << "Avg Time (s)"
            << std::setw(std_width) << "Speedup" << std::endl;

    std::cout << std::string(76, '-') << std::endl;

    std::cout << std::left << std::setw(std_width) << "Naive sparse"
            << std::right << std::setw(std_width) << std::fixed << std::setprecision(9) << naive_time
            << std::setw(std_width) << std::fixed << std::setprecision(2) <<  naive_time / naive_time << "x" << std::endl;

    std::cout << std::left << std::setw(std_width) << "Simple optimization"
            << std::right << std::setw(std_width) << std::fixed << std::setprecision(9) << simple_opt_time
            << std::setw(std_width) << std::fixed << std::setprecision(2) << naive_time / simple_opt_time << "x" << std::endl;

    std::cout << std::left << std::setw(std_width) << "Preprocessed"
            << std::right << std::setw(std_width) << std::fixed << std::setprecision(9) << preprocess_time
            << std::setw(std_width) << std::fixed << std::setprecision(2) << naive_time / preprocess_time << "x" << std::endl;

    std::cout << std::left << std::setw(std_width) << "Selective parallel"
            << std::right << std::setw(std_width) << std::fixed << std::setprecision(9) << parallel_time
            << std::setw(std_width) << std::fixed << std::setprecision(2) << naive_time / parallel_time << "x" << std::endl;

    std::cout << std::left << std::setw(std_width) << "SIMD Rows"
            << std::right << std::setw(std_width) << std::fixed << std::setprecision(9) << simd_time
            << std::setw(std_width) << std::fixed << std::setprecision(2) << naive_time / simd_time << "x" << std::endl;
    

    return 0;
}
