#include <vector>
#include <iostream>
#include "spmv/SSS.h"
#include "spmv/CSR.h"

#include <chrono>
#include "spmv/csr_formatter.h"

int main() {
    CSRMatrix A;
    CSR sim = assemble_simetric_csr_matrix("data/nd3k.mtx");
    A.col_idx = sim.col_ind;
    A.row_ptr = sim.row_ptr;
    A.val = sim.val;
    SSSMatrix S = readMTXtoSSS("data/nd3k.mtx") ;

    // Example input vector x
    std::vector<double> x(A.row_ptr.size(), 1.0); // Initialize result 

    auto start = std::chrono::system_clock::now();
    std::vector<double> y = SSSspmv(S, x);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    
    // Perform SpMV
    /*
    auto start = std::chrono::system_clock::now();
    std::vector<double> y = spmv(A, x);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    
    */
    // Output test time 
    std::cout << "Execution time: " << elapsed_seconds.count() << " microseconds" << std::endl;
    
    return 0;
}