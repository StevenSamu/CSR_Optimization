#include <vector>
#include <iostream>
#include "spmv/CSR.h"
#include <chrono>
#include "spmv/csr_formatter.h"

int main() {
    CSRMatrix A;
    CSR sim = assemble_simetric_csr_matrix("data/494_bus.mtx");
    A.col_idx = sim.col_ind;
    A.row_ptr = sim.row_ptr;
    A.val = sim.val;

    // Example input vector x
    std::vector<double> x(496, 1.0); // Initialize result 

    // Perform SpMV
 
    auto start = std::chrono::system_clock::now();
    std::vector<double> y = spmv(A, x);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    

    // Output test time 
    std::cout << "Execution time: " << elapsed_seconds.count() << " microseconds" << std::endl;

    for (auto val : y) {
        std::cout << val << " " << std::flush;
    }

    return 0;
}