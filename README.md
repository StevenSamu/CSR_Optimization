# CSR_Optimization

This is a repository used to track and build an optimized imeplementation of a CSR (compressed sparse row) sparse matrix-vector multiplication system that utilizes multi-threading, vectorization and other performance engineering techniques. 

Performance Engineering Techniques: 

Mulit-Threading
- Utilize Multi-Threading to allow fast processing of individual rows in the sparse matrix. This will save significant compute time. Utilizing a sempahor/mutex based system (TBD), I will incorperate a way for processes to aknowledge if there is avaiable threads to work with. 

- Initally, I went to use the standard thread library availible in C++, but with research I found evidence that implementing OpenMP instead would provide better performance as it is more geared towards large batch parallelization, which is exactly what I am implementing. Doing this brought runtime down for a 494x494 matrix from 0.0013488 microseconds to  1.44e-05 microseconds, which is significant. 


Vectorization (SIMD)
- Utilized SIMD (Single Instreuction, Multiple Data) to perform the multiplication and addition of the SPMV elements efficiently, removing the need for scalar processing. This allow for higher throughput. This puts the time complexity of solving a row from O(non_zero_row_elements) to O(non_zero_row_elements/SIMD_size) and in our case SIMD_size would be 4, as this is the max amount of double values that the 256 bit AVX2 can operate with at once. 

SSS Format for Symmetric Matrices
- Doing reasearch into the topic, for symtetric Matrices, it doesnt make sense to use CSR formatting. The CSR format stores up to half of redunant data for Symm. matrix's thus needing to use specialized formats. This is where SSS comes in. I plan to use SSS formatting to reduce the number of computations and data processing required, reducing run time for symmetric matrices. 

Inlining
- Using inlining for the multiplyAndSum is a small yet impactful portion of the CSR and SSS code I wrote, and using it inline allowed compile time efficency as it would be directly placed within the body of the code during compile time. 

References:

- https://github.com/notini/csr-formatter, utilized this formatter from .mtx to CSR format
- https://kkourt.io/papers/ipdps13.pdf, referenced this paper for symetric matrices

Old Function Versions:
```
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
```
```
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
```

```
std::vector<double> spmv(const CSRMatrix& A, const std::vector<double>& x) {
    const unsigned int numThreads = std::thread::hardware_concurrency(); // number of possible threads avaible from my machine, 12 in my case
    const int numRows = A.row_ptr.size() - 1;  // Number of rows in the matrix
    std::vector<double> y(numRows, 0.0); // Initialize result vector y

    const int rowsPerThread = std::ceil(static_cast<double>(numRows) / numThreads);
    

    // Launch a thread for each range of rows
    std::vector<std::thread> threads; // vector of threads
    for (int i = 0; i < numThreads; i++) {
        int startRow = i * rowsPerThread; 
        int endRow = std::min((i + 1) * rowsPerThread, numRows); 
        threads.emplace_back(processRow, std::ref(A), std::ref(x), std::ref(y), startRow, endRow);
        if (endRow == numRows) { // end row hit dont need more threads, for smaller matricies
            break; 
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }
    return y;
}
```

```
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

```

```
inline double multiplyAndSum(const std::vector<double>& A, const std::vector<double>& x) {
    // Load the vectors into AVX registers
    __m256d avx_vec1 = _mm256_loadu_pd(&A[0]); // Assuming A and x have at least 4 elements
    __m256d avx_vec2 = _mm256_loadu_pd(&x[0]);

    // Perform element-wise multiplication
    __m256d avx_result = _mm256_mul_pd(avx_vec1, avx_vec2);

    // Perform horizontal addition to accumulate the sum
    __m256d hsum = _mm256_hadd_pd(avx_result, avx_result);

    // Extract the result from the AVX register
    __m128d low = _mm256_extractf128_pd(hsum, 0);
    __m128d high = _mm256_extractf128_pd(hsum, 1);
    __m128d sum = _mm_add_pd(low, high);

    // Extract the final sum from the SSE register
    double result;
    _mm_store_sd(&result, sum);
    return result;
}
```
