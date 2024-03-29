cmake_minimum_required(VERSION 3.16)
project(SparseMatrixVectorMultiplication)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(spmv_include_dir
    "${CMAKE_CURRENT_LIST_DIR}/include"
)
# Add executable target
add_executable(spmv spmv.cpp)
target_include_directories(spmv PUBLIC ${spmv_include_dir})

