cmake_minimum_required(VERSION 3.16)
project(SparseMatrixVectorMultiplication)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Maximum optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # GCC or Clang
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -funroll-loops -ffast-math")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ftree-vectorize -fomit-frame-pointer")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno -ffinite-math-only")
    
    # Enable link-time optimization
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    
elseif(MSVC)
    # Microsoft Visual C++
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /Oi /Ot /Oy /GT /GL /fp:fast")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

set(spmv_include_dir
    "${CMAKE_CURRENT_LIST_DIR}/include"
)

# Add executable target
add_executable(spmv src/spmv.cpp)
target_include_directories(spmv PUBLIC ${spmv_include_dir})

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(spmv PUBLIC OpenMP::OpenMP_CXX)
endif()

# Print optimization flags (for debugging)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CXX flags release: ${CMAKE_CXX_FLAGS_RELEASE}")