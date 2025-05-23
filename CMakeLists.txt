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
    # GCC or Clang - Enhanced SIMD support
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -funroll-loops -ffast-math")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ftree-vectorize -fomit-frame-pointer")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno -ffinite-math-only")
    
    # Explicitly enable SIMD instruction sets for compatibility
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx -mavx2 -mfma -msse4.2")
    
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

# REMOVED: target_compile_options(spmv PRIVATE -mavx)
# This was conflicting with -march=native and insufficient for AVX2/FMA

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(spmv PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add debug build flags for development
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -mavx -mavx2 -mfma")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /W3 /arch:AVX2")
endif()

# Print optimization flags (for debugging)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CXX flags release: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CXX flags debug: ${CMAKE_CXX_FLAGS_DEBUG}")

# Optional: Check if compiler supports required features
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
    message(STATUS "✓ Compiler supports AVX2 and FMA instructions")
else()
    message(WARNING "⚠ Compiler may not support required SIMD instructions")
endif()