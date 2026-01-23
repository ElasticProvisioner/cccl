# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21)

project(cuda_compute_cpp_benchmarks 
  DESCRIPTION "C++ benchmarks for cuda.compute comparison"
  LANGUAGES CXX CUDA
)

# Configure paths to CCCL root (4 levels up from benchmarks/compute)
set(CCCL_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../")
get_filename_component(CCCL_SOURCE_DIR "${CCCL_SOURCE_DIR}" ABSOLUTE)
set(CCCL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_cccl_build")

message(STATUS "CCCL Source: ${CCCL_SOURCE_DIR}")
message(STATUS "CCCL Binary: ${CCCL_BINARY_DIR}")

# Add CCCL CMake modules to path
list(APPEND CMAKE_MODULE_PATH "${CCCL_SOURCE_DIR}/cmake")

# Include CCCL dependency helpers
include("${CCCL_SOURCE_DIR}/cmake/CCCLGetDependencies.cmake")

# Get dependencies using CCCL's functions
cccl_get_nvbench()
cccl_get_cub()
cccl_get_thrust()
cccl_get_libcudacxx()
cccl_get_cudatoolkit()

# GPU architectures (user-configurable)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES not set, using 'native'")
  set(CMAKE_CUDA_ARCHITECTURES native)
else()
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Create benchmark executable
add_executable(nvbench_transform_cpp nvbench_transform_cpp.cu)

# C++/CUDA standards (matching CCCL requirements)
set_target_properties(nvbench_transform_cpp PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Link all dependencies
target_link_libraries(nvbench_transform_cpp
  PRIVATE
    CUB::CUB
    Thrust::Thrust
    libcudacxx::libcudacxx
    nvbench::main  # Provides main() function
)

# Output to bin/ directory (git-ignored)
set_target_properties(nvbench_transform_cpp PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

message(STATUS "Benchmark executable will be output to: ${CMAKE_CURRENT_SOURCE_DIR}/bin/nvbench_transform_cpp")
