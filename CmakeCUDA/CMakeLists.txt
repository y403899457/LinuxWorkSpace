cmake_minimum_required (VERSION 2.8)

project (CMkCUDA001)

find_package (CUDA QUIET REQUIRED)

aux_source_directory (. DIR_SRC)

cuda_add_executable (CMkCUDA001 CCTest.cu CudaHello.cc)
