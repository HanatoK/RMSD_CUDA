cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
include(CheckIncludeFile)
project(rmsd_cuda LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(cusolver_bin "${CMAKE_PROJECT_NAME}_cusolver")

add_executable(${cusolver_bin} main.cpp rmsd_cuda.h rmsd_cuda.cu rmsd_cuda_kernel.cu rmsd_cuda_kernel.cuh)
target_link_libraries(${cusolver_bin} CUDA::cusolver CUDA::cudart CUDA::nvtx3 fmt)
set_property(TARGET ${cusolver_bin} PROPERTY LANGUAGE CUDA)

set(nr_bin "${CMAKE_PROJECT_NAME}_nr")
add_executable(${nr_bin} main.cpp rmsd_cuda.h rmsd_cuda.cu rmsd_cuda_kernel.cu rmsd_cuda_kernel.cuh)
target_compile_definitions(${nr_bin} PRIVATE USE_NR)
target_link_libraries(${nr_bin} CUDA::cusolver CUDA::cudart CUDA::nvtx3 fmt)
set_property(TARGET ${nr_bin} PROPERTY LANGUAGE CUDA)

set(nr_graph_bin "${CMAKE_PROJECT_NAME}_nr_graph")
add_executable(${nr_graph_bin} main.cpp rmsd_cuda.h rmsd_cuda.cu rmsd_cuda_kernel.cu rmsd_cuda_kernel.cuh)
target_compile_definitions(${nr_graph_bin} PRIVATE USE_NR USE_CUDA_GRAPH)
target_link_libraries(${nr_graph_bin} CUDA::cusolver CUDA::cudart CUDA::nvtx3 fmt)
set_property(TARGET ${nr_graph_bin} PROPERTY LANGUAGE CUDA)
