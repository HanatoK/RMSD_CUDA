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

add_executable(${CMAKE_PROJECT_NAME} main.cpp rmsd_cuda.h rmsd_cuda.cu rmsd_cuda_kernel.cu rmsd_cuda_kernel.cuh)
target_link_libraries(${CMAKE_PROJECT_NAME} CUDA::cusolver CUDA::cudart CUDA::nvtx3 fmt)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY LANGUAGE CUDA)
