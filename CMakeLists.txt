cmake_minimum_required (VERSION 3.10)

project(SymphonyQG LANGUAGES CXX)

set(CXX_STANDARD 17)


# AVX512 required
include(CheckCXXSourceRuns)
set(CMAKE_REQUIRED_FLAGS "-march=avx512")

include_directories(${PROJECT_SOURCE_DIR}/symqglib)
include_directories(/usr/include/python3.11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EIGEN "-DEIGEN_DONT_PARALLELIZE")

SET(CMAKE_CXX_FLAGS  "-Wall -Wextra -Ofast -lrt -march=native -mavx512f -mavx512bw -mavx512vl -fpic -fopenmp -ftree-vectorize -fexceptions ${TEMPLATE_CONSTANTS} ${EIGEN}")

add_subdirectory(test)
