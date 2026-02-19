set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(indexing indexing.cpp)
add_executable(bench_sift1m bench_sift1m.cpp)
add_executable(build_index build_index.cpp)

# Portable AVX-512 target (no -march=native, for cross-machine deployment)
add_executable(bench_sift1m_avx512 bench_sift1m.cpp)
set_target_properties(bench_sift1m_avx512 PROPERTIES
    COMPILE_OPTIONS ""
    COMPILE_FLAGS "-std=c++17 -Ofast -mavx512f -mavx512bw -mavx512vl -mfma -fopenmp -ftree-vectorize -DEIGEN_DONT_PARALLELIZE"
    LINK_FLAGS "-fopenmp")