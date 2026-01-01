cmake_minimum_required(VERSION 3.16.2)
set(CMAKE_CUDA_COMPILER
    /usr/local/cuda/bin/nvcc
)

project(sam3_trt LANGUAGES CXX CUDA)

add_compile_options(-std=c++17)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)


find_package(
    PkgConfig
    REQUIRED
)

pkg_check_modules(OpenCV REQUIRED opencv4)

find_package(
    CUDA
    REQUIRED
)


include_directories(
    include
    SYSTEM
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_ROOT_DIR}/include
)

add_library(sam3_trt SHARED 
    src/sam3/sam3_trt/sam3.cu
    src/sam3/sam3_trt/prepost.cu
    )

set_target_properties(sam3_trt PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "native")

target_link_libraries(
    sam3_trt PUBLIC ${CUDA_LIBRARIES} nvinfer nvonnxparser stdc++fs
)

add_executable(sam3_pcs_app 
    src/sam3/sam3_apps/sam3_pcs_app.cpp
    )

set_target_properties(sam3_pcs_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "native")


target_link_libraries(
    sam3_pcs_app
    PUBLIC sam3_trt
           ${OpenCV_LIBRARIES}
           pthread
           stdc++fs
)