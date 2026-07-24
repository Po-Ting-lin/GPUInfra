cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA GPU architectures")
endif()

project(GPUInfra VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(gpuinfra
    src/Cel.cu
    src/GpuContextManager.cpp
    src/Sdd.cu
)

target_include_directories(gpuinfra
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(gpuinfra
    PUBLIC
        Threads::Threads
        CUDA::cudart
        CUDA::cuda_driver
)

if(MSVC)
    target_compile_options(gpuinfra PRIVATE /W4 /permissive-)
else()
    target_compile_options(gpuinfra PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
        >
    )
endif()

add_executable(gpuinfra_demo src/main.cpp)
target_link_libraries(gpuinfra_demo PRIVATE gpuinfra)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME gpuinfra_demo_runs COMMAND gpuinfra_demo)
endif()
