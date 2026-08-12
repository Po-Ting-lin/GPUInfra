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
    src/DummyTask.cpp
    src/GraphStuff.cpp
    src/GraphTypes.cpp
    src/GpuContextManager.cpp
    src/Mi.cu
    src/ParameterRegistry.cpp
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
    add_executable(gpuinfra_tests tests/gpuinfra_tests.cpp)
    target_link_libraries(gpuinfra_tests PRIVATE gpuinfra)

    add_test(NAME gpuinfra_protocol_tests COMMAND gpuinfra_tests)
    add_test(NAME gpuinfra_demo_runs COMMAND gpuinfra_demo)
    add_test(NAME gpuinfra_demo_interleaved_runs COMMAND gpuinfra_demo 2 1 interleaved)
    add_test(NAME gpuinfra_demo_min_factor_runs COMMAND gpuinfra_demo 2 1 batched 16)
    add_test(NAME gpuinfra_demo_max_factor_runs COMMAND gpuinfra_demo 1 0 batched 256)
    add_test(NAME gpuinfra_demo_rejects_unaligned_factor COMMAND gpuinfra_demo 2 1 batched 17)
    set_tests_properties(gpuinfra_demo_rejects_unaligned_factor PROPERTIES WILL_FAIL TRUE)
    add_test(NAME gpuinfra_demo_rejects_oversized_factor COMMAND gpuinfra_demo 2 1 batched 272)
    set_tests_properties(gpuinfra_demo_rejects_oversized_factor PROPERTIES WILL_FAIL TRUE)
endif()
