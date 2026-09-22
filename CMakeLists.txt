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

set(GPUINFRA_H2D_SIZE_MULTIPLIER 1 CACHE STRING "H2D traffic multiplier")
set(GPUINFRA_D2H_SIZE_MULTIPLIER 1 CACHE STRING "D2H traffic multiplier")
set(GPUINFRA_COMPUTE_SIZE_MULTIPLIER 3 CACHE STRING "Compute work multiplier")

foreach(workload_multiplier
    GPUINFRA_H2D_SIZE_MULTIPLIER
    GPUINFRA_D2H_SIZE_MULTIPLIER
    GPUINFRA_COMPUTE_SIZE_MULTIPLIER
)
    if(NOT "${${workload_multiplier}}" MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "${workload_multiplier} must be a positive integer")
    endif()
endforeach()

option(GPUINFRA_ENABLE_NVTX "Compile NVTX profiling annotations" OFF)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(gpuinfra
    src/DataCache/GpuCacheEntry.cpp
    src/DataCache/GpuCacheManager.cpp
    src/DataCache/GpuDataAccess.cpp
    src/DataCache/GpuResidencyTable.cpp
    src/GpuDiagnostics.cpp
    src/Context/GpuContextManager.cpp
    src/Context/GpuTopology.cpp
    src/StaticData/StaticGpuData.cpp
    src/StaticData/StaticData.cpp
)
target_include_directories(gpuinfra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(gpuinfra_demo_support
    demo/Algo/Cel.cu
    demo/Algo/Mi.cu
    demo/Algo/Sdd.cu
    demo/DummyTask.cpp
    demo/Grape/DummyGraph.cpp
    demo/Grape/NumaExecutor.cpp
    demo/Grape/FrameCpuAtom.cpp
    demo/ParameterRegistry.cpp
)
target_include_directories(gpuinfra_demo_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/demo)
target_link_libraries(gpuinfra_demo_support PUBLIC gpuinfra)
target_compile_definitions(gpuinfra_demo_support
    PUBLIC
        GPUINFRA_H2D_SIZE_MULTIPLIER=${GPUINFRA_H2D_SIZE_MULTIPLIER}
        GPUINFRA_D2H_SIZE_MULTIPLIER=${GPUINFRA_D2H_SIZE_MULTIPLIER}
        GPUINFRA_COMPUTE_SIZE_MULTIPLIER=${GPUINFRA_COMPUTE_SIZE_MULTIPLIER}
)

target_link_libraries(gpuinfra
    PUBLIC
        Threads::Threads
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cufft
)

if(GPUINFRA_ENABLE_NVTX)
    find_path(GPUINFRA_NVTX_INCLUDE_DIR nvtx3/nvtx3.hpp HINTS ${CUDAToolkit_INCLUDE_DIRS} REQUIRED)
    target_include_directories(gpuinfra SYSTEM PUBLIC ${GPUINFRA_NVTX_INCLUDE_DIR})
    target_compile_definitions(gpuinfra PUBLIC GPUINFRA_ENABLE_NVTX=1)
    target_link_libraries(gpuinfra PUBLIC ${CMAKE_DL_LIBS})
endif()

foreach(project_target gpuinfra gpuinfra_demo_support)
    if(MSVC)
        target_compile_options(${project_target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${project_target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:
                -Wall
                -Wextra
                -Wpedantic
                -Wconversion
            >
        )
    endif()

endforeach()

add_executable(gpuinfra_demo demo/main.cpp)
target_link_libraries(gpuinfra_demo PRIVATE gpuinfra_demo_support)

include(CTest)
if(BUILD_TESTING)
    add_executable(gpuinfra_tests
        tests/gpuinfra_tests.cpp
        tests/test_support.cpp
        tests/cache_tests.cpp
        tests/static_data_tests.cpp
        tests/diagnostics_tests.cpp
        tests/graph_tests.cpp
    )
    target_link_libraries(gpuinfra_tests PRIVATE gpuinfra_demo_support)

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
