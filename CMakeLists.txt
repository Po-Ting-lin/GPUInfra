cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA GPU architectures")
endif()

project(GPUInfra VERSION 1.0.0 LANGUAGES CXX)

option(GPUINFRA_BUILD_DEMO "Build CUDA demo and its integration tests" ON)
if(GPUINFRA_BUILD_DEMO)
    enable_language(CUDA)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
        message(FATAL_ERROR "The demo requires CUDA C++17 (CUDA 11.0+). Use -DGPUINFRA_BUILD_DEMO=OFF to build the C++17 host-only core with an older toolkit.")
    endif()
endif()

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
find_package(CUDAToolkit 10.1 REQUIRED)

# cuFFT is versioned independently of CUDA. Test enum declarations, not macros
# or toolkit version numbers; enum constants cannot be tested with #ifdef.
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
function(gpuinfra_check_cufft_errors)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    foreach(cufft_error MISSING_DEPENDENCY NVRTC_FAILURE NVJITLINK_FAILURE NVSHMEM_FAILURE)
        check_cxx_source_compiles("#include <cufft.h>
    int main() { return static_cast<int>(CUFFT_${cufft_error}); }"
            GPUINFRA_HAS_CUFFT_${cufft_error})
    endforeach()
    cmake_pop_check_state()
endfunction()
gpuinfra_check_cufft_errors()

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
target_compile_features(gpuinfra PUBLIC cxx_std_17)
foreach(cufft_error MISSING_DEPENDENCY NVRTC_FAILURE NVJITLINK_FAILURE NVSHMEM_FAILURE)
    target_compile_definitions(gpuinfra PRIVATE GPUINFRA_HAS_CUFFT_${cufft_error}=$<BOOL:${GPUINFRA_HAS_CUFFT_${cufft_error}}>)
endforeach()

if(GPUINFRA_BUILD_DEMO)
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

endif()

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

set(gpuinfra_warning_targets gpuinfra)
if(GPUINFRA_BUILD_DEMO)
    list(APPEND gpuinfra_warning_targets gpuinfra_demo_support)
endif()
foreach(project_target IN LISTS gpuinfra_warning_targets)
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

if(GPUINFRA_BUILD_DEMO)
    add_executable(gpuinfra_demo demo/main.cpp)
    target_link_libraries(gpuinfra_demo PRIVATE gpuinfra_demo_support)
    # Manual CPU lookup/complete-request measurement; not a timing-gated test.
    add_executable(gpuinfra_cache_policy_benchmark EXCLUDE_FROM_ALL tools/cache_policy_benchmark.cpp)
    target_link_libraries(gpuinfra_cache_policy_benchmark PRIVATE gpuinfra_demo_support)
endif()

include(CTest)
if(BUILD_TESTING AND GPUINFRA_BUILD_DEMO)
    add_executable(gpuinfra_tests
        tests/gpuinfra_tests.cpp
        tests/test_support.cpp
        tests/cache_tests.cpp
        tests/cache_policy_tests.cpp
        tests/result_cache_tests.cpp
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
