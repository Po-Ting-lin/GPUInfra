#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${project_dir}/build"
cuda_architectures="${CUDA_ARCHITECTURES:-86}"

cmake \
    -S "${project_dir}" \
    -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
    "$@"

cmake --build "${build_dir}" --parallel
