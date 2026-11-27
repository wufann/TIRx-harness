#!/bin/bash
# Build the C++ profiling workloads next to prof_gemm.py (runs inside the dev container).
set -eo pipefail
cd "$(dirname "$(readlink -f "$0")")"
hipcc -O2 --offload-arch=gfx950 rocblas_gemm.cpp -o rocblas_gemm -lrocblas
hipcc -O2 hsaco_gemm.cpp -o hsaco_gemm
echo "built $(pwd)/rocblas_gemm $(pwd)/hsaco_gemm"
