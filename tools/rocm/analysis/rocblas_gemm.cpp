// BF16 TN GEMM through rocBLAS (hipBLASLt backend on gfx950) without torch, so
// rocprofv3 can wrap it (rocprofv3 hangs on torch processes in the sglang ROCm
// 10 image). C[M,N] = A[M,K] @ Bt[N,K]^T, all row-major bf16, fp32 accumulate.
// Usage: rocblas_gemm M N K [iters]
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(x)                                                               \
  do {                                                                         \
    auto s = (x);                                                              \
    if (s != 0) {                                                              \
      fprintf(stderr, "%s failed: %d at %s:%d\n", #x, (int)s, __FILE__,        \
              __LINE__);                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
  int iters = argc > 4 ? atoi(argv[4]) : 10;
  std::vector<uint16_t> host((size_t)std::max(M, N) * K);
  std::mt19937 rng(0);
  std::normal_distribution<float> dist;
  for (auto &v : host) {
    float f = dist(rng);
    v = (uint16_t)(*reinterpret_cast<uint32_t *>(&f) >> 16);
  }
  void *a, *bt, *c;
  CHECK(hipMalloc(&a, (size_t)M * K * 2));
  CHECK(hipMalloc(&bt, (size_t)N * K * 2));
  CHECK(hipMalloc(&c, (size_t)M * N * 2));
  CHECK(hipMemcpy(a, host.data(), (size_t)M * K * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(bt, host.data(), (size_t)N * K * 2, hipMemcpyHostToDevice));
  rocblas_handle h;
  CHECK(rocblas_create_handle(&h));
  float alpha = 1.f, beta = 0.f;
  // column-major view: C^T (N x M, ld N) = (Bt row-major viewed as K x N)^T *
  // (A viewed as K x M)
  auto gemm = [&]() {
    return rocblas_gemm_ex(
        h, rocblas_operation_transpose, rocblas_operation_none, N, M, K, &alpha,
        bt, rocblas_datatype_bf16_r, K, a, rocblas_datatype_bf16_r, K, &beta, c,
        rocblas_datatype_bf16_r, N, c, rocblas_datatype_bf16_r, N,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
  };
  for (int i = 0; i < 3; ++i)
    CHECK(gemm());
  CHECK(hipDeviceSynchronize());
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  hipEventRecord(e0);
  for (int i = 0; i < iters; ++i)
    CHECK(gemm());
  hipEventRecord(e1);
  CHECK(hipEventSynchronize(e1));
  float ms;
  hipEventElapsedTime(&ms, e0, e1);
  ms /= iters;
  printf("[rocblas] %dx%dx%d: %.3f ms %.0f TFLOPS\n", M, N, K, ms,
         2.0 * M * N * K / ms / 1e9);
  return 0;
}
