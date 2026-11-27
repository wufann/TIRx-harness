// Launch a tvm-tirx BF16 GEMM code object (exported with export_hsaco.py) from
// C++, so rocprofv3 can profile it: rocprofv3 hangs on Python processes that
// use HIP in the sglang ROCm 10 image. Kernel ABI (TN, bf16 out): (int16*
// A[M*K], int16* Bt[N*K], int32* C[M*N/2]), grid (N/BN, M/BM), block NT. Usage:
// hsaco_gemm FILE.hsaco KERNEL M N K BM BN NT [iters]
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>

#define CHECK(x)                                                               \
  do {                                                                         \
    auto s = (x);                                                              \
    if (s != hipSuccess) {                                                     \
      fprintf(stderr, "%s failed: %s at %s:%d\n", #x, hipGetErrorString(s),    \
              __FILE__, __LINE__);                                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  const char *path = argv[1];
  const char *kname = argv[2];
  int M = atoi(argv[3]), N = atoi(argv[4]), K = atoi(argv[5]);
  int BM = atoi(argv[6]), BN = atoi(argv[7]), NT = atoi(argv[8]);
  int iters = argc > 9 ? atoi(argv[9]) : 10;
  std::ifstream f(path, std::ios::binary);
  std::vector<char> image((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
  hipModule_t mod;
  hipFunction_t fn;
  CHECK(hipModuleLoadData(&mod, image.data()));
  CHECK(hipModuleGetFunction(&fn, mod, kname));
  std::vector<uint16_t> host((size_t)std::max(M, N) * K);
  std::mt19937 rng(0);
  std::normal_distribution<float> dist;
  for (auto &v : host) {
    float x = dist(rng);
    v = (uint16_t)(*reinterpret_cast<uint32_t *>(&x) >> 16);
  }
  void *a, *bt, *c;
  CHECK(hipMalloc(&a, (size_t)M * K * 2));
  CHECK(hipMalloc(&bt, (size_t)N * K * 2));
  CHECK(hipMalloc(&c, (size_t)M * N * 2));
  CHECK(hipMemcpy(a, host.data(), (size_t)M * K * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(bt, host.data(), (size_t)N * K * 2, hipMemcpyHostToDevice));
  struct {
    void *a;
    void *bt;
    void *c;
  } args{a, bt, c};
  size_t size = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  auto launch = [&]() {
    return hipModuleLaunchKernel(fn, N / BN, M / BM, 1, NT, 1, 1, 0, nullptr,
                                 nullptr, config);
  };
  for (int i = 0; i < 3; ++i)
    CHECK(launch());
  CHECK(hipDeviceSynchronize());
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  CHECK(hipEventRecord(e0));
  for (int i = 0; i < iters; ++i)
    CHECK(launch());
  CHECK(hipEventRecord(e1));
  CHECK(hipEventSynchronize(e1));
  float ms;
  CHECK(hipEventElapsedTime(&ms, e0, e1));
  ms /= iters;
  printf("[hsaco] %s %dx%dx%d: %.3f ms %.0f TFLOPS\n", kname, M, N, K, ms,
         2.0 * M * N * K / ms / 1e9);
  return 0;
}
