# cuBLASLt BF16 Square GEMM Results

Benchmark source: `bf16_square_gemm_cublaslt.cu`

Build/run command:

```bash
cd /workspace/avelang/cublaslt
SM=90a ./build_and_run.sh
```

The benchmark evaluates `D = alpha * A * B + beta * C` through
`cublasLtMatmul` with BF16 A/B/C/D matrices and FP32 accumulation. Effective
throughput is computed as `2*M*N*K / avg_time`.

Generated artifacts for this run:

- `artifacts/bf16_square_gemm_cublaslt`
- `artifacts/bf16_square_gemm_cublaslt.sm_90a.ptx`
- `artifacts/bf16_square_gemm_cublaslt.sm_90a.cubin`
- `artifacts/bf16_square_gemm_cublaslt.sm_90a.sass`
- `artifacts/bf16_square_gemm_cublaslt.executable.sass`
- `artifacts/bf16_square_gemm_cublaslt.results.csv`

Note: the saved PTX/SASS correspond to this CUDA benchmark translation unit,
including the BF16 initialization kernel. The GEMM implementation itself is
selected from the NVIDIA cuBLASLt library at runtime.

## Environment

- Date: 2026-06-23
- Host path: `/workspace/avelang/cublaslt`
- CUDA compiler: CUDA 13.0, `nvcc` V13.0.88
- CUDA runtime version: 13000
- cuBLAS version: 13.0.2
- Selected CUDA device: 2
- GPU: NVIDIA H100 PCIe
- Compute capability: 9.0

## Results

| M | N | K | Repeats | Avg time (ms) | Effective TFLOPS | Algo workspace bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1024 | 200 | 0.0087 | 247.19 | 0 |
| 2048 | 2048 | 2048 | 100 | 0.0331 | 519.17 | 0 |
| 4096 | 4096 | 4096 | 50 | 0.2336 | 588.45 | 4 |
| 8192 | 8192 | 8192 | 20 | 1.5882 | 692.32 | 4 |
| 16384 | 16384 | 16384 | 5 | 16.0167 | 549.18 | 4 |

## Captured Output

`artifacts/bf16_square_gemm_cublaslt.results.csv`:

```text
# cuBLASLt BF16 square GEMM benchmark
selected_cuda_device,2
device,NVIDIA H100 PCIe
compute_capability,9.0
cuda_runtime_version,13000
cublas_version,13.0.2
M,N,K,repeats,avg_ms,tflops,algo_workspace_bytes
1024,1024,1024,200,0.0087,247.19,0
2048,2048,2048,100,0.0331,519.17,0
4096,4096,4096,50,0.2336,588.45,4
8192,8192,8192,20,1.5882,692.32,4
16384,16384,16384,5,16.0167,549.18,4
```

## Avelang Results

Benchmark source: `benchmark/gemm/bench_nvidia_wgmma_bf16_square.py`

Test kernel source:
`test/examples/gemm/nvidia/test_gemm_wgmma_bf16_square.py`

Build/run command:

```bash
cd /workspace/avelang
PYTHONPATH=python python benchmark/gemm/bench_nvidia_wgmma_bf16_square.py --sizes 1024 2048 4096 8192 16384
```

The benchmark uses the same CUDA event and CUDAGraph replay timing pattern as
`benchmark/gemm/bench_amdgpu_gemm.py`. Effective throughput is computed as
`2*M*N*K / avg_time`. The repeat counts match the cuBLASLt benchmark table: 200,
100, 50, 20, and 5.

### Avelang Environment

- Date: 2026-06-24
- Host path: `/workspace/avelang`
- Selected CUDA device: 2
- GPU: NVIDIA H100 PCIe
- Kernel: BF16 square GEMM using Hopper WGMMA intrinsics and 128B swizzled shared-memory descriptors
- Runtime status: succeeded

### Avelang WGMMA BF16 Square GEMM Results

| M | N | K | Repeats | Avg time (ms) | Effective TFLOPS | Estimated bandwidth (GB/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1024 | 200 | 0.0201 | 106.866 | 313.085 |
| 2048 | 2048 | 2048 | 100 | 0.1337 | 128.455 | 188.166 |
| 4096 | 4096 | 4096 | 50 | 0.7805 | 176.087 | 128.970 |
| 8192 | 8192 | 8192 | 20 | 6.1059 | 180.072 | 65.944 |
| 16384 | 16384 | 16384 | 5 | 68.7572 | 127.930 | 23.425 |

Captured output:

```text
M=1024 N=1024 K=1024 time_ms=0.0201 tflops=106.866 bandwidth_gbs=313.085 device=2
M=2048 N=2048 K=2048 time_ms=0.1337 tflops=128.455 bandwidth_gbs=188.166 device=2
M=4096 N=4096 K=4096 time_ms=0.7805 tflops=176.087 bandwidth_gbs=128.970 device=2
M=8192 N=8192 K=8192 time_ms=6.1059 tflops=180.072 bandwidth_gbs=65.944 device=2
M=16384 N=16384 K=16384 time_ms=68.7572 tflops=127.930 bandwidth_gbs=23.425 device=2
```

### Avelang PTX Assembly

PTX dumps generated with:

```bash
cd /workspace/avelang
PYTHONPATH=python python tools/dump_assembly.py \
  --target-triple nvptx64-nvidia-cuda \
  --target-chipset sm_90a \
  --constexprs-json '<size-specialized constexprs>' \
  test/examples/gemm/nvidia/test_gemm_wgmma_bf16_square.py:gemm_wgmma_bf16_square_kernel \
  -o cublaslt/artifacts/avelang_wgmma_bf16_square.size<SIZE>.sm_90a.ptx
```

Generated PTX:

- `artifacts/avelang_wgmma_bf16_square.size1024.sm_90a.ptx`
- `artifacts/avelang_wgmma_bf16_square.size2048.sm_90a.ptx`
- `artifacts/avelang_wgmma_bf16_square.size4096.sm_90a.ptx`
- `artifacts/avelang_wgmma_bf16_square.size8192.sm_90a.ptx`
- `artifacts/avelang_wgmma_bf16_square.size16384.sm_90a.ptx`

The 1024 specialization contains:

```text
32 wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
80 cp.async
18 bar.sync
2 cvt.rn.bf16
2 st.global
```

The saved cuBLASLt PTX artifact
`artifacts/bf16_square_gemm_cublaslt.sm_90a.ptx` contains only the benchmark's
`fill_bf16_kernel`, not the cuBLASLt library GEMM kernel selected at runtime. It
therefore contains no `wgmma` or `cp.async` instructions and is not a direct GEMM
kernel assembly comparison target.
