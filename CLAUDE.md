# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TileLang Puzzles is a progressive tutorial for learning [TileLang](https://github.com/tile-ai/tilelang), a Python DSL for writing high-performance GPU kernels. It contains 10 puzzles progressing from basic copy operations to GEMM and FlashAttention.

## Repository Structure

- `puzzles/` — Incomplete puzzle templates with `# TODO` markers for the learner to fill in
- `ans/` — Complete reference solutions for all 10 puzzles
- `common/utils.py` — Shared `test_puzzle()` and `bench_puzzle()` utilities
- `scripts/check_tilelang_env.py` — Environment verification script

## Running Puzzles

Each puzzle is a standalone script. Run from the repo root:
```bash
python3 puzzles/01-copy.py    # Run puzzle (with TODOs)
python3 ans/01-copy.py        # Run reference solution
```

Verify environment: `python3 scripts/check_tilelang_env.py`

## Linting

Uses Ruff (configured in `ruff.toml`): line-length 100, Python 3.10+, double quotes, 4-space indent.
```bash
ruff check .
ruff format .
```

## TileLang Kernel Pattern

Every kernel follows this structure:
```python
@tilelang.jit
def kernel(A, BLOCK_SIZE: int):
    # 1. Host declarations
    N = T.const("N")
    A: T.Tensor((N,), T.float16)      # annotate input
    B = T.empty((N,), T.float16)       # allocate output

    # 2. Kernel body
    with T.Kernel(num_blocks, threads=128) as block_id:
        shared = T.alloc_shared(shape, dtype)       # shared memory
        frag = T.alloc_fragment(shape, dtype)        # registers
        T.copy(src, dst)                             # tile copy
        T.gemm(A_shared, B_shared, C_frag)           # matrix multiply
        for i in T.Parallel(size):                   # parallel loop
            frag[i] = ...
    return B
```

Key TileLang primitives: `T.const`, `T.Tensor`, `T.empty`, `T.Kernel`, `T.alloc_shared`, `T.alloc_fragment`, `T.copy`, `T.gemm`, `T.fill`, `T.clear`, `T.reduce_sum`, `T.Parallel`, `T.Serial`, `T.Pipelined`, `T.ceildiv`.

## Testing Pattern

No test framework — each file has inline `run_*()` functions that call:
- `test_puzzle(tl_kernel, torch_ref, hyper_params)` — compiles kernel, generates random inputs, compares output against PyTorch reference (atol/rtol=1e-2)
- `bench_puzzle(tl_kernel, torch_ref, hyper_params, bench_torch=True)` — benchmarks kernel vs PyTorch

The last tensor parameter is always the output (auto-allocated by the test harness).

## Puzzle Progression

01-Copy, 02-VecAdd, 03-OuterVecAdd, 04-BackwardOp (easy) → 05-ReduceSum, 06-Softmax, 07-ScalarFlashAttention, 08-Matrix, 09-Convolution (medium) → 10-DequantMatmul (hard).

## When Writing Kernels

- Import convention: `import tilelang.language as T`
- Hyperparameters (block sizes, etc.) are function params with `int` annotation
- Problem dimensions use `T.const("N")` / `T.const("M, N, K")`
- Use `T.alloc_shared` for data fed to `T.gemm`; use `T.alloc_fragment` for accumulators and element-wise work
- Use `T.Pipelined` with `num_stages` instead of `T.Serial` for software-pipelined loops in optimized kernels
- Accumulator dtype is typically `T.float32` while input/output is `T.float16`
