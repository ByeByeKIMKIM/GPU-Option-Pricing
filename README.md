# GPU Option Pricing

A Monte Carlo engine for pricing path-dependent exotic options, built progressively across three stages: Python/NumPy baseline → C++ with SIMD optimization → CUDA GPU implementation.

## Overview

This project simulates asset price paths using **Geometric Brownian Motion (GBM)** to price two types of exotic options:

- **Asian Call Options** — payoff based on the average price of the asset over time
- **Down-and-Out Barrier Call Options** — payoff depends on whether the asset breaches a barrier

## Stages

### Stage 1: Python/NumPy Baseline

A vectorized Python implementation using NumPy to generate GBM paths and price both option types.

Key functions:
- `generate_gbm_paths(S0, mu, sigma, T, steps, paths)` — generates all simulation paths using the discrete GBM formula and cumulative summation
- `price_asian_call(paths, K, r, T)` — computes the discounted expected payoff for an Asian call
- `price_barrier_out_call(paths, K, B, r, T)` — computes the discounted expected payoff for a down-and-out barrier call

### Stage 2: C++ Rewrite with SIMD Optimization

A C++ translation of the Python code with two layers of optimization:

**C++ Rewrite**
- Uses a flat 1D array instead of a 2D structure for better cache/spatial locality
- Pre-computes constants outside of loops to avoid redundant work
- Pre-allocates memory to avoid costly mid-run reallocations

**AVX2 Optimization**
- Processes 8 float lanes simultaneously using 256-bit AVX2 registers
- Uses `_mm256_fmadd_ps` to fuse multiply-add into a single instruction
- Separates RNG generation from simulation for flexibility and testability
- Uses `avx_mathfun.h` for a vectorized `exp` implementation

### Stage 3: CUDA GPU Implementation

Takes the simulation to the GPU using NVIDIA CUDA (tested on T4 GPUs via Google Colab).

Key concepts:
- Each path runs on its own GPU thread — massively parallel
- Uses `cuRAND` for high-throughput GPU-native random number generation
- Fuses path generation and pricing into a single kernel pass to reduce memory bandwidth
- Uses Thrust tree-based reduction for the final summation step

## Performance Results (1,000,000 paths)

| Implementation     | Total Time   |
|--------------------|-------------|
| Python/NumPy       | 13.1546 s   |
| C++                | 9.7335 s    |
| AVX-Optimized C++  | 5.2558 s    |
| CUDA               | 0.0352 s    |

The CUDA implementation achieves a **~374x speedup** over the Python baseline.

## Background Concepts

**Why Log-Normal?** Asset prices are modeled with a log-normal distribution because prices can't go below zero and exhibit right skew — gains can compound without bound, but losses are capped at total value.

**GBM SDE:** `dS_t = μS_t dt + σS_t dW_t`

**Discrete update formula:** `S_{t+Δt} = S_t · exp((μ - σ²/2)Δt + σ√Δt · Z)`

**Why Monte Carlo for Asian options?** The arithmetic average of log-normal variables is not itself log-normal, so closed-form solutions don't exist — simulation is the practical approach.

## Validation

Using parameters `S0=100, K=100, B=90, r=0.05, σ=0.2, T=1.0`:

- Analytical barrier call price: **8.6655**
- MC simulation (1000 steps): **8.8408** (~17 cents off, attributable to discrete vs. continuous monitoring)
- MC simulation (5000 steps): **8.7459** (improved convergence)

Asian call stability check across 5 seeds: `[5.7232, 5.7434, 5.7601, 5.7468, 5.7326]` — consistent results confirm the simulation is stable.

## Takeaways

- Normal vs. log-normal distribution and why it matters for asset modeling
- The mechanics of GBM: drift, volatility, and volatility drag
- Vectorized path generation with NumPy
- Asian and barrier option payoff formulas
- C++ performance techniques: flat arrays, pre-allocation, loop-invariant hoisting
- AVX2 SIMD programming for CPU-level parallelism
- Basic GPU paradigms and CUDA programming