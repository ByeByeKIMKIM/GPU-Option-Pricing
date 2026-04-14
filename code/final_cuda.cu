#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <chrono>
#include <iostream>
#include <iomanip>

// ── 1. Constant Memory Declarations ─────────────────────────────────────────
__constant__ float c_S0;
__constant__ float c_r;
__constant__ float c_sigma;
__constant__ float c_T;
__constant__ float c_B;
__constant__ float c_K;
__constant__ int   c_steps;

// ---------------------------------------------------------------------------
// RNG setup kernel
// ---------------------------------------------------------------------------
__global__ void setup_kernel(curandState* states, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// ---------------------------------------------------------------------------
// Fused GBM kernel (Cleaned Signature)
// ---------------------------------------------------------------------------
__global__ void gbm_kernel_fused(
    int    paths,
    curandState* devStates,
    float* devAsianPayoffs,
    float* devBarrierPayoffs,
    float* devFinalPrices)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= paths) return;

    curandState localState = devStates[idx];

    float dt        = c_T / (float)c_steps;
    float drift     = (c_r - 0.5f * c_sigma * c_sigma) * dt;
    float diffusion = c_sigma * sqrtf(dt);

    float  currentPrice = c_S0;
    double runningSum   = 0.0;
    bool   isKnockedOut = (c_S0 < c_B);

    for (int j = 1; j <= c_steps; j++) {
        float Z      = curand_normal(&localState);
        currentPrice = currentPrice * expf(drift + diffusion * Z);
        runningSum  += (double)currentPrice;
        if (currentPrice < c_B) isKnockedOut = true;
    }

    devStates[idx] = localState;

    float avgPrice = (float)(runningSum / (double)c_steps);
    devAsianPayoffs[idx]   = fmaxf(avgPrice - c_K, 0.0f);
    devBarrierPayoffs[idx] = isKnockedOut ? 0.0f : fmaxf(currentPrice - c_K, 0.0f);
    devFinalPrices[idx]    = currentPrice;
}

int main() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        std::cerr << "ERROR: No CUDA devices found!\n";
        return 1;
    }

    // Host parameters
    const float S0    = 100.0f;
    const float K     = 100.0f;
    const float B     =  90.0f;
    const float r     =   0.05f;
    const float sigma =   0.20f;
    const float T     =   1.0f;
    const int   steps =  252;
    const int   paths = 1000000;

    const int threads_per_block = 256;
    const int blocks_per_grid   = (paths + threads_per_block - 1) / threads_per_block;

    // ── 2. Initialize Constant Memory ───────────────────────────────────────
    cudaMemcpyToSymbol(c_S0,    &S0,    sizeof(float));
    cudaMemcpyToSymbol(c_r,     &r,     sizeof(float));
    cudaMemcpyToSymbol(c_sigma, &sigma, sizeof(float));
    cudaMemcpyToSymbol(c_T,     &T,     sizeof(float));
    cudaMemcpyToSymbol(c_B,     &B,     sizeof(float));
    cudaMemcpyToSymbol(c_K,     &K,     sizeof(float));
    cudaMemcpyToSymbol(c_steps, &steps, sizeof(int));

    // Device allocations
    curandState* devStates;
    float *devAsian, *devBarrier, *devFinal;

    cudaMalloc(&devStates,  paths * sizeof(curandState));
    cudaMalloc(&devAsian,   paths * sizeof(float));
    cudaMalloc(&devBarrier, paths * sizeof(float));
    cudaMalloc(&devFinal,   paths * sizeof(float));

    auto t_total_start = std::chrono::high_resolution_clock::now();

    // 3. RNG Setup
    auto t1 = std::chrono::high_resolution_clock::now();
    setup_kernel<<<blocks_per_grid, threads_per_block>>>(devStates, 42ULL);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    // 4. Simulation Kernel
    auto t3 = std::chrono::high_resolution_clock::now();
    gbm_kernel_fused<<<blocks_per_grid, threads_per_block>>>(
        paths, devStates, devAsian, devBarrier, devFinal);
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();

    // 5. Thrust On-Device Reduction
    auto t5 = std::chrono::high_resolution_clock::now();
    double asianSum   = thrust::reduce(thrust::device, devAsian,   devAsian   + paths, 0.0, thrust::plus<double>());
    double barrierSum = thrust::reduce(thrust::device, devBarrier, devBarrier + paths, 0.0, thrust::plus<double>());
    double finalSum   = thrust::reduce(thrust::device, devFinal,   devFinal   + paths, 0.0, thrust::plus<double>());
    auto t6 = std::chrono::high_resolution_clock::now();

    auto t_total_end = std::chrono::high_resolution_clock::now();

    // Results logic
    double discount      = exp(-(double)r * (double)T);
    double asian_price   = (asianSum   / paths) * discount;
    double barrier_price = (barrierSum / paths) * discount;

    // Timing calculations
    double rng_ms    = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double sim_ms    = std::chrono::duration<double, std::milli>(t4 - t3).count();
    double thrust_ms = std::chrono::duration<double, std::milli>(t6 - t5).count();
    double total_ms  = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== Results (Thrust + Constant Memory) ===\n";
    std::cout << "  Asian Call Price        : $" << asian_price << "\n";
    std::cout << "  Barrier Knock-Out Price : $" << barrier_price << "\n";
    std::cout << "\n=== Benchmark (" << paths << " paths) ===\n";
    std::cout << "  RNG Setup: " << rng_ms << " ms | Sim: " << sim_ms << " ms | Thrust: " << thrust_ms << " ms\n";
    std::cout << "  Total Execution: " << total_ms << " ms\n";

    cudaFree(devStates); cudaFree(devAsian); cudaFree(devBarrier); cudaFree(devFinal);
    return 0;
}