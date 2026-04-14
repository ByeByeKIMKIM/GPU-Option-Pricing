%%writefile gbm_avx.cpp

#include "avx_mathfun.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <chrono>

typedef __m256 v8sf;

// ── GBM path generator (AVX float, block RNG) ──────────────────────────────
//
// Parameters:
//   S0, mu, sigma, T  – standard GBM parameters
//   steps, paths      – simulation dimensions
//   data              – output buffer: (steps+1) * num_paths_aligned floats,
//                       32-byte aligned (use std::aligned_alloc)
//   rng_block         – pre-filled normal random buffer: steps * num_paths_aligned
//                       floats, 32-byte aligned, laid out as [step][path]
//
void generate_gbm_paths_avx_float(float S0, float mu, float sigma, float T,
                                  int steps, int paths,
                                  float* data, const float* rng_block)
{
    // Round up to the next multiple of 8 so every row is 32-byte aligned
    int num_paths_aligned = (paths + 7) & ~7;

    // ── scalar constants ────────────────────────────────────────────────────
    float delta_t        = T / static_cast<float>(steps);
    float volatility_drag = (mu - 0.5f * sigma * sigma) * delta_t;
    float normal_scalar  = sigma * std::sqrt(delta_t);

    // ── broadcast into 8-wide AVX registers ────────────────────────────────
    v8sf v_vol_drag   = _mm256_set1_ps(volatility_drag);
    v8sf v_norm_scale = _mm256_set1_ps(normal_scalar);
    v8sf v_s0         = _mm256_set1_ps(S0);

    // ── step 0: every path starts at S0 ────────────────────────────────────
    for (int i = 0; i < num_paths_aligned; i += 8) {
        _mm256_store_ps(&data[0 * num_paths_aligned + i], v_s0);
    }

    // ── time-step loop ──────────────────────────────────────────────────────
    for (int j = 1; j <= steps; j++) {
        for (int i = 0; i < num_paths_aligned; i += 8) {

            // Load 8 pre-generated normals directly – no sampling here
            v8sf v_rand = _mm256_load_ps(&rng_block[(j - 1) * num_paths_aligned + i]);

            // Load previous prices
            v8sf v_prev_price = _mm256_load_ps(&data[(j - 1) * num_paths_aligned + i]);

            // exponent = vol_drag + norm_scale * rand  (FMA)
            v8sf v_exponent = _mm256_fmadd_ps(v_norm_scale, v_rand, v_vol_drag);

            // S_next = S_prev * exp(exponent)  – 8 lanes in parallel
            v8sf v_next_price = _mm256_mul_ps(v_prev_price, exp256_ps(v_exponent));

            // Store results
            _mm256_store_ps(&data[j * num_paths_aligned + i], v_next_price);
        }
    }
}


double price_asian_call_avx(const float* data, float K, float r, float T, int steps, int paths) {
    int num_paths_aligned = (paths + 7) & ~7;

    v8sf v_K = _mm256_set1_ps(K);
    v8sf v_steps = _mm256_set1_ps(static_cast<float>(steps));
    v8sf v_zero = _mm256_setzero_ps();

    // Accumulate final sum in double to prevent precision loss across 100k+ paths
    double total_payoff = 0.0;

    // Process 8 paths simultaneously
    for (int i = 0; i < num_paths_aligned; i += 8) {
        v8sf v_sum = _mm256_setzero_ps();

        // Sum prices from step 1 to steps (standard Asian average excludes S0)
        for (int j = 1; j <= steps; j++) {
            v8sf v_price = _mm256_load_ps(&data[j * num_paths_aligned + i]);
            v_sum = _mm256_add_ps(v_sum, v_price);
        }

        // average = total / steps
        v8sf v_avg = _mm256_div_ps(v_sum, v_steps);

        // payoff = max(average - K, 0.0)
        v8sf v_payoff = _mm256_max_ps(_mm256_sub_ps(v_avg, v_K), v_zero);

        // Store the 8 parallel payoffs to a temporary array
        alignas(32) float payoffs[8];
        _mm256_store_ps(payoffs, v_payoff);

        // Add to total, respecting the actual path count (ignoring padded lanes)
        int valid_lanes = std::min(8, paths - i);
        for (int k = 0; k < valid_lanes; k++) {
            total_payoff += payoffs[k];
        }
    }

    return (total_payoff / paths) * std::exp(-r * T);
}

// ── AVX Down-and-Out Barrier Call Pricer ────────────────────────────────────
double price_barrier_out_call_avx(const float* data, float K, float B, float r, float T, int steps, int paths) {
    int num_paths_aligned = (paths + 7) & ~7;

    v8sf v_K = _mm256_set1_ps(K);
    v8sf v_B = _mm256_set1_ps(B);
    v8sf v_zero = _mm256_setzero_ps();

    double total_payoff = 0.0;

    // Process 8 paths simultaneously
    for (int i = 0; i < num_paths_aligned; i += 8) {
        // Initialize a mask to all 1s (true).
        // Comparing zero == zero is a safe trick to generate an all-1s bitmask.
        v8sf v_valid = _mm256_cmp_ps(v_zero, v_zero, _CMP_EQ_OQ);

        // Check barrier condition from step 0 to steps
        for (int j = 0; j <= steps; j++) {
            v8sf v_price = _mm256_load_ps(&data[j * num_paths_aligned + i]);

            // Compare: true (all 1s) if price >= B, false (all 0s) if price < B
            v8sf v_cmp = _mm256_cmp_ps(v_price, v_B, _CMP_GE_OQ);

            // Bitwise AND accumulates knocked-out states. Once a lane hits 0, it stays 0.
            v_valid = _mm256_and_ps(v_valid, v_cmp);
        }

        // Calculate final raw payoff: max(S_T - K, 0.0)
        v8sf v_final_price = _mm256_load_ps(&data[steps * num_paths_aligned + i]);
        v8sf v_payoff = _mm256_max_ps(_mm256_sub_ps(v_final_price, v_K), v_zero);

        // Apply validity mask. Knocked out paths will have their payoffs bitwise-ANDed with 0.
        v_payoff = _mm256_and_ps(v_payoff, v_valid);

        alignas(32) float payoffs[8];
        _mm256_store_ps(payoffs, v_payoff);

        int valid_lanes = std::min(8, paths - i);
        for (int k = 0; k < valid_lanes; k++) {
            total_payoff += payoffs[k];
        }
    }

    return (total_payoff / paths) * std::exp(-r * T);
}

int main() {

    // ── 1. Simulation parameters ────────────────────────────────────────────
    const float S0    = 100.0f;
    const float mu    = 0.05f;
    const float sigma = 0.20f;
    const float T     = 1.0f;
    const int   steps = 252;
    const int   paths = 1'000'000;

    // Pricing parameters
    const float K     = 100.0f;
    const float B     = 90.0f;
    const float r     = 0.05f;

    int num_paths_aligned = (paths + 7) & ~7;

    // ── 2. Aligned allocation ───────────────────────────────────────────────
    const size_t data_bytes = static_cast<size_t>(steps + 1) * num_paths_aligned * sizeof(float);
    const size_t rng_bytes  = static_cast<size_t>(steps)     * num_paths_aligned * sizeof(float);

    float* data = static_cast<float*>(std::aligned_alloc(32, data_bytes));
    float* rng  = static_cast<float*>(std::aligned_alloc(32, rng_bytes));

    if (!data || !rng) {
        std::cerr << "aligned_alloc failed – not enough memory\n";
        std::free(data);
        std::free(rng);
        return 1;
    }

    // ── 3. Block RNG fill (timed separately) ────────────────────────────────
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    auto t_rng_start = std::chrono::high_resolution_clock::now();

    const size_t total_randoms = static_cast<size_t>(steps) * num_paths_aligned;
    for (size_t k = 0; k < total_randoms; ++k) {
        rng[k] = dist(gen);
    }

    auto t_rng_end = std::chrono::high_resolution_clock::now();
    double rng_ms = std::chrono::duration<double, std::milli>(t_rng_end - t_rng_start).count();

    // ── 4. Run simulation (timed separately) ────────────────────────────────
    auto t_sim_start = std::chrono::high_resolution_clock::now();

    generate_gbm_paths_avx_float(S0, mu, sigma, T, steps, paths, data, rng);

    auto t_sim_end = std::chrono::high_resolution_clock::now();
    double sim_ms = std::chrono::duration<double, std::milli>(t_sim_end - t_sim_start).count();

    // ── 5. Asian Call Pricing (timed separately) ────────────────────────────
    auto t_asian_start = std::chrono::high_resolution_clock::now();

    double asian_price = price_asian_call_avx(data, K, r, T, steps, paths);

    auto t_asian_end = std::chrono::high_resolution_clock::now();
    double asian_ms = std::chrono::duration<double, std::milli>(t_asian_end - t_asian_start).count();

    // ── 6. Barrier Knock-Out Pricing (timed separately) ─────────────────────
    auto t_barrier_start = std::chrono::high_resolution_clock::now();

    double barrier_price = price_barrier_out_call_avx(data, K, B, r, T, steps, paths);

    auto t_barrier_end = std::chrono::high_resolution_clock::now();
    double barrier_ms = std::chrono::duration<double, std::milli>(t_barrier_end - t_barrier_start).count();

    // ── 7. Timing report & Output ───────────────────────────────────────────
    double total_ms = rng_ms + sim_ms + asian_ms + barrier_ms;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== Results =====================================\n";
    std::cout << "  Asian Call Price        : $" << asian_price << "\n";
    std::cout << "  Barrier Knock-Out Price : $" << barrier_price << "\n";

    std::cout << "\n=== Benchmark (" << paths << " paths) ==============\n";
    std::cout << "  RNG fill time           : " << std::setw(8) << rng_ms     << " ms\n";
    std::cout << "  GBM Simulation time     : " << std::setw(8) << sim_ms     << " ms\n";
    std::cout << "  Asian Pricing time      : " << std::setw(8) << asian_ms   << " ms\n";
    std::cout << "  Barrier Pricing time    : " << std::setw(8) << barrier_ms << " ms\n";
    std::cout << "  -----------------------------------------------\n";
    std::cout << "  Total Execution Time    : " << std::setw(8) << total_ms   << " ms\n";
    std::cout << "=================================================\n";

    std::free(data);
    std::free(rng);
    return 0;
}