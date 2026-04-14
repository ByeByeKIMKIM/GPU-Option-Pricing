#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>

//g++ -O3 -march=native -std=c++17 -o monte_carlo main.cpp

std::vector<double> generate_gbm_paths(double S0, double mu, double sigma, int T, int steps, int paths, std::mt19937& gen) {
    std::normal_distribution<double> d(0.0, 1.0);
    std::vector<double> paths_matrix(paths * (steps + 1), 0.0);
    double delta_t = static_cast<double>(T) / steps;
    double volatility_drag = (mu - (sigma * sigma) / 2.0) * delta_t;
    double normal_scalar = sigma * std::sqrt(delta_t);

    for (int i = 0; i < paths; i++) {
        paths_matrix[i * (steps + 1)] = S0;
    }

    for (int i = 0; i < paths; i++) {
        for (int j = 1; j <= steps; j++) {
            double random = d(gen);
            int cur_index  = i * (steps + 1) + j;
            int prev_index = i * (steps + 1) + (j - 1);
            paths_matrix[cur_index] = paths_matrix[prev_index] * std::exp(volatility_drag + normal_scalar * random);
        }
    }
    return paths_matrix;
}

double price_asian_call(const std::vector<double>& paths_matrix, double K, double r, double T, int steps, int num_paths) {
    double total_payoff = 0.0;
    for (int i = 0; i < num_paths; i++) {
        auto start = paths_matrix.begin() + i * (steps + 1) + 1;
        auto end = paths_matrix.begin() + i * (steps + 1) + steps + 1;
        double total   = std::accumulate(start, end, 0.0);
        double average = total / steps;
        double payoff  = std::max(average - K, 0.0);
        total_payoff  += payoff;
    }
    return (total_payoff / num_paths) * std::exp(-r * T);
}

double price_barrier_out_call(const std::vector<double>& paths_matrix, double K, double B, double r, double T, int steps, int num_paths) {
    double total_payoff = 0.0;
    for (int i=0; i<num_paths; i++) {
        auto start = paths_matrix.begin() + i * (steps + 1);
        auto end = paths_matrix.begin() + i * (steps + 1) + steps + 1;

        bool broke = std::any_of(start, end, [&B](double price){return price < B;});
        if(!broke) {
            total_payoff += std::max(paths_matrix[i * (steps + 1) + steps] - K, 0.0);
        }
    }
    return (total_payoff / num_paths) * std::exp(-r * T);
}


int main() {
    // Parameters
    double S0    = 100.0;
    double mu    = 0.05;
    double sigma = 0.20;
    double K     = 100.0;
    double B     = 90.0;
    double r     = 0.05;
    int    T     = 1;
    int    steps = 252;
    int    paths = 1000000;

    std::mt19937 gen(42);

    std::cout << "Generating " << paths << " paths...\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    auto paths_matrix    = generate_gbm_paths(S0, mu, sigma, T, steps, paths, gen);
    double asian_price   = price_asian_call(paths_matrix, K, r, T, steps, paths);
    double barrier_price = price_barrier_out_call(paths_matrix, K, B, r, T, steps, paths);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Asian Call Price:         $" << asian_price   << "\n";
    std::cout << "Barrier Knock-Out Price:  $" << barrier_price << "\n";
    std::cout << "Time:                      " << elapsed       << "s\n";

    return 0;
}