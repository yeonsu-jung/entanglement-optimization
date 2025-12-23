#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Include the header with analytic expressions
#include "../../entanglement-optimization-cpp/cpp_analytic_expressions.hpp"

using namespace entanglement;

// Rod structure for benchmark
struct BenchmarkRod {
  Rod rod;
};

int main(int argc, char *argv[]) {
  // Number of pairs to process
  size_t N_PAIRS = 100000;
  if (argc > 1) {
    N_PAIRS = std::stoul(argv[1]);
  }

  std::cout << "Generating " << N_PAIRS << " random rod pairs..." << std::endl;

  // Random number generation
  std::mt19937 gen(42);
  std::uniform_real_distribution<> pos_dist(-5.0, 5.0);
  std::uniform_real_distribution<> angle_dist(0.0, 2 * M_PI);

  std::vector<std::pair<Rod, Rod>> pairs;
  pairs.reserve(N_PAIRS);

  for (size_t i = 0; i < N_PAIRS; ++i) {
    Rod r1, r2;
    r1.center = {pos_dist(gen), pos_dist(gen), pos_dist(gen)};
    r1.phi = angle_dist(gen);
    r1.theta = angle_dist(gen);
    r1.length = 1.0;

    r2.center = {pos_dist(gen), pos_dist(gen), pos_dist(gen)};
    r2.phi = angle_dist(gen);
    r2.theta = angle_dist(gen);
    r2.length = 1.0;

    pairs.push_back({r1, r2});
  }

  std::cout << "Starting benchmark..." << std::endl;

  // Warmup
  volatile double sink = 0;
  for (size_t i = 0; i < std::min((size_t)1000, N_PAIRS); ++i) {
    sink += rod_rod_distance(pairs[i].first, pairs[i].second);
  }

  auto start = std::chrono::high_resolution_clock::now();

  double total_dist = 0;
  double total_lk = 0;

  // Benchmark Loop
  for (const auto &pair : pairs) {
    // 1. Distance
    double d = rod_rod_distance(pair.first, pair.second);
    total_dist += d;

    // 2. Linking Number (Gauss)
    double lk = linking_number_gauss(
        pair.first.get_endpoints().first, pair.first.get_endpoints().second,
        pair.second.get_endpoints().first, pair.second.get_endpoints().second);
    total_lk += lk;

    // 3. Gradients (Distance)
    // Note: In real optimization we compute gradients for active pairs only,
    // but here we benchmark the cost processing every pair.
    RodPairGradients g = segment_distance_gradient(pair.first, pair.second);
    sink += g.grad_rod1.dx;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> elapsed = end - start;

  double avg_us = elapsed.count() / N_PAIRS;

  std::cout << "Benchmark Complete." << std::endl;
  std::cout << "Total Pairs: " << N_PAIRS << std::endl;
  std::cout << "Total Time: " << (elapsed.count() / 1000.0) << " ms"
            << std::endl;
  std::cout << "Average Time per Pair: " << avg_us << " us" << std::endl;

  // Prevent optimization
  if (sink > 1e9)
    std::cout << " " << sink;

  return 0;
}
