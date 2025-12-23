#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "entanglement.hpp"
#include "geometry.hpp"
#include "gradients.hpp"
#include "optimization.hpp"

using namespace entanglement;

// Global configuration matching standard_protocol
const int NUM_RODS = 200;
const int MAX_ITER = 300; // Matches standard_protocol inner loop
const int N_OUTER = 1;    // Benchmark one outer loop iteration basically (or
                          // standard protocol does 1?)
// Actually standard_protocol does N_outer calls. For benchmark, doing 1 call of
// MAX_ITER is sufficient to gauge speed.
const double DT = 0.01;
const double ROD_LENGTH = 1.0;

// Energy function: E = -sum_{i<j} |Lk(i,j)|
double energy_func(const std::vector<Rod> &rods) {
  double sum = 0.0;
  for (size_t i = 0; i < rods.size(); ++i) {
    for (size_t j = i + 1; j < rods.size(); ++j) {
      double lk = rod_linking_number(rods[i], rods[j], false); // Use Gauss
      sum += -std::abs(lk);
    }
  }
  return sum;
}

// Gradient function
std::vector<Gradient5D> gradient_func(const std::vector<Rod> &rods) {
  std::vector<Gradient5D> grads(rods.size());
  // Zero initialize
  for (auto &g : grads)
    g = {0, 0, 0, 0, 0};

  const double s_eps = 1e-12;
  // Calculate gradients pair-wise
  for (size_t i = 0; i < rods.size(); ++i) {
    for (size_t j = i + 1; j < rods.size(); ++j) {
      double lk = rod_linking_number(rods[i], rods[j], false);
      double s = (lk > s_eps) ? 1.0 : (lk < -s_eps ? -1.0 : 0.0);

      if (s == 0.0)
        continue;

      // Compute gradient of Lk
      // Note: linking_gradient uses finite difference by default or calls
      // analytical if avail? include/gradients.hpp says "Compute gradient of
      // linking number using finite differences"
      auto lk_grad = linking_gradient(rods[i], rods[j], 1e-8, false);

      // E = -|lk| => dE = -sgn(lk) * d(lk)
      double factor = -s;

      grads[i].dx += factor * lk_grad.grad_rod1.dx;
      grads[i].dy += factor * lk_grad.grad_rod1.dy;
      grads[i].dz += factor * lk_grad.grad_rod1.dz;
      grads[i].dphi += factor * lk_grad.grad_rod1.dphi;
      grads[i].dtheta += factor * lk_grad.grad_rod1.dtheta;

      grads[j].dx += factor * lk_grad.grad_rod2.dx;
      grads[j].dy += factor * lk_grad.grad_rod2.dy;
      grads[j].dz += factor * lk_grad.grad_rod2.dz;
      grads[j].dphi += factor * lk_grad.grad_rod2.dphi;
      grads[j].dtheta += factor * lk_grad.grad_rod2.dtheta;
    }
  }
  return grads;
}

int main() {
  std::cout << "Benchmarking C++ Optimization Protocol (FIRE)" << std::endl;
  std::cout << "N_RODS: " << NUM_RODS << ", MAX_ITER: " << MAX_ITER
            << std::endl;

  // 1. Generate random configuration
  auto rods = generate_random_configuration(NUM_RODS, 10.0, ROD_LENGTH, 42);

  // 2. Setup Optimizer
  FireOptimizer::Parameters params;
  params.max_iter = MAX_ITER;
  params.dt_init = DT;
  params.dt_max = 10 * DT;
  params.dt_min = 0.02 * DT;
  params.n_min = 5;
  params.f_inc = 1.1;
  params.f_dec = 0.5;
  params.alpha_init = 0.1;
  params.f_alpha = 0.99;
  params.force_tol = 0.0; // Don't stop early for benchmark
  params.energy_tol = 0.0;
  params.verbose = true;

  FireOptimizer optimizer(params);

  // 3. Run Optimization Benchmark
  auto start = std::chrono::high_resolution_clock::now();

  auto result = optimizer.optimize(rods, energy_func, gradient_func);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  std::cout << "Benchmark Complete." << std::endl;
  std::cout << "Converged: " << (result.converged ? "Yes" : "No") << std::endl;
  std::cout << "Iterations: " << result.iterations << std::endl;
  std::cout << "Final Energy: " << result.final_energy << std::endl;
  std::cout << "Total Time: " << elapsed.count() << " seconds" << std::endl;
  std::cout << "Time per Iteration: " << (elapsed.count() / result.iterations)
            << " seconds" << std::endl;
  std::cout << "Avg Time per Pair-Step: "
            << (elapsed.count() / result.iterations /
                (NUM_RODS * (NUM_RODS - 1) / 2) * 1e6)
            << " us" << std::endl;

  return 0;
}
