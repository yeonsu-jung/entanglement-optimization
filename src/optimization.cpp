/**
 * @file optimization.cpp
 * @brief Implementation of optimization algorithms
 */

#include "optimization.hpp"
#include "distance.hpp"
#include <iostream>
#include <random>
#include <algorithm>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace entanglement {

void FireOptimizer::reset_velocities() {
    for (auto& v : velocities_) v = {0, 0, 0};
    for (auto& av : angular_vel_) av = {0, 0, 0};
}

void FireOptimizer::update_timestep(double P) {
    static int steps_since_reset = 0;
    static double alpha = params_.alpha_init;
    static double dt = params_.dt_init;
    
    if (P > 0) {
        steps_since_reset++;
        if (steps_since_reset > params_.n_min) {
            dt = std::min(dt * params_.f_inc, params_.dt_max);
            alpha *= params_.f_alpha;
        }
    } else {
        steps_since_reset = 0;
        dt = std::max(dt * params_.f_dec, params_.dt_min);
        alpha = params_.alpha_init;
        reset_velocities();
    }
}

void FireOptimizer::apply_constraints(std::vector<Rod>& rods) {
    // Ensure phi is in [0, π] and theta is in [0, 2π]
    for (auto& rod : rods) {
        rod.phi = std::fmod(rod.phi, M_PI);
        if (rod.phi < 0) rod.phi += M_PI;
        
        rod.theta = std::fmod(rod.theta, 2.0 * M_PI);
        if (rod.theta < 0) rod.theta += 2.0 * M_PI;
    }
}

FireOptimizer::Result FireOptimizer::optimize(
    std::vector<Rod>& rods,
    std::function<double(const std::vector<Rod>&)> energy_func,
    std::function<std::vector<Gradient5D>(const std::vector<Rod>&)> gradient_func) {
    
    Result result;
    
    // Initialize velocities
    velocities_.resize(rods.size(), {0, 0, 0});
    angular_vel_.resize(rods.size(), {0, 0, 0});
    
    double dt = params_.dt_init;
    double alpha = params_.alpha_init;
    int steps_since_reset = 0;
    
    double prev_energy = energy_func(rods);
    result.energy_history.push_back(prev_energy);
    
    for (int iter = 0; iter < params_.max_iter; ++iter) {
        
        // Compute forces (negative gradients)
        auto gradients = gradient_func(rods);
        
        // Convert gradients to forces and compute norms
        double force_norm = 0.0;
        std::vector<Vec3> forces(rods.size());
        std::vector<Vec3> torques(rods.size());
        
        for (size_t i = 0; i < rods.size(); ++i) {
            forces[i] = {-gradients[i].dx, -gradients[i].dy, -gradients[i].dz};
            torques[i] = {-gradients[i].dphi, -gradients[i].dtheta, 0.0};
            
            force_norm += forces[i].norm2() + torques[i].norm2();
        }
        force_norm = std::sqrt(force_norm);
        
        // Check convergence
        double energy = energy_func(rods);
        double energy_change = std::abs(energy - prev_energy);
        
        if (params_.verbose && iter % 100 == 0) {
            std::cout << "Iter " << iter << ": Energy = " << energy 
                      << ", Force norm = " << force_norm << std::endl;
        }
        
        if (force_norm < params_.force_tol && energy_change < params_.energy_tol) {
            result.converged = true;
            break;
        }
        
        // FIRE algorithm update
        double P = 0.0;
        for (size_t i = 0; i < rods.size(); ++i) {
            P += velocities_[i].dot(forces[i]) + angular_vel_[i].dot(torques[i]);
        }
        
        // Update timestep and mixing parameter
        if (P > 0) {
            steps_since_reset++;
            if (steps_since_reset > params_.n_min) {
                dt = std::min(dt * params_.f_inc, params_.dt_max);
                alpha *= params_.f_alpha;
            }
        } else {
            steps_since_reset = 0;
            dt = std::max(dt * params_.f_dec, params_.dt_min);
            alpha = params_.alpha_init;
            reset_velocities();
        }
        
        // Update velocities and positions
        for (size_t i = 0; i < rods.size(); ++i) {
            // Velocity update with FIRE mixing
            velocities_[i] = velocities_[i] * (1.0 - alpha) + forces[i].normalized() * (alpha * velocities_[i].norm());
            angular_vel_[i] = angular_vel_[i] * (1.0 - alpha) + torques[i].normalized() * (alpha * angular_vel_[i].norm());
            
            // Integrate
            velocities_[i] += forces[i] * dt;
            angular_vel_[i] += torques[i] * dt;
            
            // Update positions
            rods[i].center += velocities_[i] * dt;
            rods[i].phi += angular_vel_[i].x * dt;
            rods[i].theta += angular_vel_[i].y * dt;
        }
        
        apply_constraints(rods);
        
        prev_energy = energy;
        result.energy_history.push_back(energy);
        result.iterations = iter + 1;
    }
    
    result.final_energy = energy_func(rods);
    result.final_force_norm = compute_force_norm(gradient_func(rods));
    
    return result;
}

GradientDescentOptimizer::Result GradientDescentOptimizer::optimize(
    std::vector<Rod>& rods,
    std::function<double(const std::vector<Rod>&)> energy_func,
    std::function<std::vector<Gradient5D>(const std::vector<Rod>&)> gradient_func) {
    
    Result result;
    double learning_rate = params_.learning_rate;
    double prev_energy = energy_func(rods);
    
    for (int iter = 0; iter < params_.max_iter; ++iter) {
        
        auto gradients = gradient_func(rods);
        double force_norm = compute_force_norm(gradients);
        
        double energy = energy_func(rods);
        double energy_change = std::abs(energy - prev_energy);
        
        if (params_.verbose && iter % 100 == 0) {
            std::cout << "Iter " << iter << ": Energy = " << energy 
                      << ", Force norm = " << force_norm << std::endl;
        }
        
        if (force_norm < params_.force_tol && energy_change < params_.energy_tol) {
            result.converged = true;
            break;
        }
        
        // Adaptive learning rate
        if (params_.adaptive_rate) {
            if (energy > prev_energy) {
                learning_rate *= 0.8;  // Reduce if energy increased
            } else {
                learning_rate *= 1.05; // Increase if energy decreased
            }
            learning_rate = std::clamp(learning_rate, 1e-6, 0.1);
        }
        
        // Update positions
        for (size_t i = 0; i < rods.size(); ++i) {
            rods[i].center.x -= learning_rate * gradients[i].dx;
            rods[i].center.y -= learning_rate * gradients[i].dy;
            rods[i].center.z -= learning_rate * gradients[i].dz;
            rods[i].phi -= learning_rate * gradients[i].dphi;
            rods[i].theta -= learning_rate * gradients[i].dtheta;
        }
        
        // Apply constraints
        for (auto& rod : rods) {
            rod.phi = std::fmod(rod.phi, M_PI);
            if (rod.phi < 0) rod.phi += M_PI;
            
            rod.theta = std::fmod(rod.theta, 2.0 * M_PI);
            if (rod.theta < 0) rod.theta += 2.0 * M_PI;
        }
        
        prev_energy = energy;
        result.iterations = iter + 1;
    }
    
    result.final_energy = energy_func(rods);
    result.final_force_norm = compute_force_norm(gradient_func(rods));
    
    return result;
}

double compute_force_norm(const std::vector<Gradient5D>& gradients) {
    double norm2 = 0.0;
    for (const auto& grad : gradients) {
        norm2 += grad.norm() * grad.norm();
    }
    return std::sqrt(norm2);
}

void apply_periodic_boundaries(std::vector<Rod>& rods, double box_size) {
    double half_box = box_size * 0.5;
    for (auto& rod : rods) {
        rod.center.x = rod.center.x - box_size * std::round(rod.center.x / box_size);
        rod.center.y = rod.center.y - box_size * std::round(rod.center.y / box_size);
        rod.center.z = rod.center.z - box_size * std::round(rod.center.z / box_size);
    }
}

std::vector<Rod> generate_random_configuration(int num_rods, double box_size, 
                                               double rod_length, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pos_dist(-box_size/2, box_size/2);
    std::uniform_real_distribution<double> phi_dist(0, M_PI);
    std::uniform_real_distribution<double> theta_dist(0, 2*M_PI);
    
    std::vector<Rod> rods;
    rods.reserve(num_rods);
    
    for (int i = 0; i < num_rods; ++i) {
        Vec3 center{pos_dist(rng), pos_dist(rng), pos_dist(rng)};
        double phi = phi_dist(rng);
        double theta = theta_dist(rng);
        
        rods.emplace_back(center, phi, theta, rod_length);
    }
    
    return rods;
}

void resolve_overlaps(std::vector<Rod>& rods, double min_distance) {
    const int max_attempts = 1000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> offset_dist(-0.1, 0.1);
    
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        bool found_overlap = false;
        
        for (size_t i = 0; i < rods.size() && !found_overlap; ++i) {
            for (size_t j = i + 1; j < rods.size(); ++j) {
                if (rod_distance(rods[i], rods[j]) < min_distance) {
                    // Move rods apart slightly
                    Vec3 offset{offset_dist(rng), offset_dist(rng), offset_dist(rng)};
                    rods[i].center += offset;
                    rods[j].center -= offset;
                    found_overlap = true;
                    break;
                }
            }
        }
        
        if (!found_overlap) break;
    }
}

} // namespace entanglement