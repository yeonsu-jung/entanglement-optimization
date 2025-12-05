/**
 * @file system.cpp
 * @brief Implementation of system management
 */

#include "system.hpp"
#include "distance.hpp"
#include "entanglement.hpp"
#include "gradients.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace entanglement {

void RodSystem::setup_threading() const {
#ifdef USE_OPENMP
    if (params_.num_threads > 0) {
        omp_set_num_threads(params_.num_threads);
    }
#endif
}

double RodSystem::total_energy() const {
    energy_evaluations_++;
    setup_threading();
    
    double energy = 0.0;
    const size_t n = rods_.size();
    
#ifdef USE_OPENMP
    #pragma omp parallel for reduction(+:energy) schedule(dynamic)
#endif
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            energy += pairwise_energy(i, j);
        }
    }
    
    return energy;
}

double RodSystem::harmonic_energy() const {
    double energy = 0.0;
    const size_t n = rods_.size();
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double distance = (params_.box_size > 0) ? 
                rod_distance_pbc(rods_[i], rods_[j], params_.box_size) :
                rod_distance(rods_[i], rods_[j]);
            
            double delta = distance - params_.collision_radius;
            energy += params_.harmonic_amplitude * delta * delta;
        }
    }
    
    return energy;
}

double RodSystem::entanglement_energy() const {
    double energy = 0.0;
    const size_t n = rods_.size();
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            energy += params_.entanglement_weight * 
                     rod_linking_number(rods_[i], rods_[j], params_.use_arai_formula);
        }
    }
    
    return energy;
}

std::vector<Gradient5D> RodSystem::total_gradients() const {
    gradient_evaluations_++;
    setup_threading();
    
    const size_t n = rods_.size();
    std::vector<Gradient5D> gradients(n);
    
    // Initialize to zero
    for (auto& grad : gradients) {
        grad.zero();
    }
    
#ifdef USE_OPENMP
    #pragma omp parallel
    {
        std::vector<Gradient5D> local_gradients(n);
        for (auto& grad : local_gradients) grad.zero();
        
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                auto pair_grads = pairwise_gradients(i, j);
                local_gradients[i] = local_gradients[i] + pair_grads.grad_rod1;
                local_gradients[j] = local_gradients[j] + pair_grads.grad_rod2;
            }
        }
        
        #pragma omp critical
        {
            for (size_t i = 0; i < n; ++i) {
                gradients[i] = gradients[i] + local_gradients[i];
            }
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            auto pair_grads = pairwise_gradients(i, j);
            gradients[i] = gradients[i] + pair_grads.grad_rod1;
            gradients[j] = gradients[j] + pair_grads.grad_rod2;
        }
    }
#endif
    
    return gradients;
}

std::vector<Gradient5D> RodSystem::harmonic_gradients() const {
    const size_t n = rods_.size();
    std::vector<Gradient5D> gradients(n);
    
    for (auto& grad : gradients) grad.zero();
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            auto pair_grads = harmonic_gradient(rods_[i], rods_[j], 
                                               params_.collision_radius, 
                                               params_.harmonic_amplitude);
            gradients[i] = gradients[i] + pair_grads.grad_rod1;
            gradients[j] = gradients[j] + pair_grads.grad_rod2;
        }
    }
    
    return gradients;
}

std::vector<Gradient5D> RodSystem::entanglement_gradients() const {
    const size_t n = rods_.size();
    std::vector<Gradient5D> gradients(n);
    
    for (auto& grad : gradients) grad.zero();
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            auto pair_grads = linking_gradient(rods_[i], rods_[j], 1e-8, params_.use_arai_formula);
            
            gradients[i].dx += params_.entanglement_weight * pair_grads.grad_rod1.dx;
            gradients[i].dy += params_.entanglement_weight * pair_grads.grad_rod1.dy;
            gradients[i].dz += params_.entanglement_weight * pair_grads.grad_rod1.dz;
            gradients[i].dphi += params_.entanglement_weight * pair_grads.grad_rod1.dphi;
            gradients[i].dtheta += params_.entanglement_weight * pair_grads.grad_rod1.dtheta;
            
            gradients[j].dx += params_.entanglement_weight * pair_grads.grad_rod2.dx;
            gradients[j].dy += params_.entanglement_weight * pair_grads.grad_rod2.dy;
            gradients[j].dz += params_.entanglement_weight * pair_grads.grad_rod2.dz;
            gradients[j].dphi += params_.entanglement_weight * pair_grads.grad_rod2.dphi;
            gradients[j].dtheta += params_.entanglement_weight * pair_grads.grad_rod2.dtheta;
        }
    }
    
    return gradients;
}

double RodSystem::pairwise_energy(size_t i, size_t j) const {
    double distance = (params_.box_size > 0) ? 
        rod_distance_pbc(rods_[i], rods_[j], params_.box_size) :
        rod_distance(rods_[i], rods_[j]);
    
    double delta = distance - params_.collision_radius;
    double harmonic = params_.harmonic_amplitude * delta * delta;
    
    double entanglement = params_.entanglement_weight * 
                         rod_linking_number(rods_[i], rods_[j], params_.use_arai_formula);
    
    return harmonic + entanglement;
}

RodPairGradients RodSystem::pairwise_gradients(size_t i, size_t j) const {
    return effective_gradient(rods_[i], rods_[j], 
                             params_.collision_radius, 
                             params_.harmonic_amplitude);
}

std::vector<double> RodSystem::all_distances() const {
    std::vector<double> distances;
    const size_t n = rods_.size();
    distances.reserve(n * (n-1) / 2);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double dist = (params_.box_size > 0) ? 
                rod_distance_pbc(rods_[i], rods_[j], params_.box_size) :
                rod_distance(rods_[i], rods_[j]);
            distances.push_back(dist);
        }
    }
    
    return distances;
}

std::vector<double> RodSystem::all_linking_numbers() const {
    std::vector<double> linking_numbers;
    const size_t n = rods_.size();
    linking_numbers.reserve(n * (n-1) / 2);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double link = rod_linking_number(rods_[i], rods_[j], params_.use_arai_formula);
            linking_numbers.push_back(link);
        }
    }
    
    return linking_numbers;
}

std::vector<double> RodSystem::all_angles() const {
    std::vector<double> angles;
    const size_t n = rods_.size();
    angles.reserve(n * (n-1) / 2);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            angles.push_back(rod_angle(rods_[i], rods_[j]));
        }
    }
    
    return angles;
}

bool RodSystem::check_overlaps(double threshold) const {
    if (threshold < 0) threshold = params_.collision_radius;
    
    const size_t n = rods_.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double dist = rod_distance(rods_[i], rods_[j]);
            if (dist < threshold) {
                return true;
            }
        }
    }
    return false;
}

void RodSystem::print_system_info() const {
    std::cout << "Rod System Information:" << std::endl;
    std::cout << "  Number of rods: " << rods_.size() << std::endl;
    std::cout << "  Collision radius: " << params_.collision_radius << std::endl;
    std::cout << "  Harmonic amplitude: " << params_.harmonic_amplitude << std::endl;
    std::cout << "  Box size: " << (params_.box_size > 0 ? std::to_string(params_.box_size) : "infinite") << std::endl;
    std::cout << "  Total energy: " << total_energy() << std::endl;
    std::cout << "  Energy evaluations: " << energy_evaluations_ << std::endl;
    std::cout << "  Gradient evaluations: " << gradient_evaluations_ << std::endl;
}

void RodSystem::save_configuration(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    file << "# Rod configuration: x y z phi theta length\n";
    for (const auto& rod : rods_) {
        file << rod.center.x << " " << rod.center.y << " " << rod.center.z << " "
             << rod.phi << " " << rod.theta << " " << rod.length << "\n";
    }
}

void RodSystem::load_configuration(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }
    
    rods_.clear();
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        double x, y, z, phi, theta, length;
        
        if (iss >> x >> y >> z >> phi >> theta >> length) {
            rods_.emplace_back(Vec3{x, y, z}, phi, theta, length);
        }
    }
}

// FastPacker implementation
FastPacker::Result FastPacker::generate_packing(int target_rods) {
    Result result;
    result.attempted = 0;
    result.placed = 0;
    result.rods.clear();
    
    std::mt19937 gen(params_.seed);
    std::uniform_real_distribution<double> pos_dist(-params_.container_size, params_.container_size);
    std::uniform_real_distribution<double> phi_dist(0.0, M_PI);
    std::uniform_real_distribution<double> theta_dist(0.0, 2.0 * M_PI);
    
    const double min_distance = params_.rod_diameter;
    
    for (int attempt = 0; attempt < params_.max_attempts && result.placed < target_rods; ++attempt) {
        result.attempted++;
        
        // Generate random rod
        Vec3 center{pos_dist(gen), pos_dist(gen), pos_dist(gen)};
        double phi = phi_dist(gen);
        double theta = theta_dist(gen);
        Rod candidate(center, phi, theta, params_.rod_length);
        
        // Check for overlaps
        bool valid = true;
        for (const auto& existing : result.rods) {
            if (rod_distance(candidate, existing) < min_distance) {
                valid = false;
                break;
            }
        }
        
        if (valid) {
            result.rods.push_back(candidate);
            result.placed++;
        }
    }
    
    // Calculate packing fraction
    double rod_volume = M_PI * std::pow(params_.rod_diameter / 2.0, 2) * params_.rod_length;
    double container_volume = std::pow(2.0 * params_.container_size, 3);
    result.packing_fraction = (result.placed * rod_volume) / container_volume;
    
    return result;
}

} // namespace entanglement