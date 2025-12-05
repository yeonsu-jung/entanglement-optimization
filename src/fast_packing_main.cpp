/**
 * @file fast_packing_main.cpp
 * @brief Fast rod packing application - C++ equivalent of the Python fast_packing.py
 */

#include "system.hpp"
#include "optimization.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>

using namespace entanglement;

int main(int argc, char* argv[]) {
    
    // Default parameters (equivalent to Python version)
    double container_size = 3.0;     // C parameter
    double rod_length = 1.0;
    double alpha = 100.0;             // aspect ratio
    int n_points = 30;
    int max_attempts = 1000000;
    int seed = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--container-size" && i+1 < argc) {
            container_size = std::stod(argv[++i]);
        } else if (arg == "--rod-length" && i+1 < argc) {
            rod_length = std::stod(argv[++i]);
        } else if (arg == "--alpha" && i+1 < argc) {
            alpha = std::stod(argv[++i]);
        } else if (arg == "--n-points" && i+1 < argc) {
            n_points = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i+1 < argc) {
            seed = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --container-size C   Half-size of container [-C,C]^3 (default: 3.0)\n"
                      << "  --rod-length L       Rod length (default: 1.0)\n"
                      << "  --alpha A           Aspect ratio L/D (default: 100.0)\n"
                      << "  --n-points N        Number of density points to test (default: 30)\n"
                      << "  --seed S            Random seed (default: 0)\n"
                      << "  --help              Show this help\n";
            return 0;
        }
    }
    
    double rod_diameter = rod_length / alpha;
    
    std::cout << "Fast Rod Packing - C++ Implementation" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "Container size: " << container_size << std::endl;
    std::cout << "Rod length: " << rod_length << std::endl;
    std::cout << "Rod diameter: " << rod_diameter << std::endl;
    std::cout << "Aspect ratio: " << alpha << std::endl;
    std::cout << "Random seed: " << seed << std::endl;
    std::cout << std::endl;
    
    // Estimate maximum number of rods (similar to Python version)
    int N_max_est = static_cast<int>((2*container_size) * (2*container_size) * (2*container_size) / 
                                    (rod_diameter * rod_length * rod_length) * 10);
    
    // Generate geometric series of rod counts (reverse order like Python)
    std::vector<int> NN;
    double ratio = std::pow(static_cast<double>(N_max_est) / 10.0, 1.0 / (n_points - 1));
    for (int i = 0; i < n_points; ++i) {
        int N = static_cast<int>(10.0 * std::pow(ratio, n_points - 1 - i));
        NN.push_back(N);
    }
    
    std::cout << "Testing " << n_points << " points from N=" << NN.back() 
              << " to N=" << NN.front() << std::endl;
    std::cout << "Estimated max N: " << N_max_est << std::endl;
    std::cout << std::endl;
    
    // Setup fast packer
    FastPacker::Parameters pack_params;
    pack_params.container_size = container_size;
    pack_params.rod_length = rod_length;
    pack_params.rod_diameter = rod_diameter;
    pack_params.max_attempts = max_attempts;
    pack_params.seed = seed;
    
    FastPacker packer(pack_params);
    
    // Results storage
    std::vector<int> attempts_list;
    std::vector<std::vector<Rod>> placed_rods;
    std::vector<double> timing_data;
    
    // Run packing experiments
    for (int N : NN) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto result = packer.generate_packing(N);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double seconds = duration.count() / 1000.0;
        
        double us_per_rod = (result.placed > 0) ? (seconds * 1e6 / result.placed) : 0.0;
        
        std::cout << "N=" << std::setw(6) << N 
                  << " placed=" << std::setw(6) << result.placed
                  << " time=" << std::fixed << std::setprecision(3) << seconds << " sec"
                  << " (" << std::setprecision(1) << us_per_rod << " µs/rod)" 
                  << std::endl;
        
        attempts_list.push_back(result.total_attempts);
        placed_rods.push_back(result.rods);
        timing_data.push_back(seconds);
    }
    
    // Final verification on last run
    if (!placed_rods.empty() && !placed_rods.back().empty()) {
        const auto& final_rods = placed_rods.back();
        bool inside = true;
        
        for (const auto& rod : final_rods) {
            if (rod.center.x < -container_size || rod.center.x > container_size ||
                rod.center.y < -container_size || rod.center.y > container_size ||
                rod.center.z < -container_size || rod.center.z > container_size) {
                inside = false;
                break;
            }
        }
        
        std::cout << "\nFinal verification:" << std::endl;
        std::cout << "Centroids inside box: " << (inside ? "true" : "false") << std::endl;
        
        // Check distances
        SystemParameters sys_params;
        sys_params.collision_radius = rod_diameter;
        
        RodSystem system(sys_params);
        system.set_rods(final_rods);
        
        auto distances = system.all_distances();
        auto min_dist = *std::min_element(distances.begin(), distances.end());
        auto violations = std::count_if(distances.begin(), distances.end(), 
                                       [rod_diameter](double d) { return d < rod_diameter; });
        
        std::cout << "Minimum distance: " << std::scientific << std::setprecision(6) << min_dist << std::endl;
        std::cout << "Gap to diameter: " << (min_dist - rod_diameter) << std::endl;
        std::cout << "Violations (< diameter): " << violations << " (should be 0)" << std::endl;
    }
    
    // Save results
    std::cout << "\nSaving results..." << std::endl;
    
    // Save attempts vs N data
    std::ofstream attempts_file("attempts_vs_N_cpp.txt");
    attempts_file << "# N Attempts Time_sec\n";
    for (size_t i = 0; i < NN.size(); ++i) {
        attempts_file << NN[i] << " " << attempts_list[i] << " " << timing_data[i] << "\n";
    }
    attempts_file.close();
    
    // Save final configuration
    if (!placed_rods.empty() && !placed_rods.back().empty()) {
        RodSystem system;
        system.set_rods(placed_rods.back());
        system.save_configuration("final_configuration_cpp.txt");
    }
    
    std::cout << "Results saved to attempts_vs_N_cpp.txt and final_configuration_cpp.txt" << std::endl;
    std::cout << "Fast packing benchmark completed!" << std::endl;
    
    return 0;
}