/**
 * @file optimization_demo.cpp
 * @brief Demonstration of optimization algorithms on rod systems
 */

#include "system.hpp"
#include "optimization.hpp"
#include "gradients.hpp"
#include <iostream>
#include <iomanip>

using namespace entanglement;

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

void demo_small_system() {
    print_header("Small System Optimization Demo");
    
    // Create a small system with potential overlaps
    SystemParameters params;
    params.collision_radius = 0.05;
    params.harmonic_amplitude = 1000.0;
    params.entanglement_weight = 1.0;
    
    RodSystem system(params);
    
    // Add some rods that are too close
    system.add_rod(Rod({0.0, 0.0, 0.0}, M_PI/4, 0.0, 1.0));
    system.add_rod(Rod({0.03, 0.0, 0.0}, M_PI/3, M_PI/4, 1.0));
    system.add_rod(Rod({0.0, 0.03, 0.0}, M_PI/2, M_PI/2, 1.0));
    
    std::cout << "Initial configuration:" << std::endl;
    system.print_system_info();
    
    auto distances = system.all_distances();
    std::cout << "Initial distances: ";
    for (double d : distances) {
        std::cout << std::fixed << std::setprecision(4) << d << " ";
    }
    std::cout << std::endl;
    
    // Optimize with FIRE algorithm
    FireOptimizer::Parameters fire_params;
    fire_params.max_iter = 1000;
    fire_params.force_tol = 1e-5;
    fire_params.verbose = true;
    
    FireOptimizer optimizer(fire_params);
    
    auto energy_func = [&system](const std::vector<Rod>& rods) {
        system.set_rods(rods);
        return system.total_energy();
    };
    
    auto gradient_func = [&system](const std::vector<Rod>& rods) {
        system.set_rods(rods);
        return system.total_gradients();
    };
    
    auto rods = system.rods();  // Copy for optimization
    auto result = optimizer.optimize(rods, energy_func, gradient_func);
    
    std::cout << "\nOptimization result:" << std::endl;
    std::cout << "Converged: " << (result.converged ? "Yes" : "No") << std::endl;
    std::cout << "Iterations: " << result.iterations << std::endl;
    std::cout << "Final energy: " << result.final_energy << std::endl;
    std::cout << "Final force norm: " << result.final_force_norm << std::endl;
    
    // Update system with optimized configuration
    system.set_rods(rods);
    auto final_distances = system.all_distances();
    std::cout << "Final distances: ";
    for (double d : final_distances) {
        std::cout << std::fixed << std::setprecision(4) << d << " ";
    }
    std::cout << std::endl;
}

void demo_random_packing_optimization() {
    print_header("Random Packing + Optimization Demo");
    
    // Generate initial random packing
    FastPacker::Parameters pack_params;
    pack_params.container_size = 2.0;
    pack_params.rod_length = 1.0;
    pack_params.rod_diameter = 0.02;
    pack_params.max_attempts = 100000;
    pack_params.seed = 42;
    
    FastPacker packer(pack_params);
    auto pack_result = packer.generate_packing(50);
    
    std::cout << "Initial packing:" << std::endl;
    std::cout << "Attempted: " << pack_result.attempted << std::endl;
    std::cout << "Placed: " << pack_result.placed << std::endl;
    std::cout << "Packing fraction: " << pack_result.packing_fraction << std::endl;
    
    if (pack_result.placed < 5) {
        std::cout << "Too few rods placed for meaningful optimization demo." << std::endl;
        return;
    }
    
    // Setup system for optimization
    SystemParameters sys_params;
    sys_params.collision_radius = pack_params.rod_diameter;
    sys_params.harmonic_amplitude = 500.0;
    sys_params.entanglement_weight = 0.1;  // Smaller weight for large system
    
    RodSystem system(sys_params);
    system.set_rods(pack_result.rods);
    
    std::cout << "\nInitial system energy: " << system.total_energy() << std::endl;
    
    // Optimize with gradient descent (faster for larger systems)
    GradientDescentOptimizer::Parameters gd_params;
    gd_params.learning_rate = 0.001;
    gd_params.max_iter = 500;
    gd_params.force_tol = 1e-4;
    gd_params.adaptive_rate = true;
    gd_params.verbose = true;
    
    GradientDescentOptimizer optimizer(gd_params);
    
    auto energy_func = [&system](const std::vector<Rod>& rods) {
        system.set_rods(rods);
        return system.total_energy();
    };
    
    auto gradient_func = [&system](const std::vector<Rod>& rods) {
        system.set_rods(rods);
        return system.total_gradients();
    };
    
    auto rods = system.rods();  // Copy for optimization
    auto result = optimizer.optimize(rods, energy_func, gradient_func);
    
    std::cout << "\nOptimization result:" << std::endl;
    std::cout << "Converged: " << (result.converged ? "Yes" : "No") << std::endl;
    std::cout << "Iterations: " << result.iterations << std::endl;
    std::cout << "Final energy: " << result.final_energy << std::endl;
    std::cout << "Final force norm: " << result.final_force_norm << std::endl;
    
    // Analysis
    system.set_rods(rods);
    auto linking_numbers = system.all_linking_numbers();
    auto entangled_pairs = std::count_if(linking_numbers.begin(), linking_numbers.end(),
                                        [](double lk) { return std::abs(lk) > 1e-6; });
    
    std::cout << "\nFinal analysis:" << std::endl;
    std::cout << "Total pairs: " << linking_numbers.size() << std::endl;
    std::cout << "Entangled pairs: " << entangled_pairs << std::endl;
    std::cout << "System evaluations: " << system.energy_evaluations() << " energy, " 
              << system.gradient_evaluations() << " gradient" << std::endl;
}

void demo_gradient_validation() {
    print_header("Gradient Validation Demo");
    
    // Test gradients on a simple pair
    Rod rod1({0.1, 0.0, 0.0}, M_PI/4, 0.0, 1.0);
    Rod rod2({0.2, 0.1, 0.0}, M_PI/3, M_PI/4, 1.0);
    
    std::cout << "Testing distance gradients..." << std::endl;
    
    auto dist_grad = distance_gradient(rod1, rod2);
    bool dist_valid = validate_gradients(rod1, rod2, dist_grad, 1e-8, 1e-6);
    
    std::cout << "Distance gradient validation: " << (dist_valid ? "PASSED" : "FAILED") << std::endl;
    
    std::cout << "Distance grad rod1: (" 
              << std::scientific << std::setprecision(6)
              << dist_grad.grad_rod1.dx << ", " << dist_grad.grad_rod1.dy << ", " 
              << dist_grad.grad_rod1.dz << ", " << dist_grad.grad_rod1.dphi << ", " 
              << dist_grad.grad_rod1.dtheta << ")" << std::endl;
    
    std::cout << "\nTesting linking number gradients (finite difference)..." << std::endl;
    
    auto link_grad = linking_gradient(rod1, rod2, 1e-8, false);
    std::cout << "Linking grad rod1: (" 
              << link_grad.grad_rod1.dx << ", " << link_grad.grad_rod1.dy << ", " 
              << link_grad.grad_rod1.dz << ", " << link_grad.grad_rod1.dphi << ", " 
              << link_grad.grad_rod1.dtheta << ")" << std::endl;
    
    // Test consistency with different step sizes
    auto link_grad_large = linking_gradient(rod1, rod2, 1e-7, false);
    double consistency_error = 0.0;
    consistency_error += std::abs(link_grad.grad_rod1.dx - link_grad_large.grad_rod1.dx);
    consistency_error += std::abs(link_grad.grad_rod1.dy - link_grad_large.grad_rod1.dy);
    consistency_error += std::abs(link_grad.grad_rod1.dz - link_grad_large.grad_rod1.dz);
    
    std::cout << "Linking gradient consistency error: " << consistency_error << std::endl;
    std::cout << "Consistency: " << (consistency_error < 1e-4 ? "GOOD" : "POOR") << std::endl;
}

int main(int argc, char* argv[]) {
    
    std::cout << "Entanglement Optimization - Demo Application" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Check command line arguments
    bool run_all = (argc == 1);
    bool run_small = run_all;
    bool run_packing = run_all;
    bool run_validation = run_all;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--small") run_small = true;
        else if (arg == "--packing") run_packing = true;
        else if (arg == "--validation") run_validation = true;
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --small       Run small system optimization demo\n"
                      << "  --packing     Run random packing + optimization demo\n"
                      << "  --validation  Run gradient validation demo\n"
                      << "  --help        Show this help\n"
                      << "\nIf no options given, runs all demos.\n";
            return 0;
        }
    }
    
    try {
        if (run_small) {
            demo_small_system();
        }
        
        if (run_packing) {
            demo_random_packing_optimization();
        }
        
        if (run_validation) {
            demo_gradient_validation();
        }
        
        std::cout << "\nDemo completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}