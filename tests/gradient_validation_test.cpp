/**
 * @file gradient_validation_test.cpp
 * @brief Comprehensive test for gradient accuracy
 */

#include "gradients.hpp"
#include "entanglement.hpp"
#include "distance.hpp"
#include <iostream>
#include <iomanip>
#include <random>

using namespace entanglement;

void test_distance_gradients() {
    std::cout << "\n=== Distance Gradient Tests ===" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<> pos_dis(-1.0, 1.0);
    std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);
    
    int num_tests = 10;
    int passed = 0;
    
    for (int i = 0; i < num_tests; ++i) {
        Rod rod1({pos_dis(gen), pos_dis(gen), pos_dis(gen)}, 
                 angle_dis(gen), angle_dis(gen), 1.0);
        Rod rod2({pos_dis(gen), pos_dis(gen), pos_dis(gen)}, 
                 angle_dis(gen), angle_dis(gen), 1.0);
        
        auto grad = distance_gradient(rod1, rod2);
        bool valid = validate_gradients(rod1, rod2, grad, 1e-8, 1e-5);
        
        if (valid) {
            passed++;
        } else {
            std::cout << "Test " << i << " FAILED" << std::endl;
            std::cout << "  Rod1: (" << rod1.center.x << ", " << rod1.center.y 
                      << ", " << rod1.center.z << ", " << rod1.theta << ", " 
                      << rod1.phi << ")" << std::endl;
            std::cout << "  Rod2: (" << rod2.center.x << ", " << rod2.center.y 
                      << ", " << rod2.center.z << ", " << rod2.theta << ", " 
                      << rod2.phi << ")" << std::endl;
        }
    }
    
    std::cout << "Distance gradient tests: " << passed << "/" << num_tests 
              << " passed (" << (100.0 * passed / num_tests) << "%)" << std::endl;
}

void test_linking_gradient_consistency() {
    std::cout << "\n=== Linking Gradient Consistency Tests ===" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> pos_dis(-0.5, 0.5);
    std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);
    
    int num_tests = 5;
    std::vector<double> step_sizes = {1e-6, 1e-7, 1e-8, 1e-9};
    
    for (int i = 0; i < num_tests; ++i) {
        Rod rod1({pos_dis(gen), pos_dis(gen), pos_dis(gen)}, 
                 angle_dis(gen), angle_dis(gen), 1.0);
        Rod rod2({pos_dis(gen), pos_dis(gen), pos_dis(gen)}, 
                 angle_dis(gen), angle_dis(gen), 1.0);
        
        std::cout << "\nTest " << i + 1 << ":" << std::endl;
        
        std::vector<Gradient5D> gradients;
        for (double h : step_sizes) {
            auto grad = linking_gradient(rod1, rod2, h, false);
            gradients.push_back(grad.grad_rod1);
        }
        
        // Check consistency between different step sizes
        for (size_t j = 0; j < step_sizes.size(); ++j) {
            std::cout << "  h=" << std::scientific << step_sizes[j] 
                      << ": grad=(" << gradients[j].dx << ", " << gradients[j].dy 
                      << ", " << gradients[j].dz << ", " << gradients[j].dphi 
                      << ", " << gradients[j].dtheta << ")" << std::endl;
        }
        
        // Compare adjacent step sizes
        for (size_t j = 0; j < step_sizes.size() - 1; ++j) {
            double diff = std::abs(gradients[j].dx - gradients[j+1].dx) +
                         std::abs(gradients[j].dy - gradients[j+1].dy) +
                         std::abs(gradients[j].dz - gradients[j+1].dz);
            
            std::cout << "  Difference h=" << step_sizes[j] << " vs h=" 
                      << step_sizes[j+1] << ": " << diff << std::endl;
        }
    }
}

void test_special_cases() {
    std::cout << "\n=== Special Cases Tests ===" << std::endl;
    
    // Test parallel rods
    std::cout << "Testing parallel rods..." << std::endl;
    Rod rod1({0.0, 0.0, 0.0}, 0.0, 0.0, 1.0);
    Rod rod2({0.5, 0.0, 0.0}, 0.0, 0.0, 1.0);
    
    auto dist_grad = distance_gradient(rod1, rod2);
    bool valid = validate_gradients(rod1, rod2, dist_grad, 1e-8, 1e-5);
    std::cout << "Parallel rods distance gradient: " << (valid ? "VALID" : "INVALID") << std::endl;
    
    double linking = rod_linking_number(rod1, rod2, false);
    std::cout << "Parallel rods linking number: " << linking << std::endl;
    
    // Test perpendicular rods
    std::cout << "\nTesting perpendicular rods..." << std::endl;
    Rod rod3({0.0, 0.0, 0.0}, 0.0, 0.0, 1.0);
    Rod rod4({0.0, 0.5, 0.0}, M_PI/2, 0.0, 1.0);
    
    auto dist_grad2 = distance_gradient(rod3, rod4);
    bool valid2 = validate_gradients(rod3, rod4, dist_grad2, 1e-8, 1e-5);
    std::cout << "Perpendicular rods distance gradient: " << (valid2 ? "VALID" : "INVALID") << std::endl;
    
    double linking2 = rod_linking_number(rod3, rod4, false);
    std::cout << "Perpendicular rods linking number: " << linking2 << std::endl;
    
    // Test close rods (potential numerical issues)
    std::cout << "\nTesting very close rods..." << std::endl;
    Rod rod5({0.0, 0.0, 0.0}, 0.0, 0.0, 1.0);
    Rod rod6({0.001, 0.0, 0.0}, M_PI/4, M_PI/4, 1.0);
    
    double distance = rod_distance(rod5, rod6);
    std::cout << "Close rods distance: " << distance << std::endl;
    
    if (distance > 1e-10) {  // Avoid division by zero
        auto dist_grad3 = distance_gradient(rod5, rod6);
        bool valid3 = validate_gradients(rod5, rod6, dist_grad3, 1e-8, 1e-4);
        std::cout << "Close rods distance gradient: " << (valid3 ? "VALID" : "INVALID") << std::endl;
    } else {
        std::cout << "Rods too close for gradient test" << std::endl;
    }
}

void benchmark_gradient_computation() {
    std::cout << "\n=== Gradient Computation Benchmark ===" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> pos_dis(-1.0, 1.0);
    std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);
    
    int num_tests = 1000;
    
    // Generate test data
    std::vector<Rod> rods1, rods2;
    for (int i = 0; i < num_tests; ++i) {
        rods1.emplace_back(Vec3{pos_dis(gen), pos_dis(gen), pos_dis(gen)}, 
                           angle_dis(gen), angle_dis(gen), 1.0);
        rods2.emplace_back(Vec3{pos_dis(gen), pos_dis(gen), pos_dis(gen)}, 
                           angle_dis(gen), angle_dis(gen), 1.0);
    }
    
    // Benchmark distance gradients
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_tests; ++i) {
        auto grad = distance_gradient(rods1[i], rods2[i]);
        (void)grad;  // Suppress unused variable warning
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Distance gradients: " << num_tests << " evaluations in " 
              << duration.count() << " μs" << std::endl;
    std::cout << "Average: " << (duration.count() / double(num_tests)) 
              << " μs per evaluation" << std::endl;
    
    // Benchmark linking gradients (smaller sample due to cost)
    int link_tests = 100;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < link_tests; ++i) {
        auto grad = linking_gradient(rods1[i], rods2[i], 1e-8, false);
        (void)grad;
    }
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Linking gradients: " << link_tests << " evaluations in " 
              << duration.count() << " μs" << std::endl;
    std::cout << "Average: " << (duration.count() / double(link_tests)) 
              << " μs per evaluation" << std::endl;
}

int main() {
    std::cout << "Gradient Validation Test Suite" << std::endl;
    std::cout << "==============================" << std::endl;
    
    try {
        test_distance_gradients();
        test_linking_gradient_consistency();
        test_special_cases();
        benchmark_gradient_computation();
        
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "All tests completed. Check output above for any failures." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}