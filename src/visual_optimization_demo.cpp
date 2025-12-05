/**
 * @file visual_optimization_demo.cpp
 * @brief Interactive visualization of rod optimization
 */

#include "visualization.hpp"
#include "system.hpp"
#include "optimization.hpp"
#include <iostream>
#include <chrono>
#include <thread>

using namespace entanglement;

void demo_visual_small_system() {
    std::cout << "Starting visual small system optimization..." << std::endl;
    
    // Create a small system with overlapping rods
    SystemParameters params;
    params.collision_radius = 0.05;
    params.harmonic_amplitude = 1000.0;
    params.entanglement_weight = 1.0;
    
    RodSystem system(params);
    
    // Add overlapping rods
    system.add_rod(Rod({0.0, 0.0, 0.0}, M_PI/4, 0.0, 1.0));
    system.add_rod(Rod({0.03, 0.0, 0.0}, M_PI/3, M_PI/4, 1.0));
    system.add_rod(Rod({0.0, 0.03, 0.0}, M_PI/2, M_PI/2, 1.0));
    
    std::cout << "Initial energy: " << system.total_energy() << std::endl;
    
    // Setup FIRE optimizer
    FireOptimizer::Parameters fire_params;
    fire_params.max_iter = 2000;
    fire_params.force_tol = 1e-5;
    fire_params.verbose = false;  // Don't print to console during visualization
    
    FireOptimizer optimizer(fire_params);
    
    // Create visualizer
    RodVisualizer visualizer(1200, 900, "Rod Optimization - Small System");
    
    // Setup visualization settings
    RodVisualizer::ViewSettings view_settings;
    view_settings.camera_distance = 3.0f;
    view_settings.rod_radius = 0.02f;
    view_settings.show_axes = true;
    view_settings.show_contacts = true;
    view_settings.contact_threshold = 0.1f;
    visualizer.set_view_settings(view_settings);
    
    // Optimization state
    auto rods = system.rods();
    int iteration = 0;
    double last_energy = system.total_energy();
    bool converged = false;
    
    std::cout << "Controls:" << std::endl;
    std::cout << "  Mouse: Rotate view" << std::endl;
    std::cout << "  Scroll: Zoom" << std::endl;
    std::cout << "  'A': Toggle axes" << std::endl;
    std::cout << "  'R': Toggle auto-rotation" << std::endl;
    std::cout << "  'S': Save screenshot" << std::endl;
    std::cout << "  ESC: Exit" << std::endl;
    
    // Visualization loop with optimization
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_frame_time = start_time;
    const double frame_time = 1.0 / 60.0; // 60 FPS limit
    
    while (!visualizer.should_close() && !converged && iteration < fire_params.max_iter) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(current_time - last_frame_time).count();
        
        // Perform multiple optimization steps per frame for better performance
        for (int step = 0; step < 5 && !converged && iteration < fire_params.max_iter; ++step) {
            // Perform one optimization step
            auto energy_func = [&system](const std::vector<Rod>& rods) {
                system.set_rods(rods);
                return system.total_energy();
            };
            
            auto gradient_func = [&system](const std::vector<Rod>& rods) {
                system.set_rods(rods);
                return system.total_gradients();
            };
            
            // Single step optimization
            double current_energy = energy_func(rods);
            auto gradients = gradient_func(rods);
            
            // Calculate force norm
            double force_norm = 0.0;
            for (const auto& grad : gradients) {
                force_norm += grad.norm() * grad.norm();
            }
            force_norm = sqrt(force_norm);
            
            // Check convergence
            if (force_norm < fire_params.force_tol) {
                converged = true;
                break;
            }
            
            // Apply FIRE step (simplified)
            double dt = 0.01;
            for (size_t i = 0; i < rods.size(); ++i) {
                // Simple gradient descent step
                rods[i].center.x -= dt * gradients[i].dx;
                rods[i].center.y -= dt * gradients[i].dy;
                rods[i].center.z -= dt * gradients[i].dz;
                rods[i].phi -= dt * gradients[i].dphi;
                rods[i].theta -= dt * gradients[i].dtheta;
                
                // Keep angles in valid range
                rods[i].phi = std::max(0.0, std::min(M_PI, rods[i].phi));
                while (rods[i].theta < 0) rods[i].theta += 2*M_PI;
                while (rods[i].theta > 2*M_PI) rods[i].theta -= 2*M_PI;
            }
            
            iteration++;
            last_energy = current_energy;
            
            // Print progress every 100 iterations
            if (iteration % 100 == 0) {
                std::cout << "Iteration " << iteration << ": Energy = " << current_energy 
                          << ", Force norm = " << force_norm << std::endl;
            }
        }
        
        // Update system
        system.set_rods(rods);
        
        // Render current state at frame rate limit
        if (elapsed >= frame_time) {
            visualizer.render(rods);
            visualizer.update();
            last_frame_time = current_time;
        } else {
            // Just poll events to keep window responsive
            visualizer.update();
            // Small sleep to prevent 100% CPU usage
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nOptimization completed!" << std::endl;
    std::cout << "Final iteration: " << iteration << std::endl;
    std::cout << "Converged: " << (converged ? "Yes" : "No") << std::endl;
    std::cout << "Final energy: " << last_energy << std::endl;
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "Press any key in visualization window to continue..." << std::endl;
    
    // Keep visualization open until user closes
    while (!visualizer.should_close()) {
        visualizer.render(rods);
        visualizer.update();
    }
}

void demo_visual_packing_optimization() {
    std::cout << "Starting visual packing optimization..." << std::endl;
    
    // Generate initial random packing
    FastPacker::Parameters pack_params;
    pack_params.container_size = 2.0;
    pack_params.rod_length = 1.0;
    pack_params.rod_diameter = 0.03;
    pack_params.max_attempts = 50000;
    pack_params.seed = 42;
    
    FastPacker packer(pack_params);
    auto pack_result = packer.generate_packing(20);
    
    std::cout << "Initial packing: " << pack_result.placed << "/" << pack_result.attempted 
              << " rods placed (packing fraction: " << pack_result.packing_fraction << ")" << std::endl;
    
    if (pack_result.placed < 5) {
        std::cout << "Too few rods placed. Try increasing container size or reducing rod diameter." << std::endl;
        return;
    }
    
    // Setup system for optimization
    SystemParameters sys_params;
    sys_params.collision_radius = pack_params.rod_diameter;
    sys_params.harmonic_amplitude = 500.0;
    sys_params.entanglement_weight = 0.1;
    sys_params.box_size = pack_params.container_size * 2.0;  // Periodic boundaries
    
    RodSystem system(sys_params);
    system.set_rods(pack_result.rods);
    
    std::cout << "Initial system energy: " << system.total_energy() << std::endl;
    
    // Create visualizer
    RodVisualizer visualizer(1400, 1000, "Rod Optimization - Packing System");
    
    // Setup visualization settings
    RodVisualizer::ViewSettings view_settings;
    view_settings.camera_distance = 6.0f;
    view_settings.rod_radius = pack_params.rod_diameter / 2.0f;
    view_settings.show_axes = true;
    view_settings.show_container = true;
    view_settings.show_contacts = true;
    view_settings.contact_threshold = pack_params.rod_diameter * 1.2f;
    visualizer.set_view_settings(view_settings);
    
    // Auto-rotation for better viewing
    RodVisualizer::AnimationSettings anim_settings;
    anim_settings.auto_rotate = true;
    anim_settings.rotation_speed = 0.2f;
    visualizer.set_animation_settings(anim_settings);
    
    // Setup gradient descent optimizer
    GradientDescentOptimizer::Parameters gd_params;
    gd_params.learning_rate = 0.001;
    gd_params.max_iter = 3000;
    gd_params.force_tol = 1e-4;
    gd_params.adaptive_rate = true;
    gd_params.verbose = false;
    
    // Optimization state
    auto rods = system.rods();
    int iteration = 0;
    double last_energy = system.total_energy();
    bool converged = false;
    
    std::cout << "Starting optimization with " << pack_result.placed << " rods..." << std::endl;
    std::cout << "This may take a moment. Watch the rods move apart to resolve overlaps!" << std::endl;
    
    // Optimization visualization loop
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_frame_time = start_time;
    const double frame_time = 1.0 / 60.0; // 60 FPS limit
    
    while (!visualizer.should_close() && !converged && iteration < gd_params.max_iter) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(current_time - last_frame_time).count();
        
        // Perform optimization steps
        for (int step = 0; step < 5 && !converged; ++step) {  // Multiple steps per frame for speed
            auto energy_func = [&system](const std::vector<Rod>& rods) {
                system.set_rods(rods);
                return system.total_energy();
            };
            
            auto gradient_func = [&system](const std::vector<Rod>& rods) {
                system.set_rods(rods);
                return system.total_gradients();
            };
            
            double current_energy = energy_func(rods);
            auto gradients = gradient_func(rods);
            
            // Calculate force norm
            double force_norm = 0.0;
            for (const auto& grad : gradients) {
                force_norm += grad.norm() * grad.norm();
            }
            force_norm = sqrt(force_norm);
            
            // Check convergence
            if (force_norm < gd_params.force_tol) {
                converged = true;
                break;
            }
            
            // Apply gradient descent step
            double learning_rate = gd_params.learning_rate;
            for (size_t i = 0; i < rods.size(); ++i) {
                rods[i].center.x -= learning_rate * gradients[i].dx;
                rods[i].center.y -= learning_rate * gradients[i].dy;
                rods[i].center.z -= learning_rate * gradients[i].dz;
                rods[i].phi -= learning_rate * gradients[i].dphi;
                rods[i].theta -= learning_rate * gradients[i].dtheta;
                
                // Keep angles in valid range
                rods[i].phi = std::max(0.0, std::min(M_PI, rods[i].phi));
                while (rods[i].theta < 0) rods[i].theta += 2*M_PI;
                while (rods[i].theta > 2*M_PI) rods[i].theta -= 2*M_PI;
            }
            
            iteration++;
            last_energy = current_energy;
            
            // Print progress
            if (iteration % 200 == 0) {
                std::cout << "Iteration " << iteration << ": Energy = " << current_energy 
                          << ", Force norm = " << force_norm << std::endl;
            }
        }
        
        // Update system and render at frame rate limit
        system.set_rods(rods);
        if (elapsed >= frame_time) {
            visualizer.render(rods, sys_params.box_size);
            visualizer.update();
            last_frame_time = current_time;
        } else {
            // Just poll events to keep window responsive
            visualizer.update();
            // Small sleep to prevent 100% CPU usage
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nOptimization completed!" << std::endl;
    std::cout << "Final iteration: " << iteration << std::endl;
    std::cout << "Converged: " << (converged ? "Yes" : "No") << std::endl;
    std::cout << "Final energy: " << last_energy << std::endl;
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    
    // Analyze final state
    auto linking_numbers = system.all_linking_numbers();
    auto entangled_pairs = std::count_if(linking_numbers.begin(), linking_numbers.end(),
                                        [](double lk) { return std::abs(lk) > 1e-6; });
    
    std::cout << "Final analysis:" << std::endl;
    std::cout << "  Total pairs: " << linking_numbers.size() << std::endl;
    std::cout << "  Entangled pairs: " << entangled_pairs << std::endl;
    
    std::cout << "Visualization will continue. Press ESC to exit." << std::endl;
    
    // Keep visualization open
    while (!visualizer.should_close()) {
        visualizer.render(rods, sys_params.box_size);
        visualizer.update();
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Visual Rod Optimization Demo" << std::endl;
    std::cout << "============================" << std::endl;
    
    std::string demo_type = "small";
    if (argc > 1) {
        demo_type = argv[1];
    }
    
    try {
        if (demo_type == "small") {
            demo_visual_small_system();
        } else if (demo_type == "packing") {
            demo_visual_packing_optimization();
        } else {
            std::cout << "Usage: " << argv[0] << " [small|packing]" << std::endl;
            std::cout << "  small   - Small system with 3 overlapping rods" << std::endl;
            std::cout << "  packing - Random packing optimization" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}