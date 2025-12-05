/**
 * @file simple_two_rod_test.cpp
 * @brief Simple OpenGL test with just two static rods - no dynamics, no optimization
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>

// Project headers
#include "visualization.hpp"
#include "system.hpp"

using namespace entanglement;

int main() {
    std::cout << "Simple Two Rod OpenGL Test\n"
              << "==========================\n"
              << "This demo shows two static rods with no dynamics or optimization.\n\n";

    // Create a simple rod system with just two rods
    SystemParameters params;
    params.collision_radius    = 0.02;    // Small radius for visualization
    params.harmonic_amplitude  = 0.0;     // No forces
    params.entanglement_weight = 0.0;     // No entanglement forces
    params.box_size            = 0.0;     // No container

    RodSystem system(params);

    // Add two rods in interesting positions
    // Rod 1: Horizontal rod at origin
    Rod rod1;
    rod1.center = {0.0, 0.0, 0.0};
    rod1.phi    = M_PI / 2.0;  // Horizontal orientation
    rod1.theta  = 0.0;         // Along x-axis
    rod1.length = 0.8;         // Shorter for better visibility

    // Rod 2: Angled rod offset and rotated to cross rod1
    Rod rod2;
    rod2.center = {0.0, 0.0, 0.0};  // Same center for crossing effect
    rod2.phi    = M_PI / 2.0;       // Also horizontal
    rod2.theta  = M_PI / 2.0;       // Perpendicular to rod1
    rod2.length = 0.8;              // Same length

    system.add_rod(rod1);
    system.add_rod(rod2);

    std::cout << "Created two static rods:\n"
              << "  Rod 1: center=(0,0,0), phi=π/2, theta=0 (horizontal along X)\n"
              << "  Rod 2: center=(0,0,0), phi=π/2, theta=π/2 (horizontal along Y, crossing Rod 1)\n\n";

    // Create visualizer
    RodVisualizer visualizer(1000, 800, "Simple Two Rod Test");

    // Configure visualization settings
    RodVisualizer::ViewSettings view;
    view.camera_distance   = 2.5f;       // Closer camera for better visibility
    view.rod_radius        = 0.03f;      // Thicker rods for better visibility  
    view.show_axes         = true;       // Show coordinate axes
    view.show_contacts     = false;      // No contact visualization needed
    view.show_container    = false;      // No container
    view.rod_color[0]      = 0.8f;       // Red component
    view.rod_color[1]      = 0.2f;       // Green component  
    view.rod_color[2]      = 0.2f;       // Blue component (reddish rods)
    view.background_color[0] = 0.05f;    // Darker background for contrast
    view.background_color[1] = 0.05f;
    view.background_color[2] = 0.15f;
    visualizer.set_view_settings(view);

    // Enable auto-rotation for better viewing
    RodVisualizer::AnimationSettings anim;
    anim.auto_rotate    = true;
    anim.rotation_speed = 0.3f;  // Slow rotation
    visualizer.set_animation_settings(anim);

    std::cout << "Controls:\n"
              << "  Mouse: Rotate view manually\n"
              << "  Scroll: Zoom in/out\n"
              << "  'A': Toggle coordinate axes\n"
              << "  'R': Toggle auto-rotation\n"
              << "  'S': Save screenshot\n"
              << "  ESC or close window: Exit\n\n";

    std::cout << "Displaying two static rods...\n";

    // Get the rod configuration
    auto rods = system.rods();

    // Main rendering loop - just display the static rods
    while (!visualizer.should_close()) {
        // Render the static rod configuration
        visualizer.render(rods, params.box_size);
        visualizer.update();
        
        // Limit frame rate to ~60 FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    std::cout << "Test completed. Window closed.\n";
    return 0;
}