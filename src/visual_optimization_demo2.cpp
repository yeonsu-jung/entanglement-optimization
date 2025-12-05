// visual_optimization_demo.cpp
/**
 * @file visual_optimization_demo.cpp
 * @brief Interactive visualization of rod optimization (unified, line-search GD)
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>

// Project headers
#include "visualization.hpp"
#include "system.hpp"
#include "optimization.hpp" // optional

using namespace entanglement;

// ------------------------------
// Helpers
// ------------------------------

static inline double wrap_angle_0_2pi(double th) {
    th = std::fmod(th, 2.0 * M_PI);
    if (th < 0.0) th += 2.0 * M_PI;
    return th;
}

static inline void apply_descent_step(std::vector<Rod>& rods,
                                      const std::vector<Gradient5D>& G,
                                      double step)
{
    const size_t n = rods.size();
    for (size_t i = 0; i < n; ++i) {
        rods[i].center.x -= step * G[i].dx;
        rods[i].center.y -= step * G[i].dy;
        rods[i].center.z -= step * G[i].dz;
        rods[i].phi       = std::clamp(rods[i].phi   - step * G[i].dphi, 0.0, M_PI);
        rods[i].theta     = wrap_angle_0_2pi(rods[i].theta - step * G[i].dtheta);
    }
}

static inline double gradients_l2_sq(const std::vector<Gradient5D>& G) {
    double sum = 0.0;
    for (const auto& g : G) {
        sum += g.dx * g.dx + g.dy * g.dy + g.dz * g.dz
             + g.dphi * g.dphi + g.dtheta * g.dtheta;
    }
    return sum;
}

struct StepResult {
    bool   accepted;
    double energy;
    double force_norm;
};

static StepResult armijo_step(RodSystem& system,
                              std::vector<Rod>& rods,
                              double base_step,
                              double c1 = 1e-4,
                              int max_ls = 10,
                              double shrink = 0.5)
{
    system.set_rods(rods);
    const double E0 = system.total_energy();
    const auto   G  = system.total_gradients();

    const double g2 = gradients_l2_sq(G);
    const double gnorm = std::sqrt(g2);

    if (gnorm == 0.0) {
        return {true, E0, 0.0};
    }

    double step = base_step;
    for (int it = 0; it < max_ls; ++it) {
        auto trial = rods;
        apply_descent_step(trial, G, step);

        system.set_rods(trial);
        const double Et = system.total_energy();

        if (Et <= E0 - c1 * step * g2) { // Armijo
            rods = std::move(trial);
            return {true, Et, gnorm};
        }
        step *= shrink;
    }
    return {false, E0, gnorm};
}

static bool run_inner_steps(RodSystem& system,
                            std::vector<Rod>& rods,
                            int& iteration,
                            int max_iter,
                            int steps_per_frame,
                            double& last_energy,
                            double force_tol,
                            double& base_lr,
                            bool   use_adapt = true)
{
    bool converged = false;
    for (int s = 0; s < steps_per_frame; ++s) {
        if (iteration >= max_iter) break;

        auto res = armijo_step(system, rods, base_lr);
        last_energy = res.energy;

        if (use_adapt) {
            if (res.accepted) base_lr *= 1.05;
            else              base_lr *= 0.5;
            base_lr = std::clamp(base_lr, 1e-6, 1e-1);
        }

        if (res.force_norm < force_tol) {
            converged = true;
            ++iteration;
            break;
        }

        ++iteration;
    }
    return converged;
}

// ------------------------------
// Demos
// ------------------------------

void demo_visual_small_system() {
    std::cout << "Starting visual small system optimization..." << std::endl;

    SystemParameters params;
    params.collision_radius    = 0.05;    // radius
    params.harmonic_amplitude  = 1000.0;  // stiff
    params.entanglement_weight = 1.0;
    params.box_size            = 0.0;     // no container

    RodSystem system(params);

    system.add_rod(Rod({0.0, 0.0, 0.0}, M_PI / 4.0, 0.0,     1.0));
    system.add_rod(Rod({0.03, 0.0, 0.0}, M_PI / 3.0, M_PI/4, 1.0));
    system.add_rod(Rod({0.0, 0.03, 0.0}, M_PI / 2.0, M_PI/2, 1.0));

    std::cout << "Initial energy: " << system.total_energy() << std::endl;

    RodVisualizer visualizer(1200, 900, "Rod Optimization - Small System");

    RodVisualizer::ViewSettings view;
    view.camera_distance   = 3.0f;
    view.rod_radius        = 0.2f;
    view.show_axes         = true;
    view.show_contacts     = true;
    view.contact_threshold = 0.1f;
    view.show_container    = false;
    visualizer.set_view_settings(view);

    std::cout << "Controls:\n"
              << "  Mouse: Rotate view\n"
              << "  Scroll: Zoom\n"
              << "  'A': Toggle axes\n"
              << "  'R': Toggle auto-rotation\n"
              << "  'S': Save screenshot\n"
              << "  ESC: Exit\n";

    auto rods        = system.rods();
    int iteration    = 0;
    double last_E    = system.total_energy();
    bool converged   = false;

    // --- warm-up frame so you see *something* immediately ---
    system.set_rods(rods);
    visualizer.render(rods, /*box_size=*/params.box_size);
    visualizer.update();

    // Timing for throttling *compute*, not rendering
    auto start_time      = std::chrono::high_resolution_clock::now();
    auto last_step_time  = start_time;
    const double target_compute_dt = 1.0 / 60.0; // ~60 Hz compute budget

    int    max_iter        = 2000;
    int    steps_per_frame = 5;
    double base_lr         = 0.01;
    double force_tol       = 1e-5;

    std::cout << "[small] entering main loop\n";

    while (!visualizer.should_close() && !converged && iteration < max_iter) {
        // --- always render every loop ---
        system.set_rods(rods);
        visualizer.render(rods, /*box_size=*/params.box_size);
        visualizer.update();

        // --- throttle optimization work by time budget ---
        auto now = std::chrono::high_resolution_clock::now();
        double since = std::chrono::duration<double>(now - last_step_time).count();
        if (since >= target_compute_dt) {
            converged = run_inner_steps(system, rods, iteration, max_iter, steps_per_frame,
                                        last_E, force_tol, base_lr, true);
            last_step_time = now;

            if (iteration % 100 == 0) {
                const auto G = system.total_gradients();
                const double fnorm = std::sqrt(gradients_l2_sq(G));
                std::cout << "[small] iter " << iteration
                          << "  E=" << last_E
                          << "  |F|=" << fnorm
                          << "  lr=" << base_lr
                          << std::endl;
            }
        }

        // Tiny sleep to avoid 100% CPU when idle
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "\nOptimization completed!\n"
              << "Final iteration: " << iteration << "\n"
              << "Converged: " << (converged ? "Yes" : "No") << "\n"
              << "Final energy: " << last_E << "\n"
              << "Total time: " << ms << " ms\n"
              << "Close the visualization window to exit.\n";

    while (!visualizer.should_close()) {
        visualizer.render(rods, /*box_size=*/params.box_size);
        visualizer.update();
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }
}

void demo_visual_packing_optimization() {
    std::cout << "Starting visual packing optimization..." << std::endl;

    FastPacker::Parameters pack_params;
    pack_params.container_size = 2.0;
    pack_params.rod_length     = 1.0;
    pack_params.rod_diameter   = 0.03;
    pack_params.max_attempts   = 50000;
    pack_params.seed           = 42;

    FastPacker packer(pack_params);
    auto pack_result = packer.generate_packing(20);

    std::cout << "Initial packing: " << pack_result.placed
              << "/" << pack_result.attempted
              << " rods placed (packing fraction: "
              << pack_result.packing_fraction << ")\n";

    if (pack_result.placed < 5) {
        std::cout << "Too few rods placed. Try increasing container size or reducing rod diameter.\n";
        return;
    }

    SystemParameters sys_params;
    sys_params.collision_radius    = 0.5 * pack_params.rod_diameter;  // radius
    sys_params.harmonic_amplitude  = 500.0;
    sys_params.entanglement_weight = 0.1;
    sys_params.box_size            = pack_params.container_size * 2.0; // full box

    RodSystem system(sys_params);
    system.set_rods(pack_result.rods);

    std::cout << "Initial system energy: " << system.total_energy() << std::endl;

    RodVisualizer visualizer(1400, 1000, "Rod Optimization - Packing System");

    RodVisualizer::ViewSettings view;
    view.camera_distance   = 6.0f;
    view.rod_radius        = static_cast<float>(sys_params.collision_radius);
    view.show_axes         = true;
    view.show_container    = true;
    view.show_contacts     = true;
    view.contact_threshold = static_cast<float>(pack_params.rod_diameter * 1.2f);
    visualizer.set_view_settings(view);

    RodVisualizer::AnimationSettings anim;
    anim.auto_rotate    = true;
    anim.rotation_speed = 0.2f;
    visualizer.set_animation_settings(anim);

    auto rods        = system.rods();
    int iteration    = 0;
    double last_E    = system.total_energy();
    bool converged   = false;

    // Warm-up frame
    system.set_rods(rods);
    visualizer.render(rods, /*box_size=*/sys_params.box_size);
    visualizer.update();

    auto start_time     = std::chrono::high_resolution_clock::now();
    auto last_step_time = start_time;
    const double target_compute_dt = 1.0 / 60.0;

    int    max_iter        = 3000;
    int    steps_per_frame = 5;
    double base_lr         = 1e-3;
    double force_tol       = 1e-4;

    std::cout << "[packing] entering main loop\n";

    while (!visualizer.should_close() && !converged && iteration < max_iter) {
        // Always render
        system.set_rods(rods);
        visualizer.render(rods, /*box_size=*/sys_params.box_size);
        visualizer.update();

        // Time-budget the optimization steps
        auto now = std::chrono::high_resolution_clock::now();
        double since = std::chrono::duration<double>(now - last_step_time).count();
        if (since >= target_compute_dt) {
            converged = run_inner_steps(system, rods, iteration, max_iter, steps_per_frame,
                                        last_E, force_tol, base_lr, /*use_adapt=*/true);
            last_step_time = now;

            if (iteration % 200 == 0) {
                const auto G = system.total_gradients();
                const double fnorm = std::sqrt(gradients_l2_sq(G));
                std::cout << "[packing] iter " << iteration
                          << "  E=" << last_E
                          << "  |F|=" << fnorm
                          << "  lr=" << base_lr
                          << std::endl;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "\nOptimization completed!\n"
              << "Final iteration: " << iteration << "\n"
              << "Converged: " << (converged ? "Yes" : "No") << "\n"
              << "Final energy: " << last_E << "\n"
              << "Total time: " << ms << " ms\n";

    const auto linking_numbers = system.all_linking_numbers();
    const auto entangled_pairs = std::count_if(
        linking_numbers.begin(), linking_numbers.end(),
        [](double lk) { return std::abs(lk) > 1e-6; });

    std::cout << "Final analysis:\n"
              << "  Total pairs: " << linking_numbers.size() << "\n"
              << "  Entangled pairs: " << entangled_pairs << "\n";
    std::cout << "Close the visualization window to exit.\n";

    while (!visualizer.should_close()) {
        visualizer.render(rods, /*box_size=*/sys_params.box_size);
        visualizer.update();
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Visual Rod Optimization Demo\n"
              << "============================\n";

    std::string demo_type = "small";
    if (argc > 1) demo_type = argv[1];

    try {
        if (demo_type == "small") {
            demo_visual_small_system();
        } else if (demo_type == "packing") {
            demo_visual_packing_optimization();
        } else {
            std::cout << "Usage: " << argv[0] << " [small|packing]\n"
                      << "  small   - Small system with 3 overlapping rods\n"
                      << "  packing - Random packing optimization\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
