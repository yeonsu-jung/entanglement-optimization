/**
 * @file gradients.cpp
 * @brief Implementation of gradient calculations
 */

#include "gradients.hpp"
#include "distance.hpp"
#include "entanglement.hpp"

namespace entanglement {

RodPairGradients distance_gradient(const Rod& rod1, const Rod& rod2) {
    
    // Get current endpoints and distance
    auto [p1s, p1e] = rod1.endpoints();
    auto [p2s, p2e] = rod2.endpoints();
    
    // Direction vectors and optimization parameters
    Vec3 d1 = p1e - p1s;
    Vec3 d2 = p2e - p2s;
    Vec3 d12 = p2s - p1s;
    
    auto [t, u] = find_closest_parameters(d1, d2, d12);
    
    // Closest points on segments
    Vec3 closest1 = p1s + d1 * t;
    Vec3 closest2 = p2s + d2 * u;
    Vec3 separation = closest1 - closest2;
    double distance = separation.norm();
    
    if (distance < 1e-12) {
        // Segments are touching; gradient is undefined/zero
        return {{0,0,0,0,0}, {0,0,0,0,0}};
    }
    
    Vec3 unit_sep = separation / distance;
    
    // Chain rule: ∂d/∂qᵢ = (∂d/∂closest_points) * (∂closest_points/∂endpoints) * (∂endpoints/∂qᵢ)
    Vec3 grad_c1 = unit_sep;
    Vec3 grad_c2 = unit_sep * (-1.0);
    
    // ∂closest1/∂p1s = (1-t), ∂closest1/∂p1e = t
    // ∂closest2/∂p2s = (1-u), ∂closest2/∂p2e = u
    Vec3 grad_p1s = grad_c1 * (1.0 - t);
    Vec3 grad_p1e = grad_c1 * t;
    Vec3 grad_p2s = grad_c2 * (1.0 - u);
    Vec3 grad_p2e = grad_c2 * u;
    
    // Convert endpoint gradients to rod parameter gradients
    RodPairGradients result;
    
    // For rod1: p1s = center1 - 0.5*L*direction1, p1e = center1 + 0.5*L*direction1
    Vec3 direction1 = rod1.direction();
    double half_L1 = 0.5 * rod1.length;
    
    // ∂(center)/∂(x,y,z) = (1,1,1)
    result.grad_rod1.dx = grad_p1s.x + grad_p1e.x;
    result.grad_rod1.dy = grad_p1s.y + grad_p1e.y;
    result.grad_rod1.dz = grad_p1s.z + grad_p1e.z;
    
    // ∂direction/∂φ and ∂direction/∂θ
    Vec3 ddirection_dphi = derivative_phi(rod1.phi, rod1.theta);
    Vec3 ddirection_dtheta = derivative_theta(rod1.phi, rod1.theta);
    
    // ∂p1s/∂φ = -0.5*L*(∂direction/∂φ), ∂p1e/∂φ = +0.5*L*(∂direction/∂φ)
    Vec3 grad_phi_contribution = (grad_p1e - grad_p1s) * half_L1;
    Vec3 grad_theta_contribution = (grad_p1e - grad_p1s) * half_L1;
    
    result.grad_rod1.dphi = grad_phi_contribution.dot(ddirection_dphi);
    result.grad_rod1.dtheta = grad_theta_contribution.dot(ddirection_dtheta);
    
    // Same for rod2
    Vec3 direction2 = rod2.direction();
    double half_L2 = 0.5 * rod2.length;
    
    result.grad_rod2.dx = grad_p2s.x + grad_p2e.x;
    result.grad_rod2.dy = grad_p2s.y + grad_p2e.y;
    result.grad_rod2.dz = grad_p2s.z + grad_p2e.z;
    
    Vec3 ddirection2_dphi = derivative_phi(rod2.phi, rod2.theta);
    Vec3 ddirection2_dtheta = derivative_theta(rod2.phi, rod2.theta);
    
    Vec3 grad2_phi_contribution = (grad_p2e - grad_p2s) * half_L2;
    Vec3 grad2_theta_contribution = (grad_p2e - grad_p2s) * half_L2;
    
    result.grad_rod2.dphi = grad2_phi_contribution.dot(ddirection2_dphi);
    result.grad_rod2.dtheta = grad2_theta_contribution.dot(ddirection2_dtheta);
    
    return result;
}

template<typename Func>
RodPairGradients finite_difference_gradient(Func func, const Rod& rod1, const Rod& rod2, double eps) {
    
    double f0 = func(rod1, rod2);
    RodPairGradients result;
    
    // Gradients for rod1
    Rod test_rod = rod1;
    test_rod.center.x += eps;
    result.grad_rod1.dx = (func(test_rod, rod2) - f0) / eps;
    
    test_rod = rod1;
    test_rod.center.y += eps;
    result.grad_rod1.dy = (func(test_rod, rod2) - f0) / eps;
    
    test_rod = rod1;
    test_rod.center.z += eps;
    result.grad_rod1.dz = (func(test_rod, rod2) - f0) / eps;
    
    test_rod = rod1;
    test_rod.phi += eps;
    result.grad_rod1.dphi = (func(test_rod, rod2) - f0) / eps;
    
    test_rod = rod1;
    test_rod.theta += eps;
    result.grad_rod1.dtheta = (func(test_rod, rod2) - f0) / eps;
    
    // Gradients for rod2
    test_rod = rod2;
    test_rod.center.x += eps;
    result.grad_rod2.dx = (func(rod1, test_rod) - f0) / eps;
    
    test_rod = rod2;
    test_rod.center.y += eps;
    result.grad_rod2.dy = (func(rod1, test_rod) - f0) / eps;
    
    test_rod = rod2;
    test_rod.center.z += eps;
    result.grad_rod2.dz = (func(rod1, test_rod) - f0) / eps;
    
    test_rod = rod2;
    test_rod.phi += eps;
    result.grad_rod2.dphi = (func(rod1, test_rod) - f0) / eps;
    
    test_rod = rod2;
    test_rod.theta += eps;
    result.grad_rod2.dtheta = (func(rod1, test_rod) - f0) / eps;
    
    return result;
}

RodPairGradients linking_gradient(const Rod& rod1, const Rod& rod2, double eps, bool use_arai) {
    auto linking_func = [use_arai](const Rod& r1, const Rod& r2) {
        return rod_linking_number(r1, r2, use_arai);
    };
    
    return finite_difference_gradient(linking_func, rod1, rod2, eps);
}

RodPairGradients harmonic_gradient(const Rod& rod1, const Rod& rod2,
                                  double collision_radius, double amplitude) {
    
    double distance = rod_distance(rod1, rod2);
    double grad_factor = 2.0 * amplitude * (distance - collision_radius);
    
    RodPairGradients dist_grad = distance_gradient(rod1, rod2);
    
    // Scale by harmonic potential derivative
    return {dist_grad.grad_rod1 * grad_factor, dist_grad.grad_rod2 * grad_factor};
}

RodPairGradients effective_gradient(const Rod& rod1, const Rod& rod2,
                                   double collision_radius, double amplitude,
                                   double eps) {
    
    RodPairGradients harmonic_grad = harmonic_gradient(rod1, rod2, collision_radius, amplitude);
    RodPairGradients linking_grad = linking_gradient(rod1, rod2, eps);
    
    // Combine: ∇U = ∇U_harmonic + ∇U_linking
    return {
        harmonic_grad.grad_rod1 + linking_grad.grad_rod1,
        harmonic_grad.grad_rod2 + linking_grad.grad_rod2
    };
}

bool validate_gradients(const Rod& rod1, const Rod& rod2,
                       const RodPairGradients& analytical,
                       double eps, double tolerance) {
    
    // Use finite differences on distance function for validation
    auto dist_func = [](const Rod& r1, const Rod& r2) {
        return rod_distance(r1, r2);
    };
    
    RodPairGradients finite_diff = finite_difference_gradient(dist_func, rod1, rod2, eps);
    
    // Check relative errors
    auto check_component = [tolerance](double analytical, double finite_diff) {
        double rel_error = std::abs(analytical - finite_diff) / (std::abs(finite_diff) + 1e-12);
        return rel_error < tolerance;
    };
    
    return check_component(analytical.grad_rod1.dx, finite_diff.grad_rod1.dx) &&
           check_component(analytical.grad_rod1.dy, finite_diff.grad_rod1.dy) &&
           check_component(analytical.grad_rod1.dz, finite_diff.grad_rod1.dz) &&
           check_component(analytical.grad_rod1.dphi, finite_diff.grad_rod1.dphi) &&
           check_component(analytical.grad_rod1.dtheta, finite_diff.grad_rod1.dtheta) &&
           check_component(analytical.grad_rod2.dx, finite_diff.grad_rod2.dx) &&
           check_component(analytical.grad_rod2.dy, finite_diff.grad_rod2.dy) &&
           check_component(analytical.grad_rod2.dz, finite_diff.grad_rod2.dz) &&
           check_component(analytical.grad_rod2.dphi, finite_diff.grad_rod2.dphi) &&
           check_component(analytical.grad_rod2.dtheta, finite_diff.grad_rod2.dtheta);
}

// Explicit template instantiation for common use cases
template RodPairGradients finite_difference_gradient<double(*)(const Rod&, const Rod&)>(
    double(*)(const Rod&, const Rod&), const Rod&, const Rod&, double);

} // namespace entanglement