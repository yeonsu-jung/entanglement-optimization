/**
 * @file entanglement.cpp
 * @brief Implementation of entanglement calculations
 */

#include "entanglement.hpp"
#include "distance.hpp"
#include <algorithm>

namespace entanglement {

double linking_number_gauss(const Vec3& p_i, const Vec3& p_ii,
                           const Vec3& p_j, const Vec3& p_jj) {
    
    // Four vectors from quadrilateral corners
    Vec3 r_ij   = p_i - p_j;
    Vec3 r_ijj  = p_i - p_jj;
    Vec3 r_iij  = p_ii - p_j;
    Vec3 r_iijj = p_ii - p_jj;
    
    const double tol = 1e-6;
    
    // Four normal vectors (cross products)
    Vec3 n1 = r_ij.cross(r_ijj);
    n1 = n1 * (1.0 / (n1.norm() + tol));
    
    Vec3 n2 = r_ijj.cross(r_iijj);
    n2 = n2 * (1.0 / (n2.norm() + tol));
    
    Vec3 n3 = r_iijj.cross(r_iij);
    n3 = n3 * (1.0 / (n3.norm() + tol));
    
    Vec3 n4 = r_iij.cross(r_ij);
    n4 = n4 * (1.0 / (n4.norm() + tol));
    
    // Gauss linking integral (discrete approximation)
    double sum = std::asin(std::clamp(n1.dot(n2), -1.0 + tol, 1.0 - tol)) +
                 std::asin(std::clamp(n2.dot(n3), -1.0 + tol, 1.0 - tol)) +
                 std::asin(std::clamp(n3.dot(n4), -1.0 + tol, 1.0 - tol)) +
                 std::asin(std::clamp(n4.dot(n1), -1.0 + tol, 1.0 - tol));
    
    return -std::abs(sum) / (4.0 * M_PI);
}

double linking_number_arai(const Vec3& p_i, const Vec3& p_ii,
                          const Vec3& p_j, const Vec3& p_jj) {
    
    Vec3 a = p_i - p_j;
    Vec3 b = p_i - p_jj;
    Vec3 c = p_ii - p_jj;
    Vec3 d = p_ii - p_j;
    
    Vec3 cross_bc = b.cross(c);
    Vec3 cross_da = d.cross(a);
    
    double term1 = std::atan2(
        a.dot(cross_bc),
        a.norm() * b.norm() * c.norm() + 
        a.dot(b) * c.norm() + 
        c.dot(a) * b.norm() + 
        b.dot(c) * a.norm()
    );
    
    double term2 = std::atan2(
        c.dot(cross_da),
        c.norm() * d.norm() * a.norm() + 
        c.dot(d) * a.norm() + 
        a.dot(c) * d.norm() + 
        d.dot(a) * c.norm()
    );
    
    return -std::abs(term1 + term2) / (2.0 * M_PI);
}

double rod_linking_number(const Rod& rod1, const Rod& rod2, bool use_arai) {
    auto [p1s, p1e] = rod1.endpoints();
    auto [p2s, p2e] = rod2.endpoints();
    
    if (use_arai) {
        return linking_number_arai(p1s, p1e, p2s, p2e);
    } else {
        return linking_number_gauss(p1s, p1e, p2s, p2e);
    }
}

bool is_potentially_entangled(const Rod& rod1, const Rod& rod2, double threshold) {
    // Fast check: if rods are far apart or nearly parallel, unlikely to be entangled
    double distance = rod_distance(rod1, rod2);
    double max_length = std::max(rod1.length, rod2.length);
    
    if (distance > max_length) {
        return false;  // Too far apart
    }
    
    double angle = rod_angle(rod1, rod2);
    if (angle < M_PI / 6) {  // Less than 30 degrees
        return false;  // Nearly parallel
    }
    
    // For close, non-parallel rods, compute linking number
    double linking = rod_linking_number(rod1, rod2);
    return std::abs(linking) > threshold;
}

} // namespace entanglement