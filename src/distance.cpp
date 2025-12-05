/**
 * @file distance.cpp
 * @brief Implementation of distance calculations
 */

#include "distance.hpp"
#include <algorithm>

namespace entanglement {

std::pair<double, double> find_closest_parameters(const Vec3& d1, const Vec3& d2, const Vec3& d12) {
    double D1 = d1.dot(d1);
    double D2 = d2.dot(d2);
    double S1 = d1.dot(d12);
    double S2 = d2.dot(d12);
    double R = d1.dot(d2);
    double den = D1 * D2 - R * R;
    
    double t, u;
    
    if (D1 < 1e-12 || D2 < 1e-12) {
        // One or both segments are degenerate (points)
        if (D1 > 1e-12) {  // segment1 is line, segment2 is point
            u = 0.0;
            t = clamp01(S1 / D1);
        } else if (D2 > 1e-12) {  // segment2 is line, segment1 is point
            t = 0.0;
            u = clamp01(-S2 / D2);
        } else {  // both are points
            t = u = 0.0;
        }
    } else if (std::abs(den) < 1e-12) {
        // Segments are parallel
        t = 0.0;
        u = clamp01(-S2 / D2);
        double uf = u;
        if (std::abs(uf - u) > 1e-12) {
            t = clamp01((uf * R + S1) / D1);
            u = uf;
        }
    } else {
        // General case: non-parallel segments
        t = clamp01((S1 * D2 - S2 * R) / den);
        u = (t * R - S2) / D2;
        double uf = clamp01(u);
        if (std::abs(uf - u) > 1e-12) {
            t = clamp01((uf * R + S1) / D1);
            u = uf;
        }
    }
    
    return {t, u};
}

double segment_distance(const Vec3& p1s, const Vec3& p1e,
                       const Vec3& p2s, const Vec3& p2e) {
    
    Vec3 d1 = p1e - p1s;
    Vec3 d2 = p2e - p2s;
    Vec3 d12 = p2s - p1s;
    
    auto [t, u] = find_closest_parameters(d1, d2, d12);
    
    Vec3 closest_diff = d1 * t - d2 * u - d12;
    return closest_diff.norm();
}

double segment_distance_pbc(const Vec3& p1s, const Vec3& p1e,
                           const Vec3& p2s, const Vec3& p2e,
                           double box_size) {
    
    // Find minimum-image displacement between segment midpoints
    Vec3 m1 = (p1s + p1e) * 0.5;
    Vec3 m2 = (p2s + p2e) * 0.5;
    
    Vec3 d_mi = minimum_image(m2 - m1, box_size);
    Vec3 shift = d_mi - (m2 - m1);
    
    // Shift second segment to closest periodic image
    Vec3 p2s_shifted = p2s + shift;
    Vec3 p2e_shifted = p2e + shift;
    
    // Now compute standard Euclidean distance
    return segment_distance(p1s, p1e, p2s_shifted, p2e_shifted);
}

double rod_distance(const Rod& rod1, const Rod& rod2) {
    auto [p1s, p1e] = rod1.endpoints();
    auto [p2s, p2e] = rod2.endpoints();
    return segment_distance(p1s, p1e, p2s, p2e);
}

double rod_distance_pbc(const Rod& rod1, const Rod& rod2, double box_size) {
    auto [p1s, p1e] = rod1.endpoints();
    auto [p2s, p2e] = rod2.endpoints();
    return segment_distance_pbc(p1s, p1e, p2s, p2e, box_size);
}

} // namespace entanglement