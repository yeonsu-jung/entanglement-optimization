#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

// Include the header with analytic expressions
#include "../../entanglement-optimization-cpp/cpp_analytic_expressions.hpp"

using namespace entanglement;

// Simple structure to hold rod parameters
struct RodParams {
    double x, y, z;
    double phi, theta;
    double length;
};

Rod vector_to_rod(const std::vector<double>& v, double length = 1.0) {
    Rod rod;
    rod.center = {v[0], v[1], v[2]};
    rod.phi = v[3];
    rod.theta = v[4];
    rod.length = length;
    return rod;
}

int main(int argc, char* argv[]) {
    // Read input from stdin
    // Expected format: line-by-line rod pair parameters
    // x1 y1 z1 phi1 theta1 x2 y2 z2 phi2 theta2 length
    
    std::string line;
    std::cout << "[" << std::endl;
    bool first = true;

    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::vector<double> vals;
        double val;
        while (ss >> val) {
            vals.push_back(val);
        }
        
        if (vals.size() < 10) continue;
        
        double length = (vals.size() > 10) ? vals[10] : 1.0;
        
        std::vector<double> params1(vals.begin(), vals.begin() + 5);
        std::vector<double> params2(vals.begin() + 5, vals.begin() + 10);
        
        Rod rod1 = vector_to_rod(params1, length);
        Rod rod2 = vector_to_rod(params2, length);
        
        // Compute values
        double dist = rod_rod_distance(rod1, rod2);
        double lk_gauss = linking_number_gauss(rod1.get_endpoints().first, rod1.get_endpoints().second,
                                             rod2.get_endpoints().first, rod2.get_endpoints().second);
        double lk_arai = linking_number_arai(rod1.get_endpoints().first, rod1.get_endpoints().second,
                                            rod2.get_endpoints().first, rod2.get_endpoints().second);
        
        RodPairGradients grad_dist = segment_distance_gradient(rod1, rod2);
        
        // Output JSON object
        if (!first) std::cout << "," << std::endl;
        first = false;
        
        std::cout << "  {" << std::endl;
        std::cout << "    \"distance\": " << std::setprecision(16) << dist << "," << std::endl;
        std::cout << "    \"lk_gauss\": " << lk_gauss << "," << std::endl;
        std::cout << "    \"lk_arai\": " << lk_arai << "," << std::endl;
        
        std::cout << "    \"grad_dist_rod1\": [" 
                  << grad_dist.grad_rod1.dx << ", " << grad_dist.grad_rod1.dy << ", " << grad_dist.grad_rod1.dz << ", "
                  << grad_dist.grad_rod1.dphi << ", " << grad_dist.grad_rod1.dtheta << "]," << std::endl;
                  
        std::cout << "    \"grad_dist_rod2\": [" 
                  << grad_dist.grad_rod2.dx << ", " << grad_dist.grad_rod2.dy << ", " << grad_dist.grad_rod2.dz << ", "
                  << grad_dist.grad_rod2.dphi << ", " << grad_dist.grad_rod2.dtheta << "]" << std::endl;
                  
        std::cout << "  }";
    }
    
    std::cout << std::endl << "]" << std::endl;
    
    return 0;
}
