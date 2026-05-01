#pragma once // Prevents the compiler from including this file twice

#include <Eigen/Dense>
#include <vector>

class MinSnapNatural {
public:
    // Constructor: Pass in the number of control points or segments
    MinSnapNatural(int num_segments);

    // Public method to generate the W matrix
    Eigen::MatrixXd generate_W_matrix();

    // Public method to generate the S matrix
    Eigen::MatrixXd generate_S_matrix();

private:
    int N; // Store the number of control points
    
    // You can declare private helper functions here if needed
    // Eigen::MatrixXd build_corner_stencil();
};