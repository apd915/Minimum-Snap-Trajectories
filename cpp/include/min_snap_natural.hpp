#pragma once

#include <Eigen/Dense>
#include <cstdint> 
#include <vector>
#include <utility>

// ==========================================
// HELPER STRUCTS
// ==========================================

// Replaces the Python dictionary for SFC constraints
struct SFCConstraint {
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
};

// Packages the 5 matrices returned by the SVD calculation
struct SVDResult {
    Eigen::MatrixXd BCombined;
    Eigen::MatrixXd U1;
    Eigen::MatrixXd U2;
    Eigen::MatrixXd Sigma;
    Eigen::MatrixXd V;
};

// ==========================================
// CORE SOLVER CLASS
// ==========================================

class MinSnapEvalNatural {
private:
    // ---------------------------------------------------------
    // PRIVATE STATE VARIABLES
    // ---------------------------------------------------------
    int8_t degree_; 
    int16_t M_; // num_segments
    int16_t numControlPoints_;

    Eigen::VectorXd knots_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd W_;
    Eigen::MatrixXd BCombined_; 

    // ---------------------------------------------------------
    // PRIVATE METHODS (Core Math)
    // ---------------------------------------------------------

    Eigen::VectorXd createUniformKnotPoints(int16_t numCtrlPts, int8_t degree, double startTime);
    
    // Core SVD Solver sequence
    void calculateQ();
    
    // Dynamic structural builders
    std::pair<Eigen::MatrixXd, double> getMMatrix(int8_t degree);
    
    Eigen::MatrixXd getTVector(int8_t degree, int8_t derivativeOrder, double tau);
    
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> getBoundaryStates(const Eigen::MatrixXd& mMatrix, int8_t degree);
    
    SVDResult createSVD(int16_t numControlPoints);
    
    Eigen::MatrixXd getSMatrix(int16_t M, int8_t k);

public:
    // ---------------------------------------------------------
    // PUBLIC API (What the ROS node is allowed to call)
    // ---------------------------------------------------------
    
    // Constructor
    MinSnapEvalNatural(int16_t numSegments, int8_t degree = 4);

    // Dynamic segment updater
    void updateSegments(int16_t newNumSegments);

    // Getters for the trajectory planner
    Eigen::MatrixXd getQMatrix() const;
    Eigen::VectorXd getKnots() const;
    
    // Returns BCombined for the OSQP Equality Constraints (A_eq)
    Eigen::MatrixXd getBCombined() const;

    // Penalty matrix generator
    Eigen::MatrixXd getWMatrix(double rhoVel = 0.0, double rhoAccel = 0.0, double rhoSnap = 1.0);

    // Returns the stored W_ matrix (pure minimum-snap penalty computed at construction)
    Eigen::MatrixXd getW() const { return W_; }

    // Derivative matrix builder (needed by trajectory_planner for MINVO kinodynamic constraints)
    Eigen::MatrixXd getFastCascadedDMatrix(int16_t M, int8_t degree, int8_t derivativeOrder);
    
    // Massively stacks SFC inequalities for the QP solver
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> getSfcMatrices(
        const std::vector<SFCConstraint>& sfcConstraints, 
        const std::vector<int16_t>& numPtsList
    );
};