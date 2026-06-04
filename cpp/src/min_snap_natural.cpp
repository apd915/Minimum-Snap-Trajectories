#include "min_snap_natural.hpp" 
#include <stdexcept>
#include <cmath> // Required for std::pow

// Safely apply namespaces ONLY in the .cpp file
using namespace std;
using namespace Eigen;


// ==========================================
// ANONYMOUS NAMESPACE (Private Math Helpers)
// ==========================================
namespace {
    // Fast, exact integer factorial
    int64_t factorial(int8_t n) {
        int64_t res = 1;
        for (int8_t i = 2; i <= n; ++i) {
            res *= i;
        }
        return res;
    }

    // Fast, exact integer combination (nCr)
    int64_t nCr(int8_t n, int8_t r) {
        if (r < 0 || r > n) return 0;
        if (r == 0 || r == n) return 1;
        if (r > n / 2) r = n - r; // Exploit symmetry for speed
        
        int64_t res = 1;
        for (int8_t i = 1; i <= r; ++i) {
            res = res * (n - i + 1) / i;
        }
        return res;
    }
}

// ==========================================
// PUBLIC API
// ==========================================

MinSnapEvalNatural::MinSnapEvalNatural(int16_t numSegments, int8_t degree) {
    // 1. Intuitive Safety Checks
    if (degree < 4) {
        throw invalid_argument("Minimum Snap requires a polynomial of at least degree 4.");
    }
    if (numSegments < 3) {
        throw invalid_argument("To satisfy 6 physical constraints, you need at least 3 flight segments.");
    }
    
    degree_ = degree;

    // Only initialize state here, do NOT do the math yet.
    updateSegments(numSegments);
}

void MinSnapEvalNatural::updateSegments(int16_t newNumSegments) {
    M_ = newNumSegments;
    numControlPoints_ = M_ + degree_;
    
    double startTime = 0.0;
    
    // Generate the uniform knot vector
    knots_ = createUniformKnotPoints(numControlPoints_, degree_, startTime);
    
    // Now trigger the heavy math!
    calculateQ();
}

// ==========================================
// PRIVATE METHODS (Core Math)
// ==========================================

VectorXd MinSnapEvalNatural::createUniformKnotPoints(int16_t numCtrlPts, int8_t degree, double startTime) {
    /*
     * Python equivalent: np.arange(-self.degree, self.M + self.degree + 1)
     * For a Natural B-Spline, the knots stretch perfectly uniformly into negative time.
     */
    
    int16_t numKnots = numCtrlPts + degree + 1;
    VectorXd knots(numKnots);
    
    for (int16_t i = 0; i < numKnots; ++i) {
        // static_cast ensures we safely convert the integer math into C-floats
        knots(i) = static_cast<double>(i - degree) + startTime;
    }
    
    return knots;
}

void MinSnapEvalNatural::calculateQ() {
    // 1. Generate the Boundary SVD Partition
    SVDResult svd = createSVD(numControlPoints_);
    
    // Save BCombined for the QP solver's equality constraints later
    BCombined_ = svd.BCombined; 

    // 2. Get the analytical integration penalty matrix (W)
    W_ = getWMatrix(0.0, 0.0, 1.0); // Minimize Snap (4th derivative)

    // 3. Solve the Minimum Energy System
    MatrixXd A_bar = svd.U2.transpose() * W_ * svd.U2;
    MatrixXd B_bar = svd.U2.transpose() * W_;

    // We use LDLT decomposition because A_bar is symmetric positive semi-definite
    MatrixXd X_T = A_bar.ldlt().solve(B_bar);
    MatrixXd X = X_T.transpose();

    // 4. Final Q is calculated via null space subtraction
    MatrixXd I = MatrixXd::Identity(numControlPoints_, numControlPoints_);
    
    // FIXED: Correctly reconstruct the pseudo-inverse: V * Sigma^-1 * U1^T
    MatrixXd sigmaInv = svd.Sigma.inverse();
    Q_ = svd.V * sigmaInv * svd.U1.transpose() * (I - X * svd.U2.transpose());
}

// ==========================================
// PENALTY MATRIX (W) & STENCIL BUILDERS
// ==========================================

MatrixXd MinSnapEvalNatural::getFastCascadedDMatrix(int16_t M, int8_t degree, int8_t derivativeOrder) {
    // 1. Fetch the hardcoded stencil for the derivative order
    std::vector<double> stencil;
    if (derivativeOrder == 1) {
        stencil = {-1.0, 1.0};
    } else if (derivativeOrder == 2) {
        stencil = {1.0, -2.0, 1.0};
    } else if (derivativeOrder == 3) {
        stencil = {-1.0, 3.0, -3.0, 1.0};
    } else if (derivativeOrder == 4) {
        stencil = {1.0, -4.0, 6.0, -4.0, 1.0};
    } else {
        throw std::invalid_argument("Stencil for derivative order not hardcoded.");
    }

    // 2. Matrix Dimensions (mapping from M+d-l up to M+d)
    int16_t rows = M + degree;
    int16_t cols = M + degree - derivativeOrder;
    MatrixXd dCascaded = MatrixXd::Zero(rows, cols);

    // 3. Tile the stencil column-wise
    for (int16_t i = 0; i < cols; ++i) {
        for (size_t j = 0; j < stencil.size(); ++j) {
            dCascaded(i + j, i) = stencil[j];
        }
    }

    return dCascaded;
}

Eigen::MatrixXd MinSnapEvalNatural::getSMatrix(int16_t M, int8_t k) {
    int16_t size = M + k;
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(size, size);
    
    // 1. Fetch the infinite interior stencil based on k
    std::vector<double> stencil;
    if (k == 0) {
        stencil = {1.0};
    } else if (k == 1) {
        stencil = {4.0/6.0, 1.0/6.0}; // Note: Python was [1/6, 4/6, 1/6]. We just store the right half: [center, offset1]
    } else if (k == 2) {
        stencil = {66.0/120.0, 26.0/120.0, 1.0/120.0};
    } else if (k == 3) {
        stencil = {2416.0/5040.0, 1191.0/5040.0, 120.0/5040.0, 1.0/5040.0};
    } else {
        throw std::invalid_argument("Integral stencil for k is not yet hardcoded.");
    }

    // 2. Populate the main diagonal
    for (int16_t i = 0; i < size; ++i) {
        S(i, i) += stencil[0];
    }
    
    // 3. Populate sub and super diagonals
    for (size_t offset = 1; offset < stencil.size(); ++offset) {
        for (int16_t i = 0; i < size - offset; ++i) {
            S(i, i + offset) += stencil[offset];     // Super-diagonal
            S(i + offset, i) += stencil[offset];     // Sub-diagonal
        }
    }

    // 4. APPLY BOUNDARY TRUNCATION PATCHES
    if (k == 1) {
        S(0, 0) = 1.0/3.0;
        S(size-1, size-1) = 1.0/3.0;
    } 
    else if (k == 2) {
        // Main Diagonal Patches
        S(0, 0) = 1.0/20.0;
        S(1, 1) = 1.0/2.0;
        S(size-1, size-1) = 1.0/20.0;
        S(size-2, size-2) = 1.0/2.0;
        
        // First Off-Diagonal
        S(0, 1) = 13.0/120.0; S(1, 0) = 13.0/120.0;
        S(size-1, size-2) = 13.0/120.0; S(size-2, size-1) = 13.0/120.0;
    } 
    else if (k == 3) {
        // Main Diagonal Patches
        S(0, 0) = 1.0/252.0;
        S(1, 1) = 151.0/630.0;
        S(2, 2) = 599.0/1260.0;
        
        S(size-1, size-1) = 1.0/252.0;
        S(size-2, size-2) = 151.0/630.0;
        S(size-3, size-3) = 599.0/1260.0;

        // First Off-Diagonal Patches
        S(0, 1) = 43.0/1680.0; S(1, 0) = 43.0/1680.0;
        S(1, 2) = 59.0/280.0;  S(2, 1) = 59.0/280.0;
        
        S(size-1, size-2) = 43.0/1680.0; S(size-2, size-1) = 43.0/1680.0;
        S(size-2, size-3) = 59.0/280.0;  S(size-3, size-2) = 59.0/280.0;

        // Second Off-Diagonal Patches
        S(0, 2) = 1.0/84.0; S(2, 0) = 1.0/84.0;
        S(size-1, size-3) = 1.0/84.0; S(size-3, size-1) = 1.0/84.0;
    }

    return S;
}

MatrixXd MinSnapEvalNatural::getWMatrix(double rhoVel, double rhoAccel, double rhoSnap) {
    MatrixXd wTotal = MatrixXd::Zero(numControlPoints_, numControlPoints_);
    
    if (rhoSnap > 0.0 && degree_ >= 4) {
        MatrixXd dSnap = getFastCascadedDMatrix(M_, degree_, 4);
        MatrixXd wSnap;
        if (degree_ - 4 == 0) {
            wSnap = dSnap * dSnap.transpose(); // FIXED ORDER
        } else {
            wSnap = dSnap * getSMatrix(M_, degree_ - 4) * dSnap.transpose(); // FIXED ORDER
        }
        wTotal += rhoSnap * wSnap;
    }
    
    if (rhoAccel > 0.0 && degree_ >= 2) {
        MatrixXd dAccel = getFastCascadedDMatrix(M_, degree_, 2);
        MatrixXd wAccel = dAccel * getSMatrix(M_, degree_ - 2) * dAccel.transpose(); // FIXED ORDER
        wTotal += rhoAccel * wAccel;
    }
    
    if (rhoVel > 0.0 && degree_ >= 1) {
        MatrixXd dVel = getFastCascadedDMatrix(M_, degree_, 1);
        MatrixXd wVel = dVel * getSMatrix(M_, degree_ - 1) * dVel.transpose(); // FIXED ORDER
        wTotal += rhoVel * wVel;
    }

    return wTotal;
}

pair<MatrixXd, double> MinSnapEvalNatural::getMMatrix(int8_t degree) {
    // Initialize a matrix of zeros
    MatrixXd M = MatrixXd::Zero(degree + 1, degree + 1);
    
    for (int8_t r = 0; r <= degree; ++r) {
        for (int8_t j = 0; j <= degree; ++j) {
            int64_t val = 0; // Use 64-bit int to prevent combinatorial overflow
            
            for (int8_t k = 0; k <= degree - r; ++k) {
                // term1: Alternating sign (-1)^k
                int64_t term1 = (k % 2 == 0) ? 1 : -1;
                
                // term2 & term3: Combinations
                int64_t term2 = nCr(degree + 1, k);
                int64_t term3 = nCr(degree, j);
                
                // term4: Power function with 0^0 safety catch
                int8_t base = degree - r - k;
                int64_t term4 = 1; 
                if (!(base == 0 && j == 0)) {
                    // Calculate power and safely cast back to integer
                    term4 = static_cast<int64_t>(round(pow(base, j)));
                }
                
                val += term1 * term2 * term3 * term4;
            }
            
            // Cast the exact integer result to a double for Eigen
            M(r, j) = static_cast<double>(val); 
        }
    }
    
    double scalar = static_cast<double>(factorial(degree));
    
    // Modern C++ brace initialization automatically packs the std::pair
    return {M, scalar}; 
}

MatrixXd MinSnapEvalNatural::getTVector(int8_t degree, int8_t derivativeOrder, double tau) {
    // Initialize a column vector of zeros: (degree + 1) rows, 1 column
    MatrixXd T = MatrixXd::Zero(degree + 1, 1);
    
    for (int8_t i = 0; i <= degree; ++i) {
        int8_t power = degree - i;
        
        if (power >= derivativeOrder) {
            // Calculate the cascaded derivative scalar using the power rule
            int64_t scalar = 1;
            if (derivativeOrder > 0) {
                // Equivalent to math.prod(range(power - derivative_order + 1, power + 1))
                for (int8_t j = power - derivativeOrder + 1; j <= power; ++j) {
                    scalar *= j;
                }
            }
            
            // Calculate tau^(power - derivative_order)
            int8_t exponent = power - derivativeOrder;
            double tauTerm = 1.0; // Default for tau^0
            
            if (exponent > 0) {
                tauTerm = pow(tau, exponent);
            }
            
            // Assign the final computed value
            T(i, 0) = static_cast<double>(scalar) * tauTerm;
        }
    }
    
    return T;
}

pair<MatrixXd, MatrixXd> MinSnapEvalNatural::getBoundaryStates(const MatrixXd& mMatrix, int8_t degree) {
    // ------------------------------------------
    // START BOUNDARY (tau = 0)
    // ------------------------------------------
    MatrixXd startPos = getTVector(degree, 0, 0.0);
    MatrixXd startVel = getTVector(degree, 1, 0.0);
    MatrixXd startAcc = getTVector(degree, 2, 0.0);
    
    MatrixXd tStartCombined(degree + 1, 3);
    tStartCombined << startPos, startVel, startAcc;
    
    MatrixXd bD_M0 = mMatrix * tStartCombined;

    // ------------------------------------------
    // END BOUNDARY (tau = 1)
    // ------------------------------------------
    MatrixXd endPos = getTVector(degree, 0, 1.0);
    MatrixXd endVel = getTVector(degree, 1, 1.0);
    MatrixXd endAcc = getTVector(degree, 2, 1.0);
    
    MatrixXd tEndCombined(degree + 1, 3);
    tEndCombined << endPos, endVel, endAcc;
    
    MatrixXd bD_MM = mMatrix * tEndCombined;

    return {bD_M0, bD_MM};
}

SVDResult MinSnapEvalNatural::createSVD(int16_t numControlPoints) {
    // 1. Resolve the M Matrix
    // Note: In C++, we skip the hardcoded M_STENCILS lookup for now. 
    // The C++ combinatorial generation is so fast it effectively acts as O(1).
    auto [mMatrixRaw, scalar] = getMMatrix(degree_);
    MatrixXd mMatrix = mMatrixRaw / scalar;

    // 2. Generate Boundary Blocks
    auto [bD_M0, bD_MM] = getBoundaryStates(mMatrix, degree_);

    // 3. Initialize full-size boundary matrices
    MatrixXd b0Full = MatrixXd::Zero(numControlPoints, 3);
    MatrixXd bMFull = MatrixXd::Zero(numControlPoints, 3);

    // Paste the active blocks into their respective ends
    int8_t windowSize = degree_ + 1;
    b0Full.topRows(windowSize) = bD_M0;
    bMFull.bottomRows(windowSize) = bD_MM;

    // Glue them together horizontally: [B(0) B(M)]
    MatrixXd bCombined(numControlPoints, 6);
    bCombined << b0Full, bMFull;

    // 4. Run the Singular Value Decomposition (BDCSVD)
    // We strictly instruct Eigen to compute the full U and V matrices
    BDCSVD<MatrixXd> svd(bCombined, ComputeFullU | ComputeFullV);

    // Extract the components
    int numConstraints = bCombined.cols(); // Should be 6
    
    SVDResult result;
    result.BCombined = bCombined;
    result.U1 = svd.matrixU().leftCols(numConstraints);
    result.U2 = svd.matrixU().rightCols(numControlPoints - numConstraints);
    result.Sigma = svd.singularValues().asDiagonal();
    
    // In Python, np.linalg.svd returns V transposed (Vh). 
    // Eigen returns V normally, so we transpose it here to match your exact Python logic.
    result.V = svd.matrixV(); 

    return result;
}

// ==========================================
// QP CONSTRAINT BUILDERS
// ==========================================

pair<MatrixXd, VectorXd> MinSnapEvalNatural::getSfcMatrices(
    const std::vector<SFCConstraint>& sfcConstraints, 
    const std::vector<int16_t>& numPtsList) 
{
    // 1. Pre-calculate the total number of inequalities to avoid dynamic memory reallocation
    int32_t totalInequalities = 0;
    for (size_t i = 0; i < sfcConstraints.size(); ++i) {
        totalInequalities += sfcConstraints[i].A.rows() * numPtsList[i];
    }
    
    // 2. Allocate the massive constraint matrices EXACTLY once
    int32_t totalColumns = numControlPoints_ * 3;
    MatrixXd aSfcTotal = MatrixXd::Zero(totalInequalities, totalColumns);
    VectorXd bSfcTotal = VectorXd::Zero(totalInequalities);
    
    int32_t currentRow = 0;
    int16_t startIdx = 0;
    
    // 3. Populate the constraints
    for (size_t i = 0; i < sfcConstraints.size(); ++i) {
        const MatrixXd& aMat = sfcConstraints[i].A;
        const VectorXd& bVec = sfcConstraints[i].b;
        
        int16_t numPtsInBox = numPtsList[i];
        int32_t numInequalities = aMat.rows();
        
        for (int16_t j = 0; j < numPtsInBox; ++j) {
            int16_t globalCpIndex = startIdx + j;
            int32_t colStart = globalCpIndex * 3;
            
            // Insert the A matrix block directly into the massive pre-allocated matrix
            // This replaces A_padded[:, col_start:col_end] = A_mat
            aSfcTotal.block(currentRow, colStart, numInequalities, 3) = aMat;
            
            // Insert the b vector directly into the massive pre-allocated vector
            bSfcTotal.segment(currentRow, numInequalities) = bVec;
            
            // Move our row pointer down
            currentRow += numInequalities;
        }
        
        // Step forward, accounting for the intersection overlap!
        startIdx += (numPtsInBox - degree_);
    }
    
    return {aSfcTotal, bSfcTotal};
}

// ==========================================
// PUBLIC GETTERS
// ==========================================

MatrixXd MinSnapEvalNatural::getQMatrix() const {
    return Q_;
}

VectorXd MinSnapEvalNatural::getKnots() const {
    return knots_;
}

MatrixXd MinSnapEvalNatural::getBCombined() const {
    return BCombined_;
}