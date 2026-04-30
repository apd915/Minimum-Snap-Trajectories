import numpy as np

# ==========================================
# MATHEMATICAL CONSTANTS
# ==========================================
# M4: The basis matrix for a degree 4 (quartic) B-spline.
# Used to map control points to specific spatial/derivative constraints.
M4 = np.array([
    [ 1,  -4,   6,  -4,   1],
    [-4,  12,  -6, -12,  11],
    [ 6, -12,  -6,  12,  11],
    [-4,   4,   6,   4,   1],
    [ 1,   0,   0,   0,   0]
]) / 24.0

# M5: The basis matrix for a degree 5 (quintic) B-spline.
# Scalar factor: 1/120
M5 = np.array([
    [ -1,   5, -10,  10,  -5,   1],
    [  5, -20,  20,  20, -50,  26],
    [-10,  30,   0, -60,   0,  66],
    [ 10, -20, -20,  20,  50,  26],
    [ -5,   5,  10,  10,   5,   1],
    [  1,   0,   0,   0,   0,   0]
]) / 120.0

# M6: The basis matrix for a degree 6 (hextic) B-spline.
# Scalar factor: 1/720
M6 = np.array([
    [  1,  -6,  15, -20,  15,  -6,   1],
    [ -6,  30, -45, -20, 135,-150,  57],
    [ 15, -60,  30, 160,-150,-240, 302],
    [-20,  60,  30,-160,-150, 240, 302],
    [ 15, -30, -45,  20, 135, 150,  57],
    [ -6,   6,  15,  20,  15,   6,   1],
    [  1,   0,   0,   0,   0,   0,   0]
]) / 720.0

# M7: The basis matrix for a degree 7 (heptic) B-spline.
# Scalar factor: 1/5040
M7 = np.array([
    [  -1,    7,  -21,   35,  -35,   21,   -7,    1],
    [   7,  -42,   84,    0, -280,  504, -392,  120],
    [ -21,  105, -105, -315,  665,  315,-1715, 1191],
    [  35, -140,    0,  560,    0,-1680,    0, 2416],
    [ -35,  105,  105, -315, -665,  315, 1715, 1191],
    [  21,  -42,  -84,    0,  280,  504,  392,  120],
    [  -7,    7,   21,   35,   35,   21,    7,    1],
    [   1,    0,    0,    0,    0,    0,    0,    0]
]) / 5040.0


# ==========================================
# PRE-COMPUTED S-MATRIX STENCILS (Lookup Table)
# Key: k (Degree of the basis functions being integrated)
# Value: List of [Main Diagonal, 1st Off-Diagonal, 2nd Off-Diagonal...]
# ==========================================
S_STENCILS = {
    0: [1.0],                           # k=0 (Boxcars)
    1: [2/3, 1/6],                      # k=1 (Triangles)
    2: [11/20, 13/60, 1/120], # (Note: 13/60 is exactly 26/120)          # k=2 (Parabolas) 
    3: [151/315, 397/1680, 1/42, 1/5040],          # k=3 (Cubics) - To be calculated
}

# ==========================================
# PRE-COMPUTED DERIVATIVE STENCILS (Pascal's Triangle)
# Key: l (The derivative order. e.g., 4 for Snap)
# Value: The cascaded finite difference coefficients
# ==========================================
D_STENCILS = {
    1: [-1, 1],
    2: [1, -2, 1],
    3: [-1, 3, -3, 1],
    4: [1, -4, 6, -4, 1]
}

# ==========================================
# PRE-COMPUTED BOUNDARY VECTORS (T Matrices)
# Key: k (Degree of the basis functions)
# Value: Dictionary containing combined [Pos, Vel, Acc] matrices for tau=0 and tau=1
# ==========================================
T_STENCILS = {
    4: {
        'start': np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 2],
            [0, 1, 0],
            [1, 0, 0]
        ]),
        'end': np.array([
            [1, 4, 12],
            [1, 3,  6],
            [1, 2,  2],
            [1, 1,  0],
            [1, 0,  0]
        ])
    },
    5: {
        'start': np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 2],
            [0, 1, 0],
            [1, 0, 0]
        ]),
        'end': np.array([
            [1, 5, 20],
            [1, 4, 12],
            [1, 3,  6],
            [1, 2,  2],
            [1, 1,  0],
            [1, 0,  0]
        ])
    },
    6: {
        'start': np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 2],
            [0, 1, 0],
            [1, 0, 0]
        ]),
        'end': np.array([
            [1, 6, 30],
            [1, 5, 20],
            [1, 4, 12],
            [1, 3,  6],
            [1, 2,  2],
            [1, 1,  0],
            [1, 0,  0]
        ])
    },
    7: {
        'start': np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 2],
            [0, 1, 0],
            [1, 0, 0]
        ]),
        'end': np.array([
            [1, 7, 42],
            [1, 6, 30],
            [1, 5, 20],
            [1, 4, 12],
            [1, 3,  6],
            [1, 2,  2],
            [1, 1,  0],
            [1, 0,  0]
        ])
    }
}

# ==========================================
# PRE-COMPUTED BASIS MATRICES (M Matrices)
# Key: k (Degree of the basis functions)
# Value: The M matrix used to map control points to polynomial states
# ==========================================
M_STENCILS = {
    4: M4,
    5: M5,
    6: M6,
    7: M7
}