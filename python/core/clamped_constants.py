import numpy as np


# ---------------------------------------------------------
# THE CASCADED DERIVATIVE STENCIL LIBRARY
# Dictionary Keys: [Base Degree d][Derivative Level j]
# ---------------------------------------------------------
CASCADED_S_STENCILS = {
    # =========================================
    # DEGREE 4 (Standard Minimum Snap)
    # =========================================
    4: {
        # Velocity (j=1)
        1: {
            'interior': [-1, 1],
            'boundary_block': np.array([
                [-4.0,  4.0,  0.0,  0.0],
                [ 0.0, -2.0,  2.0,  0.0],
                [ 0.0,  0.0, -4/3,  4/3]
            ])
        },
        
        # Acceleration (j=2)
        2: {
            'interior': [1, -2, 1],
            'boundary_block': np.array([
                [ 12, -18,    6,     0,   0,  0],
                [  0,   3,   -5,     2,   0,  0],
                [  0,   0,  4/3,  -7/3,   1,  0],
                [  0,   0,    0,     1,  -2,  1]
            ])
        },
        
        # Snap (j=4)
        4: {
            'interior': [1, -4, 6, -4, 1],
            'boundary_block': np.array([
                [ 24, -45,  85/3, -25/3,   1,   0,  0,  0],
                [  0,   3, -23/3,  23/3,  -4,   1,  0,  0],
                [  0,   0,   4/3, -13/3,   6,  -4,  1,  0],
                [  0,   0,     0,     1,  -4,   6, -4,  1]
            ])
        }
    },

    # =========================================
    # DEGREE 5 (Continuous Minimum Snap)
    # =========================================
    5: {
        # Velocity (j=1)
        1: {
            'interior': [-1, 1],
            'boundary_block': np.array([
                [-5,    5,     0,    0,   0,  0],
                [ 0, -5/2,   5/2,    0,   0,  0],
                [ 0,    0,  -5/3,  5/3,   0,  0],
                [ 0,    0,     0, -5/4, 5/4,  0],
                [ 0,    0,     0,    0,  -1,  1]
            ])
        },
        
        # Acceleration (j=2)
        2: {
            'interior': [1, -2, 1],
            'boundary_block': np.array([
                [ 20, -30,    10,     0,    0,  0,  0],
                [  0,   5, -25/3,  10/3,    0,  0,  0],
                [  0,   0,  20/9, -35/9,  5/3,  0,  0],
                [  0,   0,     0,   5/4, -9/4,  1,  0],
                [  0,   0,     0,     0,    1, -2,  1]
            ])
        },
        
        # Snap (j=4)
        4: {
            'interior': [1, -4, 6, -4, 1],
            'boundary_block': np.array([
                [120, -225,   425/3,  -125/3,      5,   0,  0,  0,  0],
                [  0, 15/2, -325/18,  575/36, -77/12,   1,  0,  0,  0],
                [  0,    0,    20/9, -115/18,   43/6,  -4,  1,  0,  0],
                [  0,    0,       0,     5/4,  -17/4,   6, -4,  1,  0],
                [  0,    0,       0,       0,      1,  -4,  6, -4,  1]
            ])
        }
    }
}


# ---------------------------------------------------------
# 1. THE STENCIL LIBRARY
# Dictionary Key is (d - j), representing the polynomial degree
# ---------------------------------------------------------
INTEGRAL_STENCILS = {
    # Rectangles (e.g., d=4, j=4) -> Standard Identity Matrix!
    0: {
        'interior': [1],
        'boundary_block': np.array([[1]])
    },
    
    # Triangles (e.g., d=4, j=3 or d=2, j=1)
    1: { 
        'interior': [2/3, 1/6], 
        'boundary_block': np.array([[1/3]])
    },
    
    # Parabolas (e.g., d=5, j=3 or d=4, j=2)
    2: {
        'interior': [11/20, 13/60, 1/120],
        'boundary_block': np.array([
            [1/5, 7/60],
            [7/60, 1/3]
        ])
    },
    
    # Cubics (e.g., d=6, j=3) - Using your exact calculated values!
    3: {
        # The shift-invariant middle (Row 5 of your matrix)
        'interior': [0.479365079333, 0.2363095241, 0.0238095238095, 0.000198412694895],
        
        # The Top-Left 4x4 squished boundary block
        'boundary_block': np.array([
            [0.142857142857, 0.0875,         0.0184523809524, 0.00119047619046],
            [0.0875,         0.221428571428, 0.15625,         0.0345238095238 ],
            [0.0184523809524,0.15625,        0.326785714332,  0.224603174668  ],
            [0.00119047619046,0.0345238095238,0.224603174668,  0.479365079333  ]
        ])
    }
}

j=  \
    \
[[.142857142857,    .0875,              .0184523809524,     .00119047619046,  0,               0,                0,                 0,                0              ],
    
 [.0875,            .221428571428,      .15625,             .0345238095238,   .00029761904752, 0,                0,                 0,                0              ],
    
 [.0184523809524,   .15625,             .326785714332,      .224603174668,    .0237103174609,  .000198412694895, 0,                 0,                0              ],
  
 [.00119047619046,  .0345238095238,     .224603174668,      .479365079333,    .2363095241,     .0238095238095,   .000198412694895,  0,                0              ],
  
 [0,                .00029761904752,    .0237103174609,     .2363095241,      .479365079333,   .2363095241,      .0237103174609,    .00029761904752,  0              ],

 [0,                0,                  .000198412694895,   .0238095238095,   .2363095241,     .479365079333,    .224603174668,     .0345238095238,   .00119047619046],

 [0,                0,                  0,                  .000198412694895, .0237103174609,  .224603174668,    .326785714332,     .15625,           .0184523809524 ],
    
 [0,                0,                  0,                  0,                .00029761904752, .0345238095238,   .15625,            .221428571428,    .0875          ],

 [0,                0,                  0,                  0,                0,               .00119047619046,  .0184523809524,    .0875,            .142857142857  ]]




import numpy as np
from fractions import Fraction
import math

def get_D_matrix(k, M):
    """Generates a single derivative step matrix D^k."""
    size = M + k - 1
    diag = []
    
    # 1. Left boundary fractions (d/1, d/2, ... d/d-1)
    for i in range(1, k):
        diag.append(Fraction(k, i))
        
    # 2. Shift-invariant Identity middle (1, 1, 1...)
    num_ones = size - 2 * (k - 1)
    for i in range(num_ones):
        diag.append(Fraction(1, 1))
        
    # 3. Right boundary fractions (Mirrored)
    for i in range(k - 1, 0, -1):
        diag.append(Fraction(k, i))
        
    # 4. Construct the shifted full D matrix (-D_bar + shifted D_bar)
    D = np.zeros((size, size + 1), dtype=object)
    for r in range(size):
        D[r, r] = -diag[r]
        D[r, r+1] = diag[r]
        
    return D

def generate_cascaded_stencil(d, j):
    """Cascades the matrices and extracts the exact boundary block."""
    # Use a large M so the left and right boundaries don't collide during calculation
    M = 15 
    
    D_cascaded = None
    
    # Cascade from degree d down to (d - j + 1)
    # Order matters: C^(d-j) = D^(d-j+1) @ ... @ D^(d-1) @ D^d
    for k in range(d - j + 1, d + 1):
        D_current = get_D_matrix(k, M)
        print(f'k: {k}\nD = \n{D_current}\n')
        if D_cascaded is None:
            D_cascaded = D_current
        else:
            D_cascaded = np.dot(D_cascaded, D_current)
            
    # Calculate how many rows make up the squished boundary block
    # The squish affects exactly d rows for a cascaded system
    boundary_rows = d
    boundary_cols = d + j
    
    top_left_block = D_cascaded[:boundary_rows, :boundary_cols]
    
    # Print it out in a beautiful format ready for copy/pasting!
    print(f"--- CASCADED BOUNDARY BLOCK (d={d}, j={j}) ---")
    print("np.array([")
    for r in range(boundary_rows):
        row_str = "    [" + ", ".join([str(val) for val in top_left_block[r]]) + "]"
        if r < boundary_rows - 1:
            row_str += ","
        print(row_str)
    print("])")
    
    # Print the Pascal's Triangle Interior Band
    interior = D_cascaded[boundary_rows + 1, boundary_rows + 1 : boundary_rows + 1 + j + 1]
    print(f"\n--- INTERIOR PASCAL BAND ---")
    print("[" + ", ".join([str(val) for val in interior]) + "]")

# ==========================================
# Run the Generator
# ==========================================
if __name__ == "__main__":
    # Example: Continuous Minimum Snap (Degree 5, 4th Derivative)
    degree = 5
    derivative_level = 4
    
    generate_cascaded_stencil(degree, derivative_level)