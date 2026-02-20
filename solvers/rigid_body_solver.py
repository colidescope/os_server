import numpy as np

def calculate_equilibrium_reactions(applied_force, applied_moment, reaction_points, cg):
    """
    This algorithm calculates the reaction forces at given support points given an applied force and moment at the centre of gravity.
    
    This function now accepts the centre of gravity (cg) coordinates in the global
    coordinate system. The reaction_points are given in the same coordinate system.
    ...which confirms to:
        X - Positive AFT
        Y - Positive RHS
        Z - Positive UP
        Mx, My, Mz - Comform to the right-hand rule
    
    Equilibrium conditions:
      1. Sum of reaction forces = - applied_force
      2. Sum of moments (evaluated about the centre of gravity) =
             - applied_moment
         with each moment computed as:
             (reaction_point - cg) x reaction_force
      
    The system is statically indeterminate. 
    We solve it by finding the minimum-norm solution using a least squares approach.
    
    Parameters:
      applied_force (list or array): [Fx, Fy, Fz] in Newtons.
      applied_moment (list or array): [Mx, My, Mz] in Newton-meters.
      reaction_points (array-like): List/array of support point coordinates, each as [x, y, z] (in meters).
      cg (list or array): Centre of gravity coordinates [x, y, z] (in meters).
    
    Returns:
      reactions (np.array): Array with shape (n, 3) where each row is the reaction force vector [Rx, Ry, Rz]
                            at a support point.
    """
    applied_force = np.array(applied_force, dtype=float)
    applied_moment = np.array(applied_moment, dtype=float)
    reaction_points = np.array(reaction_points, dtype=float)
    cg = np.array(cg, dtype=float)
    
    num_points = reaction_points.shape[0]
    
    # Create the unknown reaction forces vector:
    # eg. x = [R1x, R1y, R1z, R2x, R2y, R2z, R3x, R3y, R3z]^T for 3 supports (9 unknowns)
    
    # Build the equilibrium matrix A (6 equations x 9 unknowns)
    A = np.zeros((6, 3 * num_points))
    
    # 1. Force Equilibrium: Sum(R_i) = - applied_force:
    for i in range(num_points):
        A[0, 3 * i + 0] = 1.0  # x-component
        A[1, 3 * i + 1] = 1.0  # y-component
        A[2, 3 * i + 2] = 1.0  # z-component
    
    # 2. Moment Equilibrium: Sum((reaction_point - cg) x R_i) = - applied_moment
    # For each reaction point i, first compute the moment arm: r = reaction_point - cg.
    # The cross product, r x R, has components:
    #   Mx: (r_y * R_z - r_z * R_y)
    #   My: (r_z * R_x - r_x * R_z)
    #   Mz: (r_x * R_y - r_y * R_x)
    for i in range(num_points):
        r = reaction_points[i] - cg  # Moment arm relative to CG
        r_x, r_y, r_z = r
        
        # Moment equilibrium about x-axis:
        A[3, 3 * i + 1] = -r_z   # Coefficient for R_i_y
        A[3, 3 * i + 2] = r_y    # Coefficient for R_i_z
        
        # Moment equilibrium about y-axis:
        A[4, 3 * i + 0] = r_z    # Coefficient for R_i_x
        A[4, 3 * i + 2] = -r_x   # Coefficient for R_i_z
        
        # Moment equilibrium about z-axis:
        A[5, 3 * i + 0] = -r_y   # Coefficient for R_i_x
        A[5, 3 * i + 1] = r_x    # Coefficient for R_i_y

    # Construct right-hand side vector b:
    # [sum_R_x, sum_R_y, sum_R_z, Mx, My, Mz]^T = -[applied_force, applied_moment]^T
    b = np.zeros(6)
    b[0:3] = -applied_force
    b[3:6] = -applied_moment

    # Solve the underdetermined system using the least squares method.
    solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Reshape the solution into the reaction forces for each support point.
    reactions = solution.reshape((num_points, 3))
    
    return reactions

def vector_magnitude(vector):
    """Return the Euclidean norm of a vector."""
    return np.linalg.norm(vector)

def split_vector(v, u):
    v = np.array(v, dtype=float)
    u = np.array(u, dtype=float)

    v_parallel = np.dot(v, u) / np.dot(u, u) * u
    v_perpendicular = v - v_parallel

    return v_parallel, v_perpendicular
    
if __name__ == "__main__":
    # Applied loads at the centre of gravity
    applied_F = [0, 0, 0]    # in Newtons
    applied_M = [0, 0, 0]      # in Newton-meters

    # Define the centre of gravity coordinates (global system)
    cg = [0.0, 0.0, 0.0]  # Example: CG at the origin; change as needed

    # Define n reaction (support) points in the global coordinate system (meters)
    # Example: An equilateral triangle in the horizontal (XY) plane.
    reaction_points = [
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5]
    ]
    
    # Calculate the reaction forces at the three support points.
    reactions = calculate_equilibrium_reactions(applied_F, applied_M, reaction_points, cg)
    
    # Display the results:
    print("Applied Force (N):", applied_F)
    print("Applied Moment (N·m):", applied_M)
    print("Centre of Gravity (m):", cg, "\n")
    
    print("---- Reaction Points and Their Reaction Loads ----")
    for i, (point, reaction) in enumerate(zip(reaction_points, reactions), start=1):
        mag = vector_magnitude(reaction)
        print(f"Support {i} at {point} -> Reaction Force: {reaction} N, Magnitude: {mag:.2f} N")
