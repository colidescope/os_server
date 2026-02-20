import numpy as np
from scipy.optimize import fsolve

def t_section_properties(b, h, t):
    """
    Compute centroid and second moment of area for a T-section.
    b = flange width
    h = total height
    t = flange thickness
    """
    # Flange area
    A_flange = b * t
    y_flange = h - t/2  # centroid of flange from bottom
    
    # Web area
    A_web = t * (h - t)
    y_web = (h - t)/2   # centroid of web from bottom
    
    # Total area
    A_total = A_flange + A_web
    
    # Neutral axis (centroid)
    y_bar = (A_flange*y_flange + A_web*y_web) / A_total
    
    # Second moment of area (parallel axis theorem)
    I_flange = (b * t**3)/12 + A_flange*(y_flange - y_bar)**2
    I_web = (t * (h - t)**3)/12 + A_web*(y_web - y_bar)**2
    
    I_total = I_flange + I_web
    
    return I_total, y_bar

def u_section_properties(b, h, t):
    """
    Compute centroid and second moment of area for a U-section (channel).
    b = flange width
    h = total height
    t = thickness (same for flange and web)
    """
    # Top flange
    A_top = b * t
    y_top = h - t/2
    
    # Bottom flange
    A_bottom = b * t
    y_bottom = t/2
    
    # Web
    A_web = t * (h - 2*t)
    y_web = h/2  # centroid of web at mid-height
    
    # Total area
    A_total = A_top + A_bottom + A_web
    
    # Neutral axis (centroid)
    y_bar = (A_top*y_top + A_bottom*y_bottom + A_web*y_web) / A_total
    
    # Second moment of area (parallel axis theorem)
    I_top = (b * t**3)/12 + A_top*(y_top - y_bar)**2
    I_bottom = (b * t**3)/12 + A_bottom*(y_bottom - y_bar)**2
    I_web = (t * (h - 2*t)**3)/12 + A_web*(y_web - y_bar)**2
    
    I_total = I_top + I_bottom + I_web
    
    return I_total, y_bar

def section_modulus_t(b, h, t):
    I, y_bar = t_section_properties(b, h, t)
    y_top = h - y_bar
    y_bottom = y_bar
    Z = I / max(y_top, y_bottom)
    return Z

def section_modulus_u(b, h, t):
    I, y_bar = u_section_properties(b, h, t)
    y_top = h - y_bar
    y_bottom = y_bar
    Z = I / max(y_top, y_bottom)
    return Z

def optimum_thickness(b, h, M, sigma_allow, section="T", guess=0.01):
    """
    Solve for minimum flange thickness t that satisfies bending stress condition.
    """
    print("M", M)
    def equation(t):
        if section == "T":
            Z = section_modulus_t(b, h, t)
        elif section == "U":
            Z = section_modulus_u(b, h, t)
        else:
            return None
        return M/Z - sigma_allow
    
    t_solution, = fsolve(equation, guess)
    return t_solution

# # Example usage (T-section):
# b = 0.2       # flange width (m)
# h = 0.4       # total height (m)
# M = 50e3      # max bending moment (Nm)
# sigma_allow = 250e6  # allowable stress (Pa)

# t_opt = optimum_thickness(b, h, M, sigma_allow)
# print(f"Optimum flange thickness: {t_opt:.4f} m")

# # Example usage (U-section):
# b = 0.2       # flange width (m)
# h = 0.4       # total height (m)
# M = 50e3      # max bending moment (Nm)
# sigma_allow = 250e6  # allowable stress (Pa)

# t_opt = optimum_thickness(b, h, M, sigma_allow, "U")
# print(f"Optimum thickness for U-section: {t_opt:.4f} m")