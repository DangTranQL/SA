import numpy as np

# Normalized Sensitivity functions
def S_alpha_xss_analytic(xss, alpha, n):                                                                     
    numer = alpha * (1 + xss**n)                                                                         
    denom = xss + alpha * n * xss**n + 2 * xss**(1+n) + xss**(1+2*n)                                     
    sensitivity = numer/denom                                                                        
    return abs(sensitivity)                                                                          
                                                        
def S_n_xss_analytic(xss, alpha, n):                                                                     
    numer = alpha * n * np.log(xss) * xss**(n-1)                                                         
    denom = 1 + alpha * n * xss**(n-1) + 2 * xss**(n) + xss**(2*n)                                       
    sensitivity = - numer/denom                                                                      
    return abs(sensitivity)/0.88972468

# ODEs of circuit
def Equ1(x, alpha, n):
    try:
        return (alpha / (1 + x**n)) - x
    except:
        return float('nan')

# Define function to evaluate vector field
def Equs(P, t, params):
    x = P[0]
    alpha = params[0]
    n     = params[1]
    val0 = Equ1(x, alpha, n)
    return np.array([val0])

# Define initial guesses for solving ODEs
def generate_initial_guesses(alpha_val, n_val):
    return [
        np.array([2]),
        np.array([0.5]),
        np.array([4.627])
    ]