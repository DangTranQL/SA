# Prelim
import numpy as np

# ODEs
def Equ1(x, y, beta_x, n):
    return beta_x / (1 + y**n) - x
def Equ2(x, y, beta_y, n):
    return beta_y * x**n / (1 + x**n) - y

# Sensitivity functions
def S_betax_xss_analytic(xss, yss, beta_x, beta_y, n):
    numer = beta_x * (1 + xss**n) * (1 + yss**n)
    denom = n**2 * beta_x * yss**n + xss * (1 + yss**n)**2 + xss**(1+n) * (1 + yss**n)**2
    sensitivity = numer/denom
    return abs(sensitivity)
def S_betax_yss_analytic(xss, yss, beta_x, beta_y, n):
    numer = n * beta_x * beta_y * xss**(n-1)
    denom = (1 + xss**n)**2 * yss + n**2 * beta_y * xss**n * yss**n + (1 + xss**n)**2 * yss**(1+n)
    sensitivity = numer/denom  
    return abs(sensitivity)
def S_betay_xss_analytic(xss, yss, beta_x, beta_y, n):
    numer = n * beta_x * (1 + xss**n) * yss**n
    denom = n**2 * beta_x * yss**n + xss * (1 + yss**n)**2 + xss**(1+n) * (1 + yss**n)**2
    sensitivity = - numer/denom    
    return abs(sensitivity)
def S_betay_yss_analytic(xss, yss, beta_x, beta_y, n):
    numer = beta_x * beta_y * xss**(n-1) * (1 + xss**n)
    denom = (1 + xss**n)**2 * yss + n**2 * beta_y * xss**n * yss**n + (1 + xss**n)**2 * yss**(1+n)
    sensitivity = numer/denom 
    return abs(sensitivity)
def S_n_xss_analytic(xss, yss, beta_x, beta_y, n):
    numer = n * beta_x * (n * np.log(xss) + np.log(yss) + np.log(yss) * xss**n) * yss**n
    denom = n**2 * beta_x * yss**n + xss * (1 + yss**n)**2 + xss**(1+n) * (1 + yss**n)**2
    sensitivity = - numer/denom   
    return abs(sensitivity)
def S_n_yss_analytic(xss, yss, beta_x, beta_y, n):
    numer = n * beta_y * xss**n * (np.log(xss) + (np.log(xss) - n * np.log(yss)) * yss**n)
    denom = (1 + xss**n)**2 * yss + n**2 * beta_y * xss**n * yss**n + (1 + xss**n)**2 * yss**(1+n)
    sensitivity = numer/denom  
    return abs(sensitivity)

# Initial guesses
def generate_initial_guesses(beta_x_val, beta_y_val):
    return [
        np.array([1,1]),
        np.array([0,0]),
        np.array([0.3,12.5]),
        np.array([50.5,0.9]),
        np.array([67.6,0.9]),
        np.array([21,0.9])
    ]