import os
import numpy as np

def itteration_step(r1, r2, p1, m1, C):
    '''
        Parameters:
            r1: r_i, bigger/outer radius
            r2: r_i+1, smaller/inner radius
            p1: p_i, pressure at r_i
            m1: m_i, mass enclosed by r_i
            C: p0^(1-gamma) * T^gamma, 

        Returns:
            p2: p_i+1, pressure at r_i+1
            m2: m_i+1, mass enclosed by r_i+1
            T: T_i, temperature at r_i
            rho: rho_i, desity at r_i
    '''

    T = (C/(p1**(-2/3)))**(3/5)
    rho = (p1*M_mole)/(R*T)
    m2 = m1 - 4/3 * np.pi * (r1**3 - r2**3) * rho
    p2 = p1 - G * m2 * rho * (1/r1 + 1/r2)

    return m2, p2, T, rho

# ------------------ constants ------------------
M_mole =  1.008 # g/mole
R = 8.31434e7   # erg / (K * mole)
G = 6.67430e-8  # dyn cm^2 / g^2

# ------------- boundary conditions -------------
R_mean_Jupiter = 69911   # km
T_surface_Jupiter = 165  # K
P_surface_Jupiter = 1    # bar
M_Jupiter = 1.8982e27    # kg

# --------------- convert to cgs ----------------
R_mean_cgs = R_mean_Jupiter * 1e5        # cm
T_surface_cgs = T_surface_Jupiter        # K
P_surface_cgs = P_surface_Jupiter * 1e6  # dyn/cm^2
M_cgs = M_Jupiter * 1000                 # g

gamma = 5/3
C_cgs = P_surface_cgs**(1-gamma) * T_surface_cgs**gamma # (dyn * K)/cm^2

# ------------------- r_grid --------------------
'''
 We want to sampe more r values closer to the surface because P changes there more rapidly. 
'''
N = 100
R = R_mean_cgs
theta = 3                       # strech factor -> higher = more more dense close to the surface
s = np.linspace(0.0, 1.0, N+1)  # normalized coordinates
r_grid = R * (1 - s**theta)     # power-law stretched coordinates

# ---------------- prepare data -----------------
data = np.zeros((N,5)) # [r, m, p, T, rho]

data[0,0] = R_mean_cgs
data[0,1] = M_cgs
data[0,2] = P_surface_cgs
data[0,3] = T_surface_cgs
data[0,4] = M_cgs / (4/3*np.pi*R_mean_cgs**3)

for i in range(len(r_grid)-1):
    r1 = r_grid[i]
    r2 = r_grid[i+1]

    data[i,0] = r1
    m1 = data[i,1] 
    p1 = data[i,2]

    m2, p2, T, rho = itteration_step(r1, r2, p1, m1, C_cgs)

    data[i,3] = T
    data[i,4] = rho

    if i < len(r_grid)-2:
        data[i+1,1] = m2
        data[i+1,2] = p2

header = 'r [cm], m [g], p [dyn/cm^2], T [K], rho [g/cm^3]'
np.savetxt('data/Jupiter_01.csv', data, delimiter=',', header=header)