import os
import numpy as np
import matplotlib.pyplot as plt

def itteration_step(EoS, r1, r2, p1, m1, C):
    '''
        Parameters:
            EoS: equ of state, must have arguments in a list [p, t, ...]
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

    # handle div by 0 error
    if r2 == 0.0:
        r2 = 0.1 # small enough

    # ================== DEBUG ==================
    if m1 < 0:
        print('\n Negative mass!!! \n')
        print(f'r1: {r1}')
        print(f'r2: {r2}')
        print(f'p1: {p1}')
        print(f'm1: {m1}')
        print('')

    if p1 < 0:
        print('\n Negative pressure!!! \n')
        print(f'r1: {r1}')
        print(f'r2: {r2}')
        print(f'p1: {p1}')
        print(f'm1: {m1}')
        print('')
    # ===========================================
    
    T = temp_from_pressure(p1, C) # FOR MONOATOMIC H!!!
    rho = EoS(p1, T)
    
    m2 = m1 - 4/3 * np.pi * (r1**3 - r2**3) * rho
    p2 = p1 + G * m2 * rho * (1/r2 - 1/r1)
        
    return m2, p2, T, rho

def temp_from_pressure(p, C):
    return (C/(p**(-2/3)))**(3/5)

def EoS_ideal_gas(p, T):
    '''
        Parameters:
            p: pressure
            T: temperature
        Returns:
            rho: density
    '''
    return (p*M_mole)/(R*T)

def EoS_polytropic(p, T):
    '''
        Parameters:
            p: pressure
            T: temperature
        Returns:
            rho: density
    '''
    K = 1.96e12
    return np.sqrt(p/K)

def EoS_analytical_iron(p, T):
    p_Gpa = p * 1e-10
    
    K0 = 156.2 # GPa
    K0_prime = 6.08
    rho0 = 8.30 # Mg/m^3


def EoS_analytical_silicate(p, T):
    pass

def EoS_tabulated_H(p, T):
    pass

def EoS_tabulated_H2O(p, T):
    pass

def plot_data(data, filename):
    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(data[:,0], data[:,1], '.-')
    ax1.set_xlabel('r [cm]')
    ax1.set_ylabel('m [g]')
    ax1.set_title('Mass')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2, sharex=ax1)
    ax2.plot(data[:,0], data[:,2], '.-')
    ax2.set_xlabel('r [cm]')
    ax2.set_ylabel('p [dyn/cm^2]')
    ax2.set_title('Pressure')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3, sharex=ax1)
    ax3.plot(data[:,0], data[:,3], '.-')
    ax3.set_xlabel('r [cm]')
    ax3.set_ylabel('T [K]')
    ax3.set_title('Temperature')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4, sharex=ax1)
    ax4.plot(data[:,0], data[:,4], '.-')
    ax4.set_xlabel('r [cm]')
    ax4.set_ylabel('$\\rho$ [g/cm^3]')
    ax4.set_title('Density')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/{filename}.pdf')
    plt.show()

def save_data(data, filename):
    header = 'r [cm], m [g], p [dyn/cm^2], T [K], rho [g/cm^3]'
    np.savetxt(f'data/{filename}.csv', data, delimiter=',', header=header)

def simulate_Jupiter(N, theta):
    '''
        Prameters:
            N: number of steps
            theta: strech factor -> higher = r_grid is more dense close to the surface
    '''

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
    s = np.linspace(0.0, 1.0, N+1)          # normalized coordinates
    r_grid = R_mean_cgs * (1 - s**theta)    # power-law stretched coordinates

    # ---------------- prepare data -----------------
    data = np.zeros((N+1,5)) # [r, m, p, T, rho]

    data[0,0] = R_mean_cgs
    data[0,1] = M_cgs
    data[0,2] = P_surface_cgs

    # --------------- run simulation ----------------
    for i in range(N):
        r1 = r_grid[i]
        r2 = r_grid[i+1]

        m1 = data[i,1] 
        p1 = data[i,2]

        # m2, p2, T, rho = itteration_step(EoS_ideal_gas, r1, r2, p1, m1, C_cgs)
        m2, p2, T, rho = itteration_step(EoS_polytropic, r1, r2, p1, m1, C_cgs)

        data[i,3] = T
        data[i,4] = rho

        if i < len(data)-1:
            data[i+1,0] = r2
            data[i+1,1] = m2
            data[i+1,2] = p2

        # print(f'{i+1}/{N}', end='\r')

    # ----------------- save data -------------------
    data =  data[:-1] # the last line contains no useful data and can be removed

    save_data(data, 'ex_02_Jupiter')
    plot_data(data, 'ex_02_Jupiter')

if __name__ == '__main__':

    # ------------------ constants ------------------
    # M_mole =  1.008 # g/mole (monoatomic H)
    M_mole =  2.22   # g/mole https://radiojove.gsfc.nasa.gov/education/educationalcd/Posters&Fliers/FactSheets/JupiterFactSheet.pdf
    R = 8.31434e7   # erg / (K * mole)
    G = 6.67430e-8  # dyn cm^2 / g^2

    simulate_Jupiter(N=100, theta=1)