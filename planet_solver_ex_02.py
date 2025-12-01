import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def itteration_step(EoS, r1, r2, p1, m1, C):
    '''
        Parameters:
            EoS: equ of state, must have arguments in a list [p, t, ...] TODO
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
    
    T = temp_from_pressure(p1, C) # TODO: FOR MONOATOMIC H -> finde other!!!
    rho = EoS(p1, T) # TODO find better way to generalise
    
    m2 = m1 - 4/3 * np.pi * (r1**3 - r2**3) * rho
    p2 = p1 + G * m2 * rho * (1/r2 - 1/r1)
        
    return m2, p2, T, rho


# ================== Ideal Gas ==================

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


# ================== polytropic =================

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

# ================== analytical =================

def EoS_analytical_Fe(p, T):
    p_Gpa = p * 1e-10
    n = [0.05845, 0.91754, 0.02119, 0.00282]
    A = [54, 65, 57, 58]
    Z = [26, 26, 26, 26]

    if p_Gpa <= 2.09e4: # Vinet (look up)
        rho = np.interp(p_Gpa, EoS_df_Fe['p'], EoS_df_Fe['rho'])
    else: # TFD
        kappa = 9.524e13 * Z**(-10/3)
        zeta = (p/kappa)**(1/5)
        epsilon = (3/(32 * np.pi**2 * Z**2))**(1/3)
        phi = 3**(1/3) / 20 + epsilon/(4 * 3**(1/3))
        x0 = 1/(zeta + phi)

        rho = sum(n*A)/sum(x0**3 / Z)

        # TODO: UNITS

    return rho

def EoS_analytical_Si(p, T):
    p_Gpa = p * 1e-10
    n = [0.9223, 0.0467, 0.0310]
    A = [28, 29, 30]
    Z = [14, 14, 14]

    if p_Gpa <= 1.35e4: # BME4 (look up)
        rho = np.interp(p_Gpa, EoS_df_Si['p'], EoS_df_Si['rho'])
    else: # TFD
        kappa = 9.524e13 * Z**(-10/3)
        zeta = (p/kappa)**(1/5)
        epsilon = (3/(32 * np.pi**2 * Z**2))**(1/3)
        phi = 3**(1/3) / 20 + epsilon/(4 * 3**(1/3))
        x0 = 1/(zeta + phi)

        rho = sum(n*A)/sum(x0**3 / Z)

        # TODO: UNITS

    return rho

# ================== tabulated ==================

def EoS_tabulated_H(p, T):
    # find closest temperature
    T_query = np.log10(T)
    T_closest = min(EoS_blocks_H.keys(), key=lambda temp: abs(temp - T_query))
    df = EoS_blocks_H[T_closest]

    # interpolate rho
    log_rho = np.interp(np.log10(p), df['log_P'], df['log_rho'])
    
    return np.exp(log_rho)

def EoS_tabulated_H2O(p, T):
    # find closest temperature
    T_query = T
    T_closest = min(EoS_blocks_H2O.keys(), key=lambda temp: abs(temp - T_query))
    df = EoS_blocks_H2O[T_closest]

    # interpolate rho
    rho = np.interp(p, df['press'], df['rho'])
    
    return rho

def parse_EoS_H(filename, columns):
    """
    Load data blocks separated by temperature headers
    
    Returns:
        dict: {temperature: dataframe} for each block
    """
    blocks = {}
    current_temp = None
    current_data = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Check if line is a temperature header
            if line.strip().startswith('#iT='):
                # Save previous block if exists
                if current_temp is not None and current_data:
                    blocks[current_temp] = pd.DataFrame(
                        current_data,
                        columns=columns
                    )
                    current_data = []
                
                # Extract temperature from header
                current_temp = float(line.split('T= ')[1])
            
            # Skip comment lines
            elif line.strip().startswith('#'):
                continue
            
            # Parse data lines
            elif line.strip():
                values = [float(x) for x in line.split()]
                current_data.append(values)
        
        # Save last block
        if current_temp is not None and current_data:
            blocks[current_temp] = pd.DataFrame(
                current_data,
                columns=[
                    'log_T', 'log_P', 'log_rho', 'log_U', 'log_S',
                    'dlrho_dlT_P', 'dlrho_dlP_T', 'dlS_dlT_P',
                    'dlS_dlP_T', 'grad_ad'
                ]
            )
    
    return blocks

def parse_EoS_H2O(filename, colums):
    df = pd.read_csv(
        filename, 
        skiprows=21, 
        sep='\\s+', 
        names=colums
    )

    df = df.sort_values(['temp', 'rho']).reset_index(drop=True)
    blocks = {T: block for T, block in df.groupby('temp')}
    return blocks

# ============== helper functions ===============

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


if __name__ == '__main__':

    # ------------------ constants ------------------
    # M_mole =  1.008 # g/mole (monoatomic H)
    M_mole =  2.22   # g/mole https://radiojove.gsfc.nasa.gov/education/educationalcd/Posters&Fliers/FactSheets/JupiterFactSheet.pdf
    R = 8.31434e7   # erg / (K * mole)
    G = 6.67430e-8  # dyn cm^2 / g^2

    EoS_df_Fe = pd.read_csv('data/EoS_Fe/EoS_Fe.csv', names=['p', 'rho'], skiprows=1)
    EoS_df_Si = pd.read_csv('data/EoS_Si/EoS_Si.csv', names=['p', 'rho'], skiprows=1)
    EoS_blocks_H = parse_EoS_H(
        filename='data/EoS_H/TABLEEOS_2021_TP_Y0275_v1.csv', 
        columns=[
            'log_T', 'log_P', 'log_rho', 'log_U', 'log_S',
            'dlrho_dlT_P', 'dlrho_dlP_T', 'dlS_dlT_P',
            'dlS_dlP_T', 'grad_ad'
        ])
    EoS_blocks_H2O = parse_EoS_H2O(
        filename='data/EoS_H2O/aqua_eos_pt_v1_0.dat',
        colums=[
            'press', 'temp', 'rho', 'ad_grad', 's', 
            'u', 'c', 'mmw', 'x_ion', 'x_d', 'phase'
        ])
    
    simulate_Jupiter(N=100, theta=5)