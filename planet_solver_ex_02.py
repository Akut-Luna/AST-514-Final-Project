import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== ideal gas ==================
def EoS_ideal_gas(p, C, M_mole):
    '''
        Parameters:
            p: [Pa] pressure
            C: [Pa^(-2/3) * K^(5/3)] adiabtic constant
            M_mole: [kg/mole] mean molecular mass per mole

        Returns:
            rho: [kg/m^3] density
    '''

    R = 8.31446261815324 # J / (K * mole) gas const
    
    T = (C/(p**(-2/3)))**(3/5) # monoatomic H
    rho = (p*M_mole)/(R*T)

    return rho

def simulate_ideal_gas(R_surf, T_surf, P_surf, M_surf, M_mole, N, theta, output_name):
    '''
        Prameters:
            R_surf: [km] radius of planet
            T_surf: [K] surface temperature of planet
            P_surf: [Pa] surface pressure of planet
            M_surf: [kg] total mass of planet
            M_mole: [kg/mole] mean molecular mass per mole
            N: number of steps
            theta: strech factor -> higher = r_grid is more dense close to the surface
            output_name: name for data and plot file
    '''
    # --------------- constants -----------------
    G = 6.67430e-11 # m^3 / (kg s^2)

    gamma = 5/3 # monoatomic H
    C = P_surf**(1-gamma) * T_surf**gamma # Pa^{-2/3} * K^{5/3}

    # ------------ inital conditions ------------
    r_grid, data = create_grids(R_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- run simulation --------------
    print('start')
    for i in range(N):
        r1 = r_grid[i]
        r2 = r_grid[i+1]
        
        if r2 == 0.0: r2 = 1e-6 # handle div by 0 error

        m1 = data[i,1] 
        p1 = data[i,2]

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
        
        # ------------------- EoS -------------------
        rho = EoS_ideal_gas(p1, C, M_mole)
        # -------------------------------------------
        
        m2 = m1 - 4/3 * np.pi * (r1**3 - r2**3) * rho
        p2 = p1 + G * m2 * rho * (1/r2 - 1/r1)

        data[i,3] = T
        data[i,4] = rho

        if i < len(data)-1:
            data[i+1,0] = r2
            data[i+1,1] = m2
            data[i+1,2] = p2

        # print(f'{i+1}/{N}', end='\r')

    # ----------------- save data -------------------
    data =  data[:-1] # the last line contains no useful data and can be removed

    save_data(data, output_name)
    plot_data(data, output_name)

# ================== polytropic =================
def EoS_polytropic(p):
    '''
        Parameters:
            p: [Pa] pressure
        Returns:
            rho: [kg/m^3] density
    '''
    # 1 dyne/cm^2 = 0.1 Pa
    # 1 g/cm^3 = 1000 kg/m^3

    P *= 0.1 # dyne / cm^2
    K = 1.96e12 # (cm^2/g)^2 * dyne/cm^2
    
    rho = np.sqrt(p/K) # g/cm^2
    rho /= 1000 # kg/m^3
    
    return rho

def simulate_polytrop(R_surf, T_surf, P_surf, M_surf, M_mole, N, theta, output_name):
    '''
        Prameters:
            R_surf: [km] radius of planet
            T_surf: [K] surface temperature of planet
            P_surf: [Pa] surface pressure of planet
            M_surf: [kg] total mass of planet
            M_mole: [kg/mole] mean molecular mass per mole
            N: number of steps
            theta: strech factor -> higher = r_grid is more dense close to the surface
            output_name: name for data and plot file
    '''

    # --------------- constants -----------------
    R = 8.31446261815324    # J / (K * mole) gas const
    G = 6.67430e-11         # m^3 / (kg s^2) grav const

    gamma = 5/3 # monoatomic H
    C = P_surf**(1-gamma) * T_surf**gamma # Pa^{-2/3} * K^{5/3}

    # ------------ inital conditions ------------
    r_grid, data = create_grids(R_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- run simulation --------------
    print('start')
    for i in range(N):
        r1 = r_grid[i]
        r2 = r_grid[i+1]
        
        if r2 == 0.0: r2 = 1e-6 # handle div by 0 error

        m1 = data[i,1] 
        p1 = data[i,2]

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
        
        T = (C/(p1**(-2/3)))**(3/5) # monoatomic H
        rho = (p1*M_mole)/(R*T)
        
        m2 = m1 - 4/3 * np.pi * (r1**3 - r2**3) * rho
        p2 = p1 + G * m2 * rho * (1/r2 - 1/r1)

        data[i,3] = T
        data[i,4] = rho

        if i < len(data)-1:
            data[i+1,0] = r2
            data[i+1,1] = m2
            data[i+1,2] = p2

        # print(f'{i+1}/{N}', end='\r')

    # ----------------- save data -------------------
    data =  data[:-1] # the last line contains no useful data and can be removed

    save_data(data, output_name)
    plot_data(data, output_name)


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
def create_grids(R, N=100, theta=1):
    # ------------------- r_grid --------------------
    '''
    We want to sampe more r values closer to the surface because P changes there more rapidly. 
    '''
    s = np.linspace(0.0, 1.0, N+1)          # normalized coordinates
    r_grid = R * (1 - s**theta)    # power-law stretched coordinates

    # ---------------- prepare data -----------------
    data = np.zeros((N+1,5)) # [r, m, p, T, rho]
    return r_grid, data

def plot_data(data, filename):
    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(data[:,0], data[:,1], '.-')
    ax1.set_xlabel('r [m]')
    ax1.set_ylabel('m [kg]')
    ax1.set_title('Mass')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2, sharex=ax1)
    ax2.plot(data[:,0], data[:,2], '.-')
    ax2.set_xlabel('r [m]')
    ax2.set_ylabel('p [Pa]')
    ax2.set_title('Pressure')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3, sharex=ax1)
    ax3.plot(data[:,0], data[:,3], '.-')
    ax3.set_xlabel('r [mm]')
    ax3.set_ylabel('T [K]')
    ax3.set_title('Temperature')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4, sharex=ax1)
    ax4.plot(data[:,0], data[:,4], '.-')
    ax4.set_xlabel('r [m]')
    ax4.set_ylabel('$\\rho$ [kg/m^3]')
    ax4.set_title('Density')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/{filename}.pdf')
    plt.show()

def save_data(data, filename):
    header = 'r [m], m [kg], p [Pa], T [K], rho [kg/m^3]'
    np.savetxt(f'data/{filename}.csv', data, delimiter=',', header=header)


if __name__ == '__main__':

    # ----------- boundary conditions -----------
    R_mean_Jupiter = 69911   # km
    T_surface_Jupiter = 165  # K
    P_surface_Jupiter = 1e5  # Pa
    M_Jupiter = 1.8982e27    # kg

    # M_mole_H = 1.008 * 0.001   # kg/mole
    M_mole_Jupiter = 2.22 * 0.001   # kg/mole https://radiojove.gsfc.nasa.gov/education/educationalcd/Posters&Fliers/FactSheets/JupiterFactSheet.pdf

    simulate_ideal_gas(
        R_mean_Jupiter, 
        T_surface_Jupiter,
        P_surface_Jupiter,
        M_Jupiter,
        M_mole_Jupiter,
        N=100,
        theta=5,
        output_name='01_ideal_gas_Jupiter'
    )

    # # ------------------ constants ------------------
    # M_mole =  1.008 # g/mole (monoatomic H)
    # M_mole =  2.22   # g/mole https://radiojove.gsfc.nasa.gov/education/educationalcd/Posters&Fliers/FactSheets/JupiterFactSheet.pdf
    # R = 8.31434e7   # erg / (K * mole)
    # G = 6.67430e-8  # dyn cm^2 / g^2

    # # ------------- convert to cgs --------------
    # R_mean_cgs = R * 1e5     # cm
    # T_surface_cgs = T        # K
    # P_surface_cgs = P * 1e6  # dyn/cm^2
    # M_cgs = M * 1000         # g

    # # 1 bar = 10⁶ dyne/cm² = 10⁶ barye
    # # [C] = (10^6 barye)^{-2/3} * K^{5/3}
    # # [C] = 10^{-4} * barye^{-2/3} * K^{5/3}
    # # [C] = 10^{-4} * (dyne/cm^2)^{-2/3} * K^{5/3}
    # C_cgs = P_surface_cgs**(1-gamma) * T_surface_cgs**gamma # (dyn * K)/cm^2
    # print(C_cgs)

    # EoS_df_Fe = pd.read_csv('data/EoS_Fe/EoS_Fe.csv', names=['p', 'rho'], skiprows=1)
    # EoS_df_Si = pd.read_csv('data/EoS_Si/EoS_Si.csv', names=['p', 'rho'], skiprows=1)
    # EoS_blocks_H = parse_EoS_H(
    #     filename='data/EoS_H/TABLEEOS_2021_TP_Y0275_v1.csv', 
    #     columns=[
    #         'log_T', 'log_P', 'log_rho', 'log_U', 'log_S',
    #         'dlrho_dlT_P', 'dlrho_dlP_T', 'dlS_dlT_P',
    #         'dlS_dlP_T', 'grad_ad'
    #     ])
    # EoS_blocks_H2O = parse_EoS_H2O(
    #     filename='data/EoS_H2O/aqua_eos_pt_v1_0.dat',
    #     colums=[
    #         'press', 'temp', 'rho', 'ad_grad', 's', 
    #         'u', 'c', 'mmw', 'x_ion', 'x_d', 'phase'
    #     ])
    
    # simulate_Jupiter(N=100, theta=5)