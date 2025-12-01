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
            T: [K] temperature
    '''

    R = 8.31446261815324 # J / (K * mole) gas const
    
    T = (C/(p**(-2/3)))**(3/5) # monoatomic H
    rho = (p*M_mole)/(R*T)

    return rho, T

def simulate_ideal_gas(R_surf, T_surf, P_surf, M_surf, M_mole, N, theta, output_name, show_plot=False):
    '''
        Prameters:
            R_surf: [m] radius of planet
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
    print('start ideal gas')
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
        rho, T = EoS_ideal_gas(p1, C, M_mole)
        # -------------------------------------------

        data[i,3] = T
        data[i,4] = rho
        
        m2 = m1 - 4/3 * np.pi * (r1**3 - r2**3) * rho
        p2 = p1 + G * m2 * rho * (1/r2 - 1/r1)

        if i < len(data)-1:
            data[i+1,0] = r2
            data[i+1,1] = m2
            data[i+1,2] = p2

        # print(f'{i+1}/{N}', end='\r')

    # ----------------- save data -------------------
    data =  data[:-1] # the last line contains no useful data and can be removed

    save_data(data, output_name)
    plot_data(data, output_name, show_plot=show_plot)

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

    p *= 0.1 # dyne / cm^2
    K = 1.96e12 # (cm^2/g)^2 * dyne/cm^2
    
    rho = np.sqrt(p/K) # g/cm^2
    rho *= 1000 # kg/m^3
    
    return rho

def simulate_polytropic(R_surf, T_surf, P_surf, M_surf, N, theta, output_name, show_plot=False):
    '''
        Prameters:
            R_surf: [m] radius of planet
            T_surf: [K] surface temperature of planet
            P_surf: [Pa] surface pressure of planet
            M_surf: [kg] total mass of planet
            N: number of steps
            theta: strech factor -> higher = r_grid is more dense close to the surface
            output_name: name for data and plot file
    '''

    # --------------- constants -----------------
    G = 6.67430e-11         # m^3 / (kg s^2) grav const

    # ------------ inital conditions ------------
    r_grid, data = create_grids(R_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- run simulation --------------
    print('start polytropic')
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
        rho = EoS_polytropic(p1)
        # -------------------------------------------
        
        m2 = m1 - 4/3 * np.pi * (r1**3 - r2**3) * rho
        p2 = p1 + G * m2 * rho * (1/r2 - 1/r1)

        data[i,3] = T_surf
        data[i,4] = rho

        if i < len(data)-1:
            data[i+1,0] = r2
            data[i+1,1] = m2
            data[i+1,2] = p2

        # print(f'{i+1}/{N}', end='\r')

    # ----------------- save data -------------------
    data =  data[:-1] # the last line contains no useful data and can be removed

    save_data(data, output_name)
    plot_data(data, output_name, show_plot=show_plot)

# ================== analytical =================
def EoS_analytical_Fe(p, df):
    '''
        Parameters:
            p: [Pa] pressure
            df: look up table

        Returns:
            rho: [kg/m^3] density
    '''
    p *= 1e-9 # Pa -> Gpa 
    n = np.array([0.05845, 0.91754, 0.02119, 0.00282])
    A = np.array([54, 65, 57, 58])
    Z = np.array([26, 26, 26, 26])

    if p <= 2.09e4: # Vinet (look up)
        rho = np.interp(p, df['p'], df['rho'])
        rho *= 1e3 # Mg/m^3 -> kg/m^3
    else: # TFD
        kappa = 9.524e13 * Z**(-10/3)
        zeta = (p/kappa)**(1/5)
        epsilon = (3/(32 * np.pi**2 * Z**2))**(1/3)
        phi = 3**(1/3) / 20 + epsilon/(4 * 3**(1/3))
        x0 = 1/(zeta + phi)

        rho = sum(n*A)/sum(x0**3 / Z)
        rho *= 1e3 # Mg/m^3 -> kg/m^3

    return rho

def EoS_analytical_Si(p, df):
    '''
        Parameters:
            p: [Pa] pressure
            df: look up table

        Returns:
            rho: [kg/m^3] density
    '''
    p *= 1e-9 # Pa -> Gpa 
    n = np.array([0.9223, 0.0467, 0.0310])
    A = np.array([28, 29, 30])
    Z = np.array([14, 14, 14])

    if p <= 1.35e4: # BME4 (look up)
        rho = np.interp(p, df['p'], df['rho'])
        rho *= 1e3 # Mg/m^3 -> kg/m^3
    
    else: # TFD
        kappa = 9.524e13 * Z**(-10/3)
        zeta = (p/kappa)**(1/5)
        epsilon = (3/(32 * np.pi**2 * Z**2))**(1/3)
        phi = 3**(1/3) / 20 + epsilon/(4 * 3**(1/3))
        x0 = 1/(zeta + phi)

        rho = sum(n*A)/sum(x0**3 / Z)
        rho *= 1e3 # Mg/m^3 -> kg/m^3

    return rho

def simulate_analytical(R_surf, T_surf, P_surf, M_surf, element, N, theta, output_name, show_plot=False):
    '''
        Prameters:
            R_surf: [m] radius of planet
            T_surf: [K] surface temperature of planet
            P_surf: [Pa] surface pressure of planet
            M_surf: [kg] total mass of planet
            element: 'Fe', or 'Si'
            N: number of steps
            theta: strech factor -> higher = r_grid is more dense close to the surface
            output_name: name for data and plot file
    '''
    # --------------- constants -----------------
    G = 6.67430e-11 # m^3 / (kg s^2)

    # ------------- look up tables --------------
    if element == 'Fe':
        df = pd.read_csv('data/EoS_Fe/EoS_Fe.csv', names=['p', 'rho'], skiprows=1)
    elif element == 'Si':
        df = pd.read_csv('data/EoS_Si/EoS_Si.csv', names=['p', 'rho'], skiprows=1)
    else:
        print(f'invalid element: {element}')
        return

    # ------------ inital conditions ------------
    r_grid, data = create_grids(R_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- run simulation --------------
    print(f'start analytical {element}')
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
        if element == 'Fe':
            rho = EoS_analytical_Fe(p1, df)
        elif element == 'Si':
            rho = EoS_analytical_Si(p1, df)
        # -------------------------------------------
        
        m2 = m1 - 4/3 * np.pi * (r1**3 - r2**3) * rho
        p2 = p1 + G * m2 * rho * (1/r2 - 1/r1)

        data[i,3] = T_surf
        data[i,4] = rho

        if i < len(data)-1:
            data[i+1,0] = r2
            data[i+1,1] = m2
            data[i+1,2] = p2

        # print(f'{i+1}/{N}', end='\r')

    # ----------------- save data -------------------
    data =  data[:-1] # the last line contains no useful data and can be removed

    save_data(data, output_name)
    plot_data(data, output_name, show_plot=show_plot)

# ================== tabulated ==================
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
                current_temp = float(line.split(' T= ')[1])
            
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
                columns=columns
            )
    
    return blocks

def parse_EoS_H2O(filename, columns):
    df = pd.read_csv(
        filename, 
        skiprows=21, 
        sep='\\s+', 
        names=columns
    )

    df = df.sort_values(['temp', 'rho']).reset_index(drop=True)
    blocks = {T: block for T, block in df.groupby('temp')}
    return blocks

def EoS_tabulated_H(p, T, blocks):
    '''
        Parameter:
            p: [Pa] pressure
            T: [K] temperature
            blocks: 

        Returns:
            rho: [kg/m^3]
    '''
    p_query = p * 1e-9 # Pa -> GPa
    p_query = np.log10(p_query)

    # find closest temperature
    T_query = np.log10(T)
    T_closest = min(blocks.keys(), key=lambda temp: abs(temp - T_query))
    df = blocks[T_closest]

    # interpolate rho
    log_rho = np.interp(p_query, df['log_P'], df['log_rho'])

    rho = np.exp(log_rho) * 1000 # kg/m^3
    
    return rho, T_closest

def EoS_tabulated_H2O(p, T, blocks):
    '''
        Parameter:
            p: [Pa] pressure
            T: [K] temperature
            blocks: 

        Returns:
            rho: [kg/m^3]
    '''

    # find closest temperature
    T_query = T
    T_closest = min(blocks.keys(), key=lambda temp: abs(temp - T_query))
    df = blocks[T_closest]

    # interpolate rho
    rho = np.interp(p, df['press'], df['rho'])
    
    return rho, T_closest

def simulate_tabulated(R_surf, T_surf, P_surf, M_surf, element, filename, N, theta, output_name, show_plot=False):
    '''
        Prameters:
            R_surf: [m] radius of planet
            T_surf: [K] surface temperature of planet
            P_surf: [Pa] surface pressure of planet
            M_surf: [kg] total mass of planet
            element: 'H', or 'H2O'
            filename: path to tabulated data
            N: number of steps
            theta: strech factor -> higher = r_grid is more dense close to the surface
            output_name: name for data and plot file
    '''
    # --------------- constants -----------------
    G = 6.67430e-11 # m^3 / (kg s^2)

    # ------------- look up tables --------------
    if element == 'H':
        columns = [
            'log_T', 'log_P', 'log_rho', 'log_U', 'log_s', 
            'dlrho_dlT_P', 'dlrho_dlP_T', 'dlS_dlT_P', 'dlS_dlP_T', 'grad_ad'
        ]

        blocks = parse_EoS_H(filename, columns)

    elif element == 'H2O':
        if 'pt' in filename:
            columns = [
            'press', 'temp', 'rho', 'ad_grad', 's', 
            'u', 'c', 'mmw', 'x_ion', 'x_d', 'phase'
        ]
        elif 'rhot' in filename:
            columns = [
            'rho', 'temp', 'press', 'ad_grad', 's', 
            'u', 'c', 'mmw', 'x_ion', 'x_d', 'phase'
        ]
        elif 'rhou' in filename:
            columns = [
            'rho', 'u', 'press', 'temp', 'ad_grad', 's', 
            'w', 'mmw', 'x_ion', 'x_d', 'phase'
        ]
        else:
            print(f'invalid file: {filename}')

        blocks = parse_EoS_H2O(filename, columns)

    else:
        print(f'invalid element: {element}')
        return

    # ------------ inital conditions ------------
    r_grid, data = create_grids(R_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf
    data[0,3] = T_surf

    # ------------- run simulation --------------
    print(f'start tabulated {element}')
    for i in range(N):
        r1 = r_grid[i]
        r2 = r_grid[i+1]
        
        if r2 == 0.0: r2 = 1e-6 # handle div by 0 error

        m1 = data[i,1] 
        p1 = data[i,2]
        T1 = data[i,3]

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
        if element == 'H':
            rho, T_closest_idx = EoS_tabulated_H(p1, T1, blocks)
        elif element == 'H2O':
            rho, T_closest_idx = EoS_tabulated_H2O(p1, T1, blocks)
        # -------------------------------------------
        
        data[i,4] = rho
        
        m2 = m1 - 4/3 * np.pi * (r1**3 - r2**3) * rho
        p2 = p1 + G * m2 * rho * (1/r2 - 1/r1)

        # ---------- find next temperature ----------
        if element == 'H':
            p_query = p1 * 1e-9 # Pa -> GPa
            p_query = np.log10(p_query)

            df = blocks[T_closest_idx]
            closest_idx = (df['log_P'] - p_query).abs().idxmin()
            closest_row = df.loc[closest_idx]
            grad_ad = closest_row['grad_ad']

        elif element == 'H2O':
            p_query = p1

            df = blocks[T_closest_idx]
            closest_idx = (df['press'] - p_query).abs().idxmin()
            closest_row = df.loc[closest_idx]
            grad_ad = closest_row['ad_grad']

        T2 = T1/p1 * (p2 - p1) * grad_ad

        # --------- save data for next step ---------
        if i < len(data)-1:
            data[i+1,0] = r2
            data[i+1,1] = m2
            data[i+1,2] = p2
            data[i+1,3] = T2

        # print(f'{i+1}/{N}', end='\r')

    # ----------------- save data -------------------
    data =  data[:-1] # the last line contains no useful data and can be removed

    save_data(data, output_name)
    plot_data(data, output_name, show_plot=show_plot)

# ============== helper functions ===============
def create_grids(R, N, theta):
    # ------------------- r_grid --------------------
    '''
    We want to sampe more r values closer to the surface because P changes there more rapidly. 
    '''
    s = np.linspace(0.0, 1.0, N+1)          # normalized coordinates
    r_grid = R * (1 - s**theta)    # power-law stretched coordinates

    # ---------------- prepare data -----------------
    data = np.zeros((N+1,5)) # [r, m, p, T, rho]
    return r_grid, data

def plot_data(data, filename, show_plot=False):
    name_parts = filename.split('_')
    title = ''
    for part in name_parts[1:]:
        title += part + ' '

    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(data[:,0], data[:,1], '.-')
    ax1.set_yscale('log')
    ax1.set_xlabel('r [m]')
    ax1.set_ylabel('m [kg]')
    ax1.set_title('Mass')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2, sharex=ax1)
    ax2.plot(data[:,0], data[:,2], '.-')
    ax2.set_yscale('log')
    ax2.set_xlabel('r [m]')
    ax2.set_ylabel('p [Pa]')
    ax2.set_title('Pressure')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3, sharex=ax1)
    ax3.plot(data[:,0], data[:,3], '.-')
    ax3.set_yscale('log')
    ax3.set_xlabel('r [m]')
    ax3.set_ylabel('T [K]')
    ax3.set_title('Temperature')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4, sharex=ax1)
    ax4.plot(data[:,0], data[:,4], '.-')
    ax4.set_yscale('log')
    ax4.set_xlabel('r [m]')
    ax4.set_ylabel('$\\rho$ [kg/m^3]')
    ax4.set_title('Density')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.pdf')

    if show_plot:
        plt.show()

def save_data(data, filename):
    header = 'r [m], m [kg], p [Pa], T [K], rho [kg/m^3]'
    np.savetxt(f'data/{filename}.csv', data, delimiter=',', header=header)

if __name__ == '__main__':
    N = 100
    theta = 5

    # ----------- boundary conditions -----------
    R_mean_Jupiter = 69911000 # m
    T_surface_Jupiter = 165   # K
    P_surface_Jupiter = 1e5   # Pa
    M_Jupiter = 1.8982e27     # kg

    # M_mole_H = 1.008 * 0.001   # kg/mole
    M_mole_Jupiter = 2.22 * 0.001   # kg/mole https://radiojove.gsfc.nasa.gov/education/educationalcd/Posters&Fliers/FactSheets/JupiterFactSheet.pdf

    simulate_ideal_gas(
        R_mean_Jupiter, 
        T_surface_Jupiter,
        P_surface_Jupiter,
        M_Jupiter,
        M_mole_Jupiter,
        N=N,
        theta=theta,
        output_name='01_ideal_gas_Jupiter'
    )

    simulate_polytropic(
        R_mean_Jupiter, 
        T_surface_Jupiter,
        P_surface_Jupiter,
        M_Jupiter,
        N=N,
        theta=theta,
        output_name='02_polytropic_Jupiter'
    )

    simulate_analytical(
        R_mean_Jupiter, 
        T_surface_Jupiter,
        P_surface_Jupiter,
        M_Jupiter,
        element='Fe',
        N=N,
        theta=theta,
        output_name='03_analytical_Fe_Jupiter'
    )

    simulate_analytical(
        R_mean_Jupiter, 
        T_surface_Jupiter,
        P_surface_Jupiter,
        M_Jupiter,
        element='Si',
        N=N,
        theta=theta,
        output_name='04_analytical_Si_Jupiter'
    )

    simulate_tabulated(
        R_mean_Jupiter, 
        T_surface_Jupiter,
        P_surface_Jupiter,
        M_Jupiter,
        element='H',
        filename='data/EoS_H/TABLEEOS_2021_TP_Y0275_v1.csv',
        N=N,
        theta=theta,
        output_name='05_tabulated_H_Jupiter'
    )

    simulate_tabulated(
        R_mean_Jupiter, 
        T_surface_Jupiter,
        P_surface_Jupiter,
        M_Jupiter,
        element='H2O',
        filename='data/EoS_H2O/aqua_eos_pt_v1_0.dat',
        N=N,
        theta=theta,
        output_name='06_tabulated_H2O_Jupiter'
    )

    # plt.show()

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
