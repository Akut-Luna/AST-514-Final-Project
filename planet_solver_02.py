import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== ideal gas ==================
def EoS_ideal_gas(P, C):
    '''
        Parameters:
            P: [dyne/cm^2] pressure

        Returns:
            rho: [g/cm^3] density
    '''

    # Calculate density from EOS: P = K * rho^gamma
    gamma = 5/3
    rho = (P / C) ** (1/gamma)

    return rho

def simulate_ideal_gas(R_surf, M_surf, P_surf, N, theta, output_name, show_plot=False):
    '''
        Prameters:
            R_surf: [cm] radius of planet
            P_surf: [dyne/cm^2] surface pressure of planet
            M_surf: [g] total mass of planet
            N: number of steps
            theta: strech factor -> higher = r_grid is more dense close to the surface
            output_name: name for data and plot file
    '''

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
        P1 = data[i,2]
        
        # ------------------- EoS -------------------
        rho = EoS_ideal_gas(P1, C_ideal_gas) # TODO
        # -------------------------------------------

        data[i,3] = rho

        # Calculate derivatives
        dm_dr = 4 * np.pi * r1**2 * rho
        dP_dr = -G * m1 * rho / r1**2
        
        # Update values (Euler method) TODO: RK4?
        dr = r2 - r1

        m2 = m1 + dm_dr * dr
        P2 = P1 + dP_dr * dr
        
        # -------- check if data makes sense --------
        if P2 < 0.0:
            print('>'*25, 'ALERT', '<'*25)
            print(f'negative pressure at {i+1}/{N}')
            break

        if m2 < 0.0:
            print(f'aborted at {i+1}/{N}')
            break # abort sim

        if i < len(data)-1:
            data[i+1,0] = r2
            data[i+1,1] = m2
            data[i+1,2] = P2

        # print(f'{i+1}/{N}', end='\r')

    # ----------------- save data -------------------
    
    data =  data[:-(N-i)] # the last lines contain no useful/realistic data and can be removed

    save_data(data, output_name)
    plot_data(data, output_name, show_plot=show_plot)

# ================== polytropic =================
def EoS_polytropic(P):
    '''
        Parameters:
            P: [dyne/cm^2] pressure
        Returns:
            rho: [g/cm^3] density
    '''

    K = 1.96e12 # (cm^2/g)^2 * dyne/cm^2
    rho = np.sqrt(P/K) # g/cm^2
    
    return rho

def simulate_polytropic(R_surf, M_surf, P_surf, N, theta, output_name, show_plot=False):
    '''
        Prameters:
            R_surf: [cm] radius of planet
            M_surf: [g] total mass of planet
            P_surf: [dyne/cm^2] surface pressure of planet
            N: number of steps
            theta: strech factor -> higher = r_grid is more dense close to the surface
            output_name: name for data and plot file
    '''

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
        P1 = data[i,2]
        
        # ------------------- EoS -------------------
        rho = EoS_polytropic(P1)
        # -------------------------------------------
        
        data[i,3] = rho
        
        # Calculate derivatives
        dm_dr = 4 * np.pi * r1**2 * rho
        dP_dr = -G * m1 * rho / r1**2
        
        # Update values (Euler method) TODO: RK4?
        dr = r2 - r1

        m2 = m1 + dm_dr * dr
        P2 = P1 + dP_dr * dr
        
        # -------- check if data makes sense --------
        if P2 < 0.0:
            print('>'*25, 'ALERT', '<'*25)
            print(f'negative pressure at {i+1}/{N}')
            break

        if m2 < 0.0:
            print(f'aborted at {i+1}/{N}')
            break # abort sim

        if i < len(data)-1:
            data[i+1,0] = r2
            data[i+1,1] = m2
            data[i+1,2] = P2

        # print(f'{i+1}/{N}', end='\r')

    # ----------------- save data -------------------
    data =  data[:-(N-i)] # the last lines contain no useful/realistic data and can be removed

    save_data(data, output_name)
    plot_data(data, output_name, show_plot=show_plot)

# ================== analytical =================
def EoS_analytical_Fe(P, df):
    '''
        Parameters:
            P: [dyne/cm^2] pressure
            df: look up table

        Returns:
            rho: [g/cm^3] density
    '''
    P *= 1e-10 # dyne/cm^2 -> Gpa 
    n = np.array([0.05845, 0.91754, 0.02119, 0.00282])
    A = np.array([54, 65, 57, 58])
    Z = np.array([26, 26, 26, 26])

    if P <= 2.09e4: # Vinet (look up)
        rho = np.interp(P, df['p'], df['rho']) # Mg/m^3 = g/cm^3
    else: # TFD
        kappa = 9.524e13 * Z**(-10/3)
        zeta = (P/kappa)**(1/5)
        epsilon = (3/(32 * np.pi**2 * Z**2))**(1/3)
        phi = 3**(1/3) / 20 + epsilon/(4 * 3**(1/3))
        x0 = 1/(zeta + phi)

        rho = sum(n*A)/sum(x0**3 / Z) # Mg/m^3 = g/cm^3

    return rho

def EoS_analytical_MgSiO3(P, df):
    '''
        Parameters:
            p: [dyne/cm^2] pressure
            df: look up table

        Returns:
            rho: [g/cm^3] density
    '''
    P *= 1e-10 # dyne/cm^2 -> Gpa 
    n = np.array([0.9223, 0.0467, 0.0310])
    A = np.array([28, 29, 30])
    Z = np.array([14, 14, 14])

    if P <= 1.35e4: # BME4 (look up)
        rho = np.interp(P, df['p'], df['rho']) # Mg/m^3 = g/cm^3
    
    else: # TFD
        kappa = 9.524e13 * Z**(-10/3)
        zeta = (P/kappa)**(1/5)
        epsilon = (3/(32 * np.pi**2 * Z**2))**(1/3)
        phi = 3**(1/3) / 20 + epsilon/(4 * 3**(1/3))
        x0 = 1/(zeta + phi)

        rho = sum(n*A)/sum(x0**3 / Z) # Mg/m^3 = g/cm^3

    return rho

def simulate_analytical(R_surf, M_surf, P_surf, element, N, theta, output_name, show_plot=False):
    '''
        Prameters:
            R_surf: [cm] radius of planet
            M_surf: [g] total mass of planet
            P_surf: [dyne/cm^2] surface pressure of planet
            element: 'Fe', or 'MgSiO3'
            N: number of steps
            theta: strech factor -> higher = r_grid is more dense close to the surface
            output_name: name for data and plot file
    '''

    # ------------- look up tables --------------
    if element == 'Fe':
        df = pd.read_csv('data/EoS_Fe/EoS_Fe.csv', names=['p', 'rho'], skiprows=1)
    elif element == 'MgSiO3':
        df = pd.read_csv('data/EoS_MgSiO3/EoS_MgSiO3.csv', names=['p', 'rho'], skiprows=1)
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
        P1 = data[i,2]
        
        # ------------------- EoS -------------------
        if element == 'Fe':
            rho = EoS_analytical_Fe(P1, df)
        elif element == 'MgSiO3':
            rho = EoS_analytical_MgSiO3(P1, df)
        # -------------------------------------------
        
        data[i,3] = rho
        
        # Calculate derivatives
        dm_dr = 4 * np.pi * r1**2 * rho
        dP_dr = -G * m1 * rho / r1**2
        
        # Update values (Euler method) TODO: RK4?
        dr = r2 - r1

        m2 = m1 + dm_dr * dr
        P2 = P1 + dP_dr * dr
        
        # -------- check if data makes sense --------
        if P2 < 0.0:
            print('>'*25, 'ALERT', '<'*25)
            print(f'negative pressure at {i+1}/{N}')
            break

        if m2 < 0.0:
            print(f'aborted at {i+1}/{N}')
            break # abort sim

        if i < len(data)-1:
            data[i+1,0] = r2
            data[i+1,1] = m2
            data[i+1,2] = P2

        # print(f'{i+1}/{N}', end='\r')

    # ----------------- save data -------------------
    data =  data[:-(N-i)] # the last lines contain no useful/realistic data and can be removed

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

def EoS_tabulated_H(P, T, blocks):
    '''
        Parameters:
            P: [dyne/cm^2] pressure
            T: [K] temperature
            blocks: 

        Returns:
            rho: [g/cm^3]
    '''
    P_query = P * 1e-10 # dyne/cm^2 -> GPa
    P_query = np.log10(P_query)
    T_query = np.log10(T) # K

    # find closest temperature
    T_closest = min(blocks.keys(), key=lambda temp: abs(temp - T_query))
    df = blocks[T_closest]

    # interpolate rho
    log_rho = np.interp(P_query, df['log_P'], df['log_rho'])
    rho = np.exp(log_rho) # g/cm^3
    
    return rho, T_closest

def EoS_tabulated_H2O(P, T, blocks):
    '''
        Parameters:
            P: [dyne/cm^2] pressure
            T: [K] temperature
            blocks: 

        Returns:
            rho: [g/cm^3]
    '''

    P_query = P * 0.1 # dyne/cm^2 -> Pa
    T_query = T # K
    
    # find closest temperature
    T_closest = min(blocks.keys(), key=lambda temp: abs(temp - T_query))
    df = blocks[T_closest]

    # interpolate rho
    rho = np.interp(P_query, df['press'], df['rho'])
    rho *= 1e-3 # kg/m^3 -> g/cm^3
    
    return rho, T_closest

def simulate_tabulated(R_surf, M_surf, P_surf, T_surf, element, filename, N, theta, output_name, show_plot=False):
    '''
        Prameters:
            R_surf: [cm] radius of planet
            M_surf: [g] total mass of planet
            P_surf: [dyne/cm^2] surface pressure of planet
            T_surf: [K] surface temperature of planet
            element: 'H', or 'H2O'
            filename: path to tabulated data
            N: number of steps
            theta: strech factor -> higher = r_grid is more dense close to the surface
            output_name: name for data and plot file
    '''
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
    data[0,4] = T_surf

    # ------------- run simulation --------------
    print(f'start tabulated {element}')
    for i in range(N):
        r1 = r_grid[i]
        r2 = r_grid[i+1]
        
        if r2 == 0.0: r2 = 1e-6 # handle div by 0 error

        m1 = data[i,1] 
        P1 = data[i,2]
        T1 = data[i,4]
        
        # ------------------- EoS -------------------
        if element == 'H':
            rho, T_closest_idx = EoS_tabulated_H(P1, T1, blocks)
        elif element == 'H2O':
            rho, T_closest_idx = EoS_tabulated_H2O(P1, T1, blocks)
        # -------------------------------------------
        
        data[i,3] = rho
        
        # Calculate derivatives
        dm_dr = 4 * np.pi * r1**2 * rho
        dP_dr = -G * m1 * rho / r1**2
        
        # Update values (Euler method) TODO: RK4?
        dr = r2 - r1

        m2 = m1 + dm_dr * dr
        P2 = P1 + dP_dr * dr
        
        # ---------- find next temperature ----------
        if element == 'H':
            p_query = P1 * 1e-9 # Pa -> GPa
            p_query = np.log10(p_query)

            df = blocks[T_closest_idx]
            closest_idx = (df['log_P'] - p_query).abs().idxmin()
            closest_row = df.loc[closest_idx]
            grad_ad = closest_row['grad_ad']

        elif element == 'H2O':
            p_query = P1

            df = blocks[T_closest_idx]
            closest_idx = (df['press'] - p_query).abs().idxmin()
            closest_row = df.loc[closest_idx]
            grad_ad = closest_row['ad_grad']

        T2 = T1/P1 * (P2 - P1) * grad_ad + T1

        # -------- check if data makes sense --------
        if P2 < 0.0 or T2 < 0.0:
            print('>'*25, 'ALERT', '<'*25)
            if P2 < 0.0: print(f'negative pressure at {i+1}/{N}')
            else: print(f'negative temperature at {i+1}/{N}')
            break
            
        if m2 < 0.0:
            print(f'aborted at {i+1}/{N}')
            break # abort sim

        # --------- save data for next step ---------
        if i < len(data)-1:
            data[i+1,0] = r2
            data[i+1,1] = m2
            data[i+1,2] = P2
            data[i+1,4] = T2

        # print(f'{i+1}/{N}', end='\r')

    # ----------------- save data -------------------
    data =  data[:-(N-i)] # the last lines contain no useful/realistic data and can be removed

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
    # ax1.set_yscale('log')
    ax1.set_xlabel('r [cm]')
    ax1.set_ylabel('m [g]')
    ax1.set_title('Mass')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2, sharex=ax1)
    ax2.plot(data[:,0], data[:,2], '.-')
    ax2.set_yscale('log')
    ax2.set_xlabel('r [cm]')
    ax2.set_ylabel('p [dyne cm$^{-2}$]')
    ax2.set_title('Pressure')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3, sharex=ax1)
    ax3.plot(data[:,0], data[:,3], '.-')
    ax3.set_yscale('log')
    ax3.set_xlabel('r [m]')
    ax3.set_ylabel('$\\rho$ [g cm$^{-3}$]')
    ax3.set_title('Density')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4, sharex=ax1)
    ax4.plot(data[:,0], data[:,4], '.-')
    # ax4.set_yscale('log')
    ax4.set_xlabel('r [cm]')
    ax4.set_ylabel('T [K]')
    ax4.set_title('Temperature')
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

    # TODO: ideal gas const
    # TODO: 3D interpolate
    # TODO: RK4

    # ----------- Physical Constants ------------
    G = 6.67430e-8  # cm^3 g^-1 s^-2
    
    C_ideal_gas = 1.96e12 # (cm^2/g)^2 * dyne/cm^2 TODO

    # ----------- boundary conditions -----------
    R_Jupiter = 6.9911e9            # cm
    M_Jupiter = 1.898e30            # g
    P_surface_Jupiter = 1e6         # dyne/cm^2
    T_surface_Jupiter = 165         # K

    simulate_ideal_gas(
        R_Jupiter, 
        M_Jupiter,
        P_surface_Jupiter,
        N=N,
        theta=theta,
        output_name='01_ideal_gas_Jupiter'
    )
    
    simulate_polytropic(
        R_Jupiter, 
        M_Jupiter,
        P_surface_Jupiter,
        N=N,
        theta=theta,
        output_name='02_polytropic_Jupiter'
    )

    simulate_analytical(
        R_Jupiter, 
        M_Jupiter,
        P_surface_Jupiter,
        element='Fe',
        N=N,
        theta=theta,
        output_name='03_analytical_Fe_Jupiter'
    )

    simulate_analytical(
        R_Jupiter, 
        M_Jupiter,
        P_surface_Jupiter,
        element='MgSiO3',
        N=N,
        theta=theta,
        output_name='04_analytical_MgSiO3_Jupiter'
    )

    simulate_tabulated(
        R_Jupiter, 
        M_Jupiter,
        P_surface_Jupiter,
        T_surface_Jupiter,
        element='H',
        filename='data/EoS_H/TABLEEOS_2021_TP_Y0275_v1.csv',
        N=N,
        theta=theta,
        output_name='05_tabulated_H_Jupiter'
    )

    simulate_tabulated(
        R_Jupiter, 
        M_Jupiter,
        P_surface_Jupiter,
        T_surface_Jupiter,
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
