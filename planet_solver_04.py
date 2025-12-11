import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

# ================= reduce spam =================
class WarningTracker:
    def __init__(self):
        self.warned = {}
        self.suppress_spam = True
    
    def should_warn(self, key):
        if self.suppress_spam:
            if key not in self.warned:
                self.warned[key] = True
                return True
            return False
        else:
            return True
    
    def reset(self):
        self.warned = {}

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

def simulate_ideal_gas_Euler(R_surf, M_surf, P_surf, N, theta, output_name, show_plot=False, save_plot=True):
    '''
    Prameters:
        R_surf: [cm] radius of planet
        M_surf: [g] total mass of planet
        P_surf: [dyne/cm^2] surface pressure of planet
        N: number of steps
        theta: strech factor -> higher = r_grid is more dense close to the surface
        output_name: name for data and plot file
    '''

    # ------------ Inital conditions ------------
    r_grid, data = create_grids(R_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- Run simulation --------------
    print('start ideal gas (Euler, r-grid)')
    for i in range(N):
        r1 = r_grid[i]
        r2 = r_grid[i+1]

        m1 = data[i,1] 
        P1 = data[i,2]
        
        # ------------------- EoS -------------------
        rho = EoS_ideal_gas(P1, C_ideal_gas)
        # -------------------------------------------

        res = solve_Euler_with_events(r1, r2, m1, P1, rho, data, i)
        if res == False:
            break        

    # -------------- Clean up data --------------
    data =  data[:-(N-i)] # the last lines contain no useful/realistic data and can be removed

    
    data[:,1] = 4/3 * np.pi * data[:,0]**3 * data[:,3] # mass fix


    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)

    # ----------------- Save data ---------------
    save_data(data, N, output_name)
    if save_plot:
        plot_data(data, N, output_name, show_plot=show_plot)

def simulate_ideal_gas_ivp(R_surf, M_surf, P_surf, method, N, theta, output_name, show_plot=False, save_plot=True):
    '''    
    Parameters:
        R_surf: [cm] radius of planet
        M_surf: [g] total mass of planet
        P_surf: [dyne/cm^2] surface pressure of planet
        method: ode solver 'RK45', 'DOP853', 'Radau'
        N: number of steps
        theta: stretch factor -> higher = r_grid is more dense close to the surface
        output_name: name for data and plot file
    '''
    
    # ------------ Initial conditions ------------
    r_grid, data = create_grids(R_surf, N, theta)
    
    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf
    
    # ------------- Run simulation --------------
    print(f'start ideal gas ({method}, r-grid)')
    
    # Define the system of ODEs
    def system_of_ODEs(r, y):
        m, P = y
        
        # Prevent numerical issues near center
        if r < 1e-10:
            r = 1e-10
        
        # ----------------- EoS -----------------
        rho = EoS_ideal_gas(P, C_ideal_gas)
        # ---------------------------------------
        
        # ODEs
        dm_dr = 4 * np.pi * r**2 * rho
        dP_dr = -G * m * rho / r**2
        
        return [dm_dr, dP_dr]
    
    # Solve the system using solve_ivp
    data = solve_ivp_with_events(
        system_of_ODEs,
        r_grid,
        [M_surf, P_surf],
        method,
        N,
        data
    )
    
    # ----------------- Density -----------------
    for i, P in enumerate(data[:,2]):

        # ----------------- EoS -----------------
        rho = EoS_ideal_gas(P, C_ideal_gas)
        # ---------------------------------------

        data[i,3] = rho
    
    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)
    
    # ----------------- Save data ---------------
    save_data(data, N, output_name)
    if save_plot:
        plot_data(data, N, output_name, sim_method=method, show_plot=show_plot)

def simulate_ideal_gas_Euler_m_grid(R_surf, M_surf, P_surf, N, theta, output_name, show_plot=False, save_plot=True):
    '''
    Prameters:
        R_surf: [cm] radius of planet
        M_surf: [g] total mass of planet
        P_surf: [dyne/cm^2] surface pressure of planet
        N: number of steps
        theta: strech factor -> higher = m_grid is more dense close to the surface
        output_name: name for data and plot file
    '''

    # ------------ Inital conditions ------------
    m_grid, data = create_grids(M_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- Run simulation --------------
    print('start ideal gas (Euler, m-grid)')
    for i in range(N):
        m1 = m_grid[i]
        m2 = m_grid[i+1]

        r1 = data[i,0] 
        P1 = data[i,2]
        
        # ------------------- EoS -------------------
        rho = EoS_ideal_gas(P1, C_ideal_gas)
        # -------------------------------------------

        res = solve_Euler_with_events_m_grid(m1, m2, r1, P1, rho, data, i)
        if res == False:
            break        

    # -------------- Clean up data --------------
    data =  data[:-(N-i)] # the last lines contain no useful/realistic data and can be removed

    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)

    # ----------------- Save data ---------------
    save_data(data, N, output_name, grid_type='m')
    if save_plot:
        plot_data(data, N, output_name, grid_type='m', show_plot=show_plot)

def simulate_ideal_gas_ivp_m_grid(R_surf, M_surf, P_surf, method, N, theta, output_name, show_plot=False, save_plot=True):
    '''    
    Parameters:
        R_surf: [cm] radius of planet
        M_surf: [g] total mass of planet
        P_surf: [dyne/cm^2] surface pressure of planet
        method: ode solver 'RK45', 'DOP853', 'Radau'
        N: number of steps
        theta: stretch factor -> higher = m_grid is more dense close to the surface
        output_name: name for data and plot file
    '''
    
    # ------------ Initial conditions ------------
    m_grid, data = create_grids(M_surf, N, theta)
    
    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf
    
    # ------------- Run simulation --------------
    print(f'start ideal gas ({method}, m-grid)')
    
    # Define the system of ODEs
    def system_of_ODEs(m, y):
        r, P = y
        
        # Prevent numerical issues near center
        if m < 1e-10:
            m = 1e-10
        
        # ----------------- EoS -----------------
        rho = EoS_ideal_gas(P, C_ideal_gas)
        # ---------------------------------------
        
        # ODEs
        dr_dm = 1.0 / (4 * np.pi * r**2 * rho)
        dP_dm = -G * m / (4 * np.pi * r**4)
        
        return [dr_dm, dP_dm]
    
    # Solve the system using solve_ivp
    data = solve_ivp_with_events_m_grid(
        system_of_ODEs,
        m_grid,
        [R_surf, P_surf],
        method,
        N,
        data
    )
    
    # ----------------- Density -----------------
    for i, P in enumerate(data[:,2]):

        # ----------------- EoS -----------------
        rho = EoS_ideal_gas(P, C_ideal_gas)
        # ---------------------------------------

        data[i,3] = rho
    
    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)
    
    # ----------------- Save data ---------------
    save_data(data, N, output_name, grid_type='m')
    if save_plot:
        plot_data(data, N, output_name, grid_type='m', sim_method=method, show_plot=show_plot)

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

def simulate_polytropic_Euler(R_surf, M_surf, P_surf, N, theta, output_name, show_plot=False, save_plot=True):
    '''
        Prameters:
            R_surf: [cm] radius of planet
            M_surf: [g] total mass of planet
            P_surf: [dyne/cm^2] surface pressure of planet
            N: number of steps
            theta: strech factor -> higher = r_grid is more dense close to the surface
            output_name: name for data and plot file
    '''

    # ------------ Inital conditions ------------
    r_grid, data = create_grids(R_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- Run simulation --------------
    print('start polytropic (Euler, r-grid)')
    for i in range(N):
        r1 = r_grid[i]
        r2 = r_grid[i+1]
        
        m1 = data[i,1] 
        P1 = data[i,2]
        
        # ------------------- EoS -------------------
        rho = EoS_polytropic(P1)
        # -------------------------------------------
        
        res = solve_Euler_with_events(r1, r2, m1, P1, rho, data, i)
        if res == False:
            break

    # -------------- Clean up data --------------
    data =  data[:-(N-i)] # the last lines contain no useful/realistic data and can be removed

    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)

    # ----------------- Save data ---------------
    save_data(data, N, output_name)
    if save_plot:
        plot_data(data, N, output_name, show_plot=show_plot)

def simulate_polytropic_ivp(R_surf, M_surf, P_surf, method, N, theta, output_name, show_plot=False, save_plot=True):
    '''    
    Parameters:
        R_surf: [cm] radius of planet
        M_surf: [g] total mass of planet
        P_surf: [dyne/cm^2] surface pressure of planet
        method: ode solver 'RK45', 'DOP853', 'Radau'
        N: number of steps
        theta: stretch factor -> higher = r_grid is more dense close to the surface
        output_name: name for data and plot file
    '''
    
    # ------------ Initial conditions ------------
    r_grid, data = create_grids(R_surf, N, theta)
    
    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- Run simulation --------------
    print(f'start polytropic ({method}, r-grid)')
    
    # Define the system of ODEs
    def system_of_ODEs(r, y):
        m, P = y
        
        # Prevent numerical issues near center
        if r < 1e-10:
            r = 1e-10
        
        # ----------------- EoS -----------------
        rho = EoS_polytropic(P)
        # ---------------------------------------
        
        # ODEs
        dm_dr = 4 * np.pi * r**2 * rho
        dP_dr = -G * m * rho / r**2
        
        # print(dm_dr)
        # m2 = 4/3 * np.pi * r**3 * rho # mass fix


        return [dm_dr, dP_dr]
    
    # Solve the system using solve_ivp
    data = solve_ivp_with_events(
        system_of_ODEs,
        r_grid,
        [M_surf, P_surf],
        method,
        N,
        data
    )
    
    # ----------------- Density -----------------
    for i, P in enumerate(data[:,2]):
        
        # ----------------- EoS -----------------
        rho = EoS_polytropic(P)
        # ---------------------------------------

        data[i,3] = rho
    
    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)
    
    # ----------------- Save data ---------------
    save_data(data, N, output_name)
    if save_plot:
        plot_data(data, N, output_name, sim_method=method, show_plot=show_plot)

def simulate_polytropic_Euler_m_grid(R_surf, M_surf, P_surf, N, theta, output_name, show_plot=False, save_plot=True):
    '''
        Prameters:
            R_surf: [cm] radius of planet
            M_surf: [g] total mass of planet
            P_surf: [dyne/cm^2] surface pressure of planet
            N: number of steps
            theta: strech factor -> higher = m_grid is more dense close to the surface
            output_name: name for data and plot file
    '''

    # ------------ Inital conditions ------------
    m_grid, data = create_grids(M_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- Run simulation --------------
    print('start polytropic (Euler, m-grid)')
    for i in range(N):
        m1 = m_grid[i]
        m2 = m_grid[i+1]
        
        r1 = data[i,0] 
        P1 = data[i,2]
        
        # ------------------- EoS -------------------
        rho = EoS_polytropic(P1)
        # -------------------------------------------
        
        res = solve_Euler_with_events_m_grid(m1, m2, r1, P1, rho, data, i)
        if res == False:
            break

    # -------------- Clean up data --------------
    data =  data[:-(N-i)] # the last lines contain no useful/realistic data and can be removed

    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)

    # ----------------- Save data ---------------
    save_data(data, N, output_name, grid_type='m')
    if save_plot:
        plot_data(data, N, output_name, grid_type='m', show_plot=show_plot)

def simulate_polytropic_ivp_m_grid(R_surf, M_surf, P_surf, method, N, theta, output_name, show_plot=False, save_plot=True):
    '''    
    Parameters:
        R_surf: [cm] radius of planet
        M_surf: [g] total mass of planet
        P_surf: [dyne/cm^2] surface pressure of planet
        method: ode solver 'RK45', 'DOP853', 'Radau'
        N: number of steps
        theta: stretch factor -> higher = m_grid is more dense close to the surface
        output_name: name for data and plot file
    '''
    
    # ------------ Initial conditions ------------
    m_grid, data = create_grids(M_surf, N, theta)
    
    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- Run simulation --------------
    print(f'start polytropic ({method}, m-grid)')
    
    # Define the system of ODEs
    def system_of_ODEs(m, y):
        r, P = y
        
        # Prevent numerical issues near center
        if m < 1e-10:
            m = 1e-10
        
        # ----------------- EoS -----------------
        rho = EoS_polytropic(P)
        # ---------------------------------------
        
        # ODEs
        dr_dm = 1.0 / (4 * np.pi * r**2 * rho)
        dP_dm = -G * m / (4 * np.pi * r**4)
        
        return [dr_dm, dP_dm]
    
    # Solve the system using solve_ivp
    data = solve_ivp_with_events_m_grid(
        system_of_ODEs,
        m_grid,
        [R_surf, P_surf],
        method,
        N,
        data
    )
    
    # ----------------- Density -----------------
    for i, P in enumerate(data[:,2]):
        
        # ----------------- EoS -----------------
        rho = EoS_polytropic(P)
        # ---------------------------------------

        data[i,3] = rho
    
    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)
    
    # ----------------- Save data ---------------
    save_data(data, N, output_name, grid_type='m')
    if save_plot:
        plot_data(data, N, output_name, grid_type='m', sim_method=method, show_plot=show_plot)

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

def simulate_analytical_Euler(R_surf, M_surf, P_surf, element, N, theta, output_name, show_plot=False, save_plot=True):
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

    # ------------- Look up tables --------------
    if element == 'Fe':
        df = pd.read_csv('data/EoS_Fe/EoS_Fe.csv', names=['p', 'rho'], skiprows=1)
    elif element == 'MgSiO3':
        df = pd.read_csv('data/EoS_MgSiO3/EoS_MgSiO3.csv', names=['p', 'rho'], skiprows=1)
    else:
        print(f'invalid element: {element}')
        return

    # ------------ Inital conditions ------------
    r_grid, data = create_grids(R_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- Run simulation --------------
    print(f'start analytical {element} (Euler, r-grid)')
    for i in range(N):
        r1 = r_grid[i]
        r2 = r_grid[i+1]
        
        m1 = data[i,1] 
        P1 = data[i,2]
        
        # ------------------- EoS -------------------
        if element == 'Fe':
            rho = EoS_analytical_Fe(P1, df)
        elif element == 'MgSiO3':
            rho = EoS_analytical_MgSiO3(P1, df)
        # -------------------------------------------
        
        res = solve_Euler_with_events(r1, r2, m1, P1, rho, data, i)
        if res == False:
            break

    # -------------- Clean up data --------------
    data =  data[:-(N-i)] # the last lines contain no useful/realistic data and can be removed

    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)

    # ----------------- Save data ---------------
    save_data(data, N, output_name)
    if save_plot:
        plot_data(data, N, output_name, show_plot=show_plot)

def simulate_analytical_ivp(R_surf, M_surf, P_surf, method, element, N, theta, output_name, show_plot=False, save_plot=True):
    '''    
    Parameters:
        R_surf: [cm] radius of planet
        M_surf: [g] total mass of planet
        P_surf: [dyne/cm^2] surface pressure of planet
        method: ode solver 'RK45', 'DOP853', 'Radau'
        element: 'Fe', or 'MgSiO3'
        N: number of steps
        theta: stretch factor -> higher = r_grid is more dense close to the surface
        output_name: name for data and plot file
    '''
    # ------------- Look up tables --------------
    if element == 'Fe':
        df = pd.read_csv('data/EoS_Fe/EoS_Fe.csv', names=['p', 'rho'], skiprows=1)
    elif element == 'MgSiO3':
        df = pd.read_csv('data/EoS_MgSiO3/EoS_MgSiO3.csv', names=['p', 'rho'], skiprows=1)
    else:
        print(f'invalid element: {element}')
        return
    
    # ------------ Initial conditions ------------
    r_grid, data = create_grids(R_surf, N, theta)
    
    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- Run simulation --------------
    print(f'start analytical {element} ({method}, r-grid)')
    
    # Define the system of ODEs
    def system_of_ODEs(r, y):
        m, P = y
        
        # Prevent numerical issues near center
        if r < 1e-10:
            r = 1e-10
        
        # ----------------- EoS -----------------
        if element == 'Fe':
            rho = EoS_analytical_Fe(P, df)
        elif element == 'MgSiO3':
            rho = EoS_analytical_MgSiO3(P, df)
        # ---------------------------------------
        
        # ODEs
        dm_dr = 4 * np.pi * r**2 * rho
        dP_dr = -G * m * rho / r**2
        
        return [dm_dr, dP_dr]
    
    # Solve the system using solve_ivp
    data = solve_ivp_with_events(
        system_of_ODEs,
        r_grid,
        [M_surf, P_surf],
        method,
        N,
        data
    )
    
    # ----------------- Density -----------------
    for i, P in enumerate(data[:,2]):

        # ----------------- EoS -----------------
        if element == 'Fe':
            rho = EoS_analytical_Fe(P, df)
        elif element == 'MgSiO3':
            rho = EoS_analytical_MgSiO3(P, df)
        # ---------------------------------------

        data[i,3] = rho
    
    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)
    
    # ----------------- Save data ---------------
    save_data(data, N, output_name)
    if save_plot:
        plot_data(data, N, output_name, show_plot=show_plot)

def simulate_analytical_Euler_m_grid(R_surf, M_surf, P_surf, element, N, theta, output_name, show_plot=False, save_plot=True):
    '''
        Prameters:
            R_surf: [cm] radius of planet
            M_surf: [g] total mass of planet
            P_surf: [dyne/cm^2] surface pressure of planet
            element: 'Fe', or 'MgSiO3'
            N: number of steps
            theta: strech factor -> higher = m_grid is more dense close to the surface
            output_name: name for data and plot file
    '''
    # ------------- Look up tables --------------
    if element == 'Fe':
        df = pd.read_csv('data/EoS_Fe/EoS_Fe.csv', names=['p', 'rho'], skiprows=1)
    elif element == 'MgSiO3':
        df = pd.read_csv('data/EoS_MgSiO3/EoS_MgSiO3.csv', names=['p', 'rho'], skiprows=1)
    else:
        print(f'invalid element: {element}')
        return
    # ------------ Inital conditions ------------
    m_grid, data = create_grids(M_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- Run simulation --------------
    print(f'start analytical {element} (Euler, m-grid)')
    for i in range(N):
        m1 = m_grid[i]
        m2 = m_grid[i+1]
        
        r1 = data[i,0] 
        P1 = data[i,2]
        
        # ------------------- EoS -------------------
        if element == 'Fe':
            rho = EoS_analytical_Fe(P1, df)
        elif element == 'MgSiO3':
            rho = EoS_analytical_MgSiO3(P1, df)
        # -------------------------------------------
        
        res = solve_Euler_with_events_m_grid(m1, m2, r1, P1, rho, data, i)
        if res == False:
            break

    # -------------- Clean up data --------------
    data =  data[:-(N-i)] # the last lines contain no useful/realistic data and can be removed

    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)

    # ----------------- Save data ---------------
    save_data(data, N, output_name, grid_type='m')
    if save_plot:
        plot_data(data, N, output_name, grid_type='m', show_plot=show_plot)

def simulate_analytical_ivp_m_grid(R_surf, M_surf, P_surf, method, element, N, theta, output_name, show_plot=False, save_plot=True):
    '''    
    Parameters:
        R_surf: [cm] radius of planet
        M_surf: [g] total mass of planet
        P_surf: [dyne/cm^2] surface pressure of planet
        method: ode solver 'RK45', 'DOP853', 'Radau'
        element: 'Fe', or 'MgSiO3'
        N: number of steps
        theta: stretch factor -> higher = m_grid is more dense close to the surface
        output_name: name for data and plot file
    '''
    # ------------- Look up tables --------------
    if element == 'Fe':
        df = pd.read_csv('data/EoS_Fe/EoS_Fe.csv', names=['p', 'rho'], skiprows=1)
    elif element == 'MgSiO3':
        df = pd.read_csv('data/EoS_MgSiO3/EoS_MgSiO3.csv', names=['p', 'rho'], skiprows=1)
    else:
        print(f'invalid element: {element}')
        return

    # ------------ Initial conditions ------------
    m_grid, data = create_grids(M_surf, N, theta)
    
    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf

    # ------------- Run simulation --------------
    print(f'start analytical ({method}, m-grid)')
    
    # Define the system of ODEs
    def system_of_ODEs(m, y):
        r, P = y
        
        # Prevent numerical issues near center
        if m < 1e-10:
            m = 1e-10
        
        # ----------------- EoS -----------------
        if element == 'Fe':
            rho = EoS_analytical_Fe(P, df)
        elif element == 'MgSiO3':
            rho = EoS_analytical_MgSiO3(P, df)
        # ---------------------------------------
        
        # ODEs
        dr_dm = 1.0 / (4 * np.pi * r**2 * rho)
        dP_dm = -G * m / (4 * np.pi * r**4)
        
        return [dr_dm, dP_dm]
    
    # Solve the system using solve_ivp
    data = solve_ivp_with_events_m_grid(
        system_of_ODEs,
        m_grid,
        [R_surf, P_surf],
        method,
        N,
        data
    )
    
    # ----------------- Density -----------------
    for i, P in enumerate(data[:,2]):
        
        # ----------------- EoS -----------------
        if element == 'Fe':
            rho = EoS_analytical_Fe(P, df)
        elif element == 'MgSiO3':
            rho = EoS_analytical_MgSiO3(P, df)
        # ---------------------------------------

        data[i,3] = rho
    
    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)
    
    # ----------------- Save data ---------------
    save_data(data, N, output_name, grid_type='m')
    if save_plot:
        plot_data(data, N, output_name, grid_type='m', sim_method=method, show_plot=show_plot)

# ================== tabulated ==================
def parse_EoS_H(filename, columns):
    '''
        Parameters:
            filename: path to data 
            colums: colum names
    
        Returns:
            T_vals: needed for interpolator
            P_vals: needed for interpolator
            rho_3D: 3D matrix [T, P, rho]
            grad_ad_3D: 3D matrix [T, P, grad_ad]
    '''
    file_name = os.path.basename(filename)
    if 'TABLEEOS_2021' in file_name:
        file_name = file_name.replace('TABLEEOS_2021_', '').replace('_v1.csv', '')
        
        # Use .npy for fastest I/O
        base_dir = os.path.dirname(filename)
        rho_matrix_path = os.path.join(base_dir, f'rho_matrix_{file_name}.npy')
        grad_ad_matrix_path = os.path.join(base_dir, f'grad_ad_matrix_{file_name}.npy')
        meta_path = os.path.join(base_dir, f'meta_{file_name}.npz')

        if os.path.isfile(rho_matrix_path) and os.path.isfile(grad_ad_matrix_path) and os.path.isfile(meta_path):
            # READ FILES (very fast with .npy)
            rho_3D = np.load(rho_matrix_path)
            grad_ad_3D = np.load(grad_ad_matrix_path)
            meta = np.load(meta_path)
            T_vals = meta['T_vals'].tolist()
            P_vals = meta['P_vals'].tolist()
            
        else:  # write files
            # Read all data into one DataFrame
            all_data = []
            current_temp = None
            
            with open(filename, 'r') as f:
                for line in f:
                    if line.strip().startswith('#iT='):
                        current_temp = float(line.split(' T= ')[1])
                    elif line.strip().startswith('#'):
                        continue
                    elif line.strip() and current_temp is not None:
                        values = [float(x) for x in line.split()]
                        values.insert(0, current_temp)  # Add temperature as first column
                        all_data.append(values)
            
            # Create DataFrame with temperature column
            df = pd.DataFrame(all_data, columns=['temp'] + columns)
            
            # Use pivot_table for fast matrix creation
            T_vals = sorted(df['temp'].unique())
            P_vals = sorted(df['log_P'].unique())
            
            rho_pivot = df.pivot_table(values='log_rho', index='temp', columns='log_P', aggfunc='first')
            grad_ad_pivot = df.pivot_table(values='grad_ad', index='temp', columns='log_P', aggfunc='first')
            
            # Convert to numpy arrays with correct ordering
            rho_3D = rho_pivot.reindex(index=T_vals, columns=P_vals).values
            grad_ad_3D = grad_ad_pivot.reindex(index=T_vals, columns=P_vals).values
            
            # WRITE FILES (very fast with .npy)
            np.save(rho_matrix_path, rho_3D)
            np.save(grad_ad_matrix_path, grad_ad_3D)
            np.savez(meta_path, T_vals=np.array(T_vals), P_vals=np.array(P_vals))
    else:
        print(f'invalid file: {file_name}. Use one of the TABLEEOS_2021... files.')
        return

    return T_vals, P_vals, rho_3D, grad_ad_3D

def parse_EoS_H2O(filename, columns):
    '''
        Parameters:
            filename: path to data 
            colums: colum names
    
        Returns:
            T_vals: needed for interpolator
            P_vals: needed for interpolator
            rho_3D: 3D matrix [T, P, rho]
            grad_ad_3D: 3D matrix [T, P, grad_ad]
    '''
    
    # Setup cache paths
    file_name = os.path.basename(filename)
    file_name = file_name.replace('aqua_eos_', '').replace('_v1_0.dat', '')
    base_dir = os.path.dirname(filename)
    rho_matrix_path = os.path.join(base_dir, f'rho_matrix_{file_name}.npy')
    grad_ad_matrix_path = os.path.join(base_dir, f'grad_ad_matrix_{file_name}.npy')
    meta_path = os.path.join(base_dir, f'meta_{file_name}.npz')
    
    # Check if cached files exist
    if os.path.isfile(rho_matrix_path) and os.path.isfile(grad_ad_matrix_path) and os.path.isfile(meta_path):
        # READ FROM CACHE
        rho_3D = np.load(rho_matrix_path)
        grad_ad_3D = np.load(grad_ad_matrix_path)
        meta = np.load(meta_path)
        T_vals = meta['T_vals'].tolist()
        P_vals = meta['P_vals'].tolist()
        
    else:
        # PARSE AND CREATE CACHE
        df = pd.read_csv(filename, skiprows=21, sep='\\s+', names=columns)
        
        # Sort once and use pivot for efficiency
        df = df.sort_values(['temp', 'press'])
        T_vals = sorted(df['temp'].unique())
        P_vals = sorted(df['press'].unique())
        
        # Use pivot_table for fast matrix creation (much faster than loops)
        rho_pivot = df.pivot_table(values='rho', index='temp', columns='press', aggfunc='first')
        grad_ad_pivot = df.pivot_table(values='ad_grad', index='temp', columns='press', aggfunc='first')
        
        # Ensure correct ordering and convert to numpy arrays
        rho_3D = rho_pivot.reindex(index=T_vals, columns=P_vals).values
        grad_ad_3D = grad_ad_pivot.reindex(index=T_vals, columns=P_vals).values
        
        # SAVE TO CACHE
        np.save(rho_matrix_path, rho_3D)
        np.save(grad_ad_matrix_path, grad_ad_3D)
        np.savez(meta_path, T_vals=np.array(T_vals), P_vals=np.array(P_vals))
    
    return T_vals, P_vals, rho_3D, grad_ad_3D

def EoS_tabulated_H(P, T, interpolator):
    '''
        Parameters:
            P: [dyne/cm^2] pressure
            T: [K] temperature
            interpolator: interpolator [T, P, rho]

        Returns:
            rho: [g/cm^3]
    '''
    global warning_tracker

    P_query = P * 1e-10 # dyne/cm^2 -> GPa
    P_query = np.log10(P_query)
    T_query = np.log10(T) # K

    T_vals = interpolator.grid[0]
    P_vals = interpolator.grid[1]

    if T_query < min(T_vals) or  max(T_vals) < T_query:
        if warning_tracker.should_warn('H_T_bounds'):
            print(f'T: e^{T_query:.3f} is out of bounds -> cliped to range [e^{min(T_vals)}, e^{max(T_vals)}]')
        T_query = np.clip(T_query, min(T_vals), max(T_vals))

    if P_query < min(P_vals) or max(P_vals) < P_query:
        if warning_tracker.should_warn('H_P_bounds'):
            print(f'P: e^{P_query:.3f} is out of bounds -> cliped to range [e^{min(P_vals)}, e^{max(P_vals)}]')
        P_query = np.clip(P_query, min(P_vals), max(P_vals))

    # interpolate rho
    log_rho = interpolator([T_query, P_query])[0]

    rho = np.exp(log_rho) # g/cm^3
    
    return rho

def EoS_tabulated_H2O(P, T, interpolator):
    '''
        Parameters:
            P: [dyne/cm^2] pressure
            T: [K] temperature
            interpolator: interpolator [T, P, rho]

        Returns:
            rho: [g/cm^3]
    '''
    global warning_tracker

    P_query = P * 0.1 # dyne/cm^2 -> Pa
    T_query = T # K
    
    T_vals = interpolator.grid[0]
    P_vals = interpolator.grid[1]

    if T_query < min(T_vals) or  max(T_vals) < T_query:
        if warning_tracker.should_warn('H2O_T_bounds'):
            print(f'T: {T_query:.3f} is out of bounds -> cliped to range [{min(T_vals)}, {max(T_vals)}]')
        T_query = np.clip(T_query, min(T_vals), max(T_vals))

    if P_query < min(P_vals) or max(P_vals) < P_query:
        if warning_tracker.should_warn('H2O_P_bounds'):
            print(f'P: {P_query:.3f} is out of bounds -> cliped to range [{min(P_vals)}, {max(P_vals)}]')
        P_query = np.clip(P_query, min(P_vals), max(P_vals))

    # interpolate rho
    rho = interpolator([T_query, P_query])[0]

    rho *= 1e-3 # kg/m^3 -> g/cm^3
    
    return rho

def simulate_tabulated_Euler(R_surf, M_surf, P_surf, T_surf, element, filename, N, theta, output_name, show_plot=False, save_plot=True):
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
    global warning_tracker
    warning_tracker.reset()

    # ------------- Look up tables --------------
    if element == 'H':
        columns = [
            'log_T', 'log_P', 'log_rho', 'log_U', 'log_s', 
            'dlrho_dlT_P', 'dlrho_dlP_T', 'dlS_dlT_P', 'dlS_dlP_T', 'grad_ad'
        ]

        T_vals, P_vals, rho_3D, grad_ad_3D = parse_EoS_H(filename, columns)
        interpolator_rho = RegularGridInterpolator((T_vals, P_vals), rho_3D)
        interpolator_grad_ad = RegularGridInterpolator((T_vals, P_vals), grad_ad_3D)
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

        T_vals, P_vals, rho_3D, grad_ad_3D = parse_EoS_H2O(filename, columns)
        interpolator_rho = RegularGridInterpolator((T_vals, P_vals), rho_3D)
        interpolator_grad_ad = RegularGridInterpolator((T_vals, P_vals), grad_ad_3D)
    else:
        print(f'invalid element: {element}')
        return

    # ------------ Inital conditions ------------
    r_grid, data = create_grids(R_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf
    data[0,5] = T_surf

    # ------------- Run simulation --------------
    print(f'start tabulated {element} (Euler, r-grid)')
    for i in range(N):
        r1 = r_grid[i]
        r2 = r_grid[i+1]
        
        m1 = data[i,1] 
        P1 = data[i,2]
        T1 = data[i,5]
        
        # ------------------- EoS -------------------
        if element == 'H':
            rho = EoS_tabulated_H(P1, T1, interpolator_rho)
        elif element == 'H2O':
            rho = EoS_tabulated_H2O(P1, T1, interpolator_rho)
        # -------------------------------------------
        
        data[i,3] = rho
        
        # ---------- Calculate derivatives ----------
        if r2 < 1e-10: # handle div by 0 error
            r2 = 1e-10 

        dm_dr = 4 * np.pi * r1**2 * rho
        dP_dr = -G * m1 * rho / r1**2
        
        # ------ Update values (Euler method) -------
        dr = r2 - r1
        m2 = m1 + dm_dr * dr
        P2 = P1 + dP_dr * dr

        # ---------- Find next temperature ----------
        if element == 'H':
            P_query = P1 * 1e-10 # dyne/cm^2 -> GPa
            P_query = np.log10(P_query)
            T_query = np.log10(T1) # K

            if T_query < min(T_vals) or  max(T_vals) < T_query:
                T_query = np.clip(T_query, min(T_vals), max(T_vals))

            if P_query < min(P_vals) or max(P_vals) < P_query:
                P_query = np.clip(P_query, min(P_vals), max(P_vals))

            grad_ad = interpolator_grad_ad([T_query, P_query])[0]

        elif element == 'H2O':
            P_query = P1 * 0.1 # dyne/cm^2 -> Pa
            T_query = T1 # K

            if T_query < min(T_vals) or  max(T_vals) < T_query:
                T_query = np.clip(T_query, min(T_vals), max(T_vals))

            if P_query < min(P_vals) or max(P_vals) < P_query:
                P_query = np.clip(P_query, min(P_vals), max(P_vals))

            grad_ad = interpolator_grad_ad([T_query, P_query])[0]

        T2 = T1/P1 * (P2 - P1) * grad_ad + T1

        # -------- Check if data makes sense --------
        if P2 < 0.0 or T2 < 0.0:
            print('>'*25, 'ALERT', '<'*25)
            if P2 < 0.0: 
                print(f'negative pressure at {i+1}/{N}')
            else: 
                print(f'negative temperature at {i+1}/{N}')
            break
            
        if m2 < 0.0:
            print(f' aborted at {i+1}/{N}')
            break # abort sim

        # --------- Save data for next step ---------
        if i < len(data)-1:
            data[i+1,0] = r2
            data[i+1,1] = m2
            data[i+1,2] = P2
            data[i+1,5] = T2

    # -------------- Clean up data --------------
    data =  data[:-(N-i)] # the last lines contain no useful/realistic data and can be removed

    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)

    # ----------------- Save data ---------------
    save_data(data, N, output_name)
    if save_plot:
        plot_data(data, N, output_name, show_plot=show_plot)

def simulate_tabulated_Euler_m_grid(R_surf, M_surf, P_surf, T_surf, element, filename, N, theta, output_name, show_plot=False, save_plot=True):
    '''
    Prameters:
        R_surf: [cm] radius of planet
        M_surf: [g] total mass of planet
        P_surf: [dyne/cm^2] surface pressure of planet
        T_surf: [K] surface temperature of planet
        element: 'H', or 'H2O'
        filename: path to tabulated data
        N: number of steps
        theta: strech factor -> higher = m_grid is more dense close to the surface
        output_name: name for data and plot file
    '''
    global warning_tracker
    warning_tracker.reset()

    # ------------- Look up tables --------------
    if element == 'H':
        columns = [
            'log_T', 'log_P', 'log_rho', 'log_U', 'log_s', 
            'dlrho_dlT_P', 'dlrho_dlP_T', 'dlS_dlT_P', 'dlS_dlP_T', 'grad_ad'
        ]

        T_vals, P_vals, rho_3D, grad_ad_3D = parse_EoS_H(filename, columns)
        interpolator_rho = RegularGridInterpolator((T_vals, P_vals), rho_3D)
        interpolator_grad_ad = RegularGridInterpolator((T_vals, P_vals), grad_ad_3D)
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

        T_vals, P_vals, rho_3D, grad_ad_3D = parse_EoS_H2O(filename, columns)
        interpolator_rho = RegularGridInterpolator((T_vals, P_vals), rho_3D)
        interpolator_grad_ad = RegularGridInterpolator((T_vals, P_vals), grad_ad_3D)
    else:
        print(f'invalid element: {element}')
        return

    # ------------ Inital conditions ------------
    m_grid, data = create_grids(M_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf
    data[0,5] = T_surf

    # ------------- Run simulation --------------
    print(f'start tabulated {element} (Euler, m-grid)')
    for i in range(N):
        m1 = m_grid[i]
        m2 = m_grid[i+1]
        
        r1 = data[i,0] 
        P1 = data[i,2]
        T1 = data[i,5]
        
        # ------------------- EoS -------------------
        if element == 'H':
            rho = EoS_tabulated_H(P1, T1, interpolator_rho)
        elif element == 'H2O':
            rho = EoS_tabulated_H2O(P1, T1, interpolator_rho)
        # -------------------------------------------
        
        data[i,3] = rho
        
        # ---------- Calculate derivatives ----------
        if m2 < 1e-10: # handle div by 0 error
            m2 = 1e-10 

        dr_dm = 1.0 / (4 * np.pi * r1**2 * rho)
        dP_dm = -G * m1 / (4 * np.pi * r1**4)
        
        # ------ Update values (Euler method) -------
        dm = m2 - m1
        r2 = r1 + dr_dm * dm
        P2 = P1 + dP_dm * dm

        # ---------- Find next temperature ----------
        if element == 'H':
            P_query = P1 * 1e-10 # dyne/cm^2 -> GPa
            P_query = np.log10(P_query)
            T_query = np.log10(T1) # K

            if T_query < min(T_vals) or  max(T_vals) < T_query:
                T_query = np.clip(T_query, min(T_vals), max(T_vals))

            if P_query < min(P_vals) or max(P_vals) < P_query:
                P_query = np.clip(P_query, min(P_vals), max(P_vals))

            grad_ad = interpolator_grad_ad([T_query, P_query])[0]

        elif element == 'H2O':
            P_query = P1 * 0.1 # dyne/cm^2 -> Pa
            T_query = T1 # K

            if T_query < min(T_vals) or  max(T_vals) < T_query:
                T_query = np.clip(T_query, min(T_vals), max(T_vals))

            if P_query < min(P_vals) or max(P_vals) < P_query:
                P_query = np.clip(P_query, min(P_vals), max(P_vals))

            grad_ad = interpolator_grad_ad([T_query, P_query])[0]

        T2 = T1/P1 * (P2 - P1) * grad_ad + T1

        # -------- Check if data makes sense --------
        if P2 < 0.0 or T2 < 0.0:
            print('>'*25, 'ALERT', '<'*25)
            if P2 < 0.0: 
                print(f'negative pressure at {i+1}/{N}')
            else: 
                print(f'negative temperature at {i+1}/{N}')
            break
            
        if r2 < 0.0:
            print(f' aborted at {i+1}/{N}')
            break # abort sim

        # --------- Save data for next step ---------
        if i < len(data)-1:
            data[i+1,0] = r2
            data[i+1,1] = m2
            data[i+1,2] = P2
            data[i+1,5] = T2

    # -------------- Clean up data --------------
    data =  data[:-(N-i)] # the last lines contain no useful/realistic data and can be removed

    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)

    # ----------------- Save data ---------------
    save_data(data, N, output_name, grid_type='m')
    if save_plot:
        plot_data(data, N, output_name, grid_type='m', show_plot=show_plot)

def simulate_tabulated_ivp(R_surf, M_surf, P_surf, T_surf, method, element, filename, N, theta, output_name, show_plot=False, save_plot=True):
    '''    
    Parameters:
        R_surf: [cm] radius of planet
        M_surf: [g] total mass of planet
        P_surf: [dyne/cm^2] surface pressure of planet
        T_surf: [K] surface temperature of planet
        method: ode solver 'RK45', 'DOP853', 'Radau'
        element: 'H', or 'H2O'
        filename: path to tabulated data
        N: number of steps
        theta: stretch factor -> higher = r_grid is more dense close to the surface
        output_name: name for data and plot file
    '''
    global warning_tracker
    warning_tracker.reset()

    # ------------- Look up tables --------------
    if element == 'H':
        columns = [
            'log_T', 'log_P', 'log_rho', 'log_U', 'log_s', 
            'dlrho_dlT_P', 'dlrho_dlP_T', 'dlS_dlT_P', 'dlS_dlP_T', 'grad_ad'
        ]

        T_vals, P_vals, rho_3D, grad_ad_3D = parse_EoS_H(filename, columns)
        interpolator_rho = RegularGridInterpolator((T_vals, P_vals), rho_3D)
        interpolator_grad_ad = RegularGridInterpolator((T_vals, P_vals), grad_ad_3D)
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

        T_vals, P_vals, rho_3D, grad_ad_3D = parse_EoS_H2O(filename, columns)
        interpolator_rho = RegularGridInterpolator((T_vals, P_vals), rho_3D)
        interpolator_grad_ad = RegularGridInterpolator((T_vals, P_vals), grad_ad_3D)
    else:
        print(f'invalid element: {element}')
        return

    # ------------ Inital conditions ------------
    r_grid, data = create_grids(R_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf
    data[0,5] = T_surf

    # ------------- Run simulation --------------
    print(f'start tabulated {element} ({method}, r-grid)')
    
    # Define the system of ODEs
    def system_of_ODEs(r, y):
        m, P, T = y
        
        # Prevent numerical issues near center
        if r < 1e-10:
            r = 1e-10
        
        # ----------------- EoS -----------------
        if element == 'H':
            rho = EoS_tabulated_H(P, T, interpolator_rho)
        elif element == 'H2O':
            rho = EoS_tabulated_H2O(P, T, interpolator_rho)
        # ---------------------------------------
        
        # ODEs
        dm_dr = 4 * np.pi * r**2 * rho
        dP_dr = -G * m * rho / r**2

        # -------------- Find dT_dr -------------
        if element == 'H':
            P_query = P * 1e-10 # dyne/cm^2 -> GPa
            P_query = np.log10(P_query)
            T_query = np.log10(T) # K

            if T_query < min(T_vals) or  max(T_vals) < T_query:
                T_query = np.clip(T_query, min(T_vals), max(T_vals))

            if P_query < min(P_vals) or max(P_vals) < P_query:
                P_query = np.clip(P_query, min(P_vals), max(P_vals))

            grad_ad = interpolator_grad_ad([T_query, P_query])[0]

        elif element == 'H2O':
            P_query = P * 0.1 # dyne/cm^2 -> Pa
            T_query = T # K

            if T_query < min(T_vals) or  max(T_vals) < T_query:
                T_query = np.clip(T_query, min(T_vals), max(T_vals))

            if P_query < min(P_vals) or max(P_vals) < P_query:
                P_query = np.clip(P_query, min(P_vals), max(P_vals))

            grad_ad = interpolator_grad_ad([T_query, P_query])[0]

        dT_dr = T/P * dP_dr * grad_ad # = T2 - T1

        return [dm_dr, dP_dr, dT_dr]
    
    # Define events to stop integration
    def negative_mass_event(r, y):
        return y[0]  # Stop when mass becomes negative
    negative_mass_event.terminal = True
    negative_mass_event.direction = -1

    def negative_pressure_event(r, y):
        return y[1]  # Stop when pressure becomes negative
    negative_pressure_event.terminal = True
    negative_pressure_event.direction = -1
    
    def negative_temperature_event(r, y):
        return y[2]  # Stop when temperature becomes negative
    negative_temperature_event.terminal = True
    negative_temperature_event.direction = -1
    
    # Solve from surface to center
    r_span = (r_grid[0], r_grid[-1])
    
    # Set tolerances based on method
    if method == 'DOP853':
        rtol = 1e-10
        atol = 1e-12
    else:
        rtol = 1e-8
        atol = 1e-10
    
    sol = solve_ivp(
        system_of_ODEs,
        r_span,
        [M_surf, P_surf, T_surf],
        method=method,
        t_eval=r_grid,
        events=[negative_mass_event, negative_pressure_event, negative_temperature_event],
        rtol=rtol,
        atol=atol,
        max_step=np.inf  # Let the solver choose step size
    )
    
    if sol.status == 1:  # Terminated by event
        if len(sol.t_events[0]) > 0:
            print(f' aborted at step {len(sol.t)}/{N}')
        elif len(sol.t_events[1]) > 0:
            print(f'>'*25, 'ALERT', '<'*25)
            print(f' negative pressure at step {len(sol.t)}/{N}')
        elif len(sol.t_events[2]) > 0:
            print(f'>'*25, 'ALERT', '<'*25)
            print(f' negative temperature at step {len(sol.t)}/{N}')
    elif sol.status == -1:
        print(f' Integration failed at step {len(sol.t)}/{N}: {sol.message}')
    
    if sol.status == 0 or sol.status == 1: # success or terminated early
        # Store the solution
        n_points = len(sol.t)

        # Trim data array to actual size
        data = data[:n_points]
        
        data[:,0] = sol.t     # radius
        data[:,1] = sol.y[0]  # mass
        data[:,2] = sol.y[1]  # pressure
        data[:,5] = sol.y[2]  # temperature

        # Trim data array to actual size
        if data[:,0][-1] == 0.0:
            data = data[:-1]

    # ----------------- Density -----------------
    for i, (P, T) in enumerate(zip(data[:,2], data[:,5])):

        # ----------------- EoS -----------------
        if element == 'H':
            rho = EoS_tabulated_H(P, T, interpolator_rho)
        elif element == 'H2O':
            rho = EoS_tabulated_H2O(P, T, interpolator_rho)
        # ---------------------------------------

        data[i,3] = rho
    
    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)
    
    # ----------------- Save data ---------------
    save_data(data, N, output_name)
    if save_plot:
        plot_data(data, N, output_name, show_plot=show_plot)

def simulate_tabulated_ivp_m_grid(R_surf, M_surf, P_surf, T_surf, method, element, filename, N, theta, output_name, show_plot=False, save_plot=True):
    '''    
    Parameters:
        R_surf: [cm] radius of planet
        M_surf: [g] total mass of planet
        P_surf: [dyne/cm^2] surface pressure of planet
        T_surf: [K] surface temperature of planet
        method: ode solver 'RK45', 'DOP853', 'Radau'
        element: 'H', or 'H2O'
        filename: path to tabulated data
        N: number of steps
        theta: stretch factor -> higher = m_grid is more dense close to the surface
        output_name: name for data and plot file
    '''
    global warning_tracker
    warning_tracker.reset()

    # ------------- Look up tables --------------
    if element == 'H':
        columns = [
            'log_T', 'log_P', 'log_rho', 'log_U', 'log_s', 
            'dlrho_dlT_P', 'dlrho_dlP_T', 'dlS_dlT_P', 'dlS_dlP_T', 'grad_ad'
        ]

        T_vals, P_vals, rho_3D, grad_ad_3D = parse_EoS_H(filename, columns)
        interpolator_rho = RegularGridInterpolator((T_vals, P_vals), rho_3D)
        interpolator_grad_ad = RegularGridInterpolator((T_vals, P_vals), grad_ad_3D)
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

        T_vals, P_vals, rho_3D, grad_ad_3D = parse_EoS_H2O(filename, columns)
        interpolator_rho = RegularGridInterpolator((T_vals, P_vals), rho_3D)
        interpolator_grad_ad = RegularGridInterpolator((T_vals, P_vals), grad_ad_3D)
    else:
        print(f'invalid element: {element}')
        return

    # ------------ Inital conditions ------------
    m_grid, data = create_grids(M_surf, N, theta)

    data[0,0] = R_surf
    data[0,1] = M_surf
    data[0,2] = P_surf
    data[0,5] = T_surf

    # ------------- Run simulation --------------
    print(f'start tabulated {element} ({method}, m-grid)')
    
    # Define the system of ODEs
    def system_of_ODEs(m, y):
        r, P, T = y
        
        # Prevent numerical issues near center
        if m < 1e-10:
            m = 1e-10
        
        # ----------------- EoS -----------------
        if element == 'H':
            rho = EoS_tabulated_H(P, T, interpolator_rho)
        elif element == 'H2O':
            rho = EoS_tabulated_H2O(P, T, interpolator_rho)
        # ---------------------------------------
        
        # ODEs
        dr_dm = 1.0 / (4 * np.pi * r**2 * rho)
        dP_dm = -G * m / (4 * np.pi * r**4)

        # -------------- Find dT_dr -------------
        if element == 'H':
            P_query = P * 1e-10 # dyne/cm^2 -> GPa
            P_query = np.log10(P_query)
            T_query = np.log10(T) # K

            if T_query < min(T_vals) or  max(T_vals) < T_query:
                T_query = np.clip(T_query, min(T_vals), max(T_vals))

            if P_query < min(P_vals) or max(P_vals) < P_query:
                P_query = np.clip(P_query, min(P_vals), max(P_vals))

            grad_ad = interpolator_grad_ad([T_query, P_query])[0]

        elif element == 'H2O':
            P_query = P * 0.1 # dyne/cm^2 -> Pa
            T_query = T # K

            if T_query < min(T_vals) or  max(T_vals) < T_query:
                T_query = np.clip(T_query, min(T_vals), max(T_vals))

            if P_query < min(P_vals) or max(P_vals) < P_query:
                P_query = np.clip(P_query, min(P_vals), max(P_vals))

            grad_ad = interpolator_grad_ad([T_query, P_query])[0]

        dT_dm = T/P * dP_dm * grad_ad # = T2 - T1

        return [dr_dm, dP_dm, dT_dm]
    
    # Define events to stop integration
    def negative_radius_event(m, y):
        return y[0]  # Stop when radius becomes negative
    negative_radius_event.terminal = True
    negative_radius_event.direction = -1

    def negative_pressure_event(m, y):
        return y[1]  # Stop when pressure becomes negative
    negative_pressure_event.terminal = True
    negative_pressure_event.direction = -1
    
    def negative_temperature_event(m, y):
        return y[2]  # Stop when temperature becomes negative
    negative_temperature_event.terminal = True
    negative_temperature_event.direction = -1
    
    # Solve from surface to center
    m_span = (m_grid[0], m_grid[-1])
    
    # Set tolerances based on method
    if method == 'DOP853':
        rtol = 1e-10
        atol = 1e-12
    else:
        rtol = 1e-8
        atol = 1e-10
    
    sol = solve_ivp(
        system_of_ODEs,
        m_span,
        [R_surf, P_surf, T_surf], # = y0
        t_eval=m_grid,
        method=method,
        events=[negative_radius_event, negative_pressure_event, negative_temperature_event],
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=np.inf  # Let the solver choose step size
    )
    
    if sol.status == 1:  # Terminated by event
        if len(sol.t_events[0]) > 0:
            print(f' aborted at step {len(sol.t)}/{N}')
        elif len(sol.t_events[1]) > 0:
            print(f'>'*25, 'ALERT', '<'*25)
            print(f' negative pressure at step {len(sol.t)}/{N}')
        elif len(sol.t_events[2]) > 0:
            print(f'>'*25, 'ALERT', '<'*25)
            print(f' negative temperature at step {len(sol.t)}/{N}')
    elif sol.status == -1:
        print(f' Integration failed at step {len(sol.t)}/{N}: {sol.message}')
    
    # -------- In any case save the data --------
    # Store the solution
    n_points = len(sol.t)

    # Trim data array to actual size
    data = data[:n_points]
    
    data[:,0] = sol.y[0]  # radius
    data[:,1] = sol.t     # mass
    data[:,2] = sol.y[1]  # pressure
    data[:,5] = sol.y[2]  # temperature

    # Trim data array to actual size
    if data[:,0][-1] == 0.0:
        data = data[:-1]

    # ----------------- Density -----------------
    for i, (P, T) in enumerate(zip(data[:,2], data[:,5])):

        # ----------------- EoS -----------------
        if element == 'H':
            rho = EoS_tabulated_H(P, T, interpolator_rho)
        elif element == 'H2O':
            rho = EoS_tabulated_H2O(P, T, interpolator_rho)
        # ---------------------------------------

        data[i,3] = rho
    
    # ------------ Moment of Inertia ------------
    calculate_normalised_MoI(data, M_surf, R_surf)
    
    # ----------------- Save data ---------------
    save_data(data, N, output_name, grid_type='m')
    if save_plot:
        plot_data(data, N, output_name, grid_type='m', show_plot=show_plot)

# ============== helper functions ===============
def calculate_normalised_MoI(data, M_surf, R_surf):
    r2 = data[:,0][::-1]
    m2 = data[:,1][::-1]

    # Create shifted arrays for r1 and m1
    r1 = np.concatenate([[0], r2[:-1]])
    m1 = np.concatenate([[0], m2[:-1]])

    # Vectorized calculation of MoI for each shell
    dm = m2 - m1
    dr3 = r2**3 - r1**3
    dr5 = r2**5 - r1**5

    # Avoid division by zero - use safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(dr3 != 0, dr5/dr3, 0)
    MoI_shells = dm * 2/5 * ratio
    # Cumulative sum for total MoI at each radius
    MoIs = np.cumsum(MoI_shells)

    # Normalize
    MoIs /= (M_surf * R_surf**2)

    # Save (reverse back to original order)
    data[:,4] = MoIs[::-1]

def create_grids(max_val, N, theta):
    # -------------------- grid ---------------------
    '''
    r_grid: We want to sampe more r values closer to the surface because P changes there more rapidly. 
    m_grid: We want to sampe more m values closer to the center because 1/r^4 blows up. 
    '''
    s = np.linspace(0.0, 1.0, N+1)          # normalized coordinates
    grid = max_val * (1 - s**theta)    # power-law stretched coordinates

    # ---------------- prepare data -----------------
    data = np.zeros((N+1,6)) # [r, m, p, rho, moi, T]
    return grid, data

def plot_data(data, N, filename, grid_type='r', sim_method='Euler', show_plot=False):

    # ------------------ path -------------------
    folder_path = os.path.join('plots', f'{grid_type}_grid', f'N={N}')
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'{filename}.pdf')

    # ------------------ title ------------------
    name_parts = filename.split('_')
    planet_name = name_parts[0]
    N_eff = len(data[:,0])
    theta = float(name_parts[-1].replace('.csv', ''))
    if theta.is_integer():
        theta = int(theta)

    sim_name = ''
    for part in name_parts[2:-4]:
        if part == 'MgSiO3':
            part = 'MgSiO$_3$'
        if part and part[0].islower():
            part = part[0].upper() + part[1:]
        sim_name += part + ' '

    MoI = data[0,4]
    title_top = f'{planet_name} \u2013 {sim_name} \u2013 {sim_method} \u2013 {grid_type}-grid'
    title_bottom = f'\nN={N_eff}, $\\theta$={theta}, ' + '$I_\\text{norm}$=' + f'{MoI:.2e}'
    title = title_top + title_bottom

    # ------------------- plot ------------------
    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(data[:,0], data[:,1], '.-')
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
        
    if data[:,5][-1] != 0:
        ax4 = plt.subplot(2, 2, 4, sharex=ax1)
        ax4.plot(data[:,0], data[:,5], '.-')
        ax4.set_xlabel('r [cm]')
        ax4.set_ylabel('T [K]')
        ax4.set_title('Temperature')
        ax4.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(file_path)

    if show_plot:
        plt.show()
    plt.close()

def save_data(data, N, filename, grid_type='r'):
    header = 'r [cm], m [g], p [dyne/cm^2], rho [g/cm^3], I_norm [1], T [K]'
    folder_path = os.path.join('data', 'simulation_results', f'{grid_type}_grid', f'N={N}')
    os.makedirs(folder_path, exist_ok=True)
    path = os.path.join(folder_path, f'{filename}.csv')
    np.savetxt(path, data, delimiter=',', header=header)

def solve_Euler_with_events(r1, r2, m1, P1, rho, data, i):
    data[i,3] = rho

    # ---------- Calculate derivatives ----------
    if r2 < 1e-10: # handle div by 0 error
        r2 = 1e-10 

    dm_dr = 4 * np.pi * r1**2 * rho
    dP_dr = -G * m1 * rho / r1**2
    
    # ------ Update values (Euler method) -------
    dr = r2 - r1
    m2 = m1 + dm_dr * dr
    P2 = P1 + dP_dr * dr

    # -------- Check if data makes sense --------
    if P2 < 0.0:
        print('>'*25, 'ALERT', '<'*25)
        print(f'negative pressure at {i+1}/{N}')
        return False

    if m2 < 0.0:
        print(f' aborted at {i+1}/{N}')
        return False
    
    # --------- Save data for next step ---------
    if i < len(data)-1:
        data[i+1,0] = r2
        data[i+1,1] = m2
        data[i+1,2] = P2

    return True

def solve_Euler_with_events_m_grid(m1, m2, r1, P1, rho, data, i):
    data[i,3] = rho

    # ---------- Calculate derivatives ----------
    if m2 < 1e-10: # handle div by 0 error
        m2 = 1e-10 

    dr_dm = 1.0 / (4 * np.pi * r1**2 * rho)
    dP_dm = -G * m1 / (4 * np.pi * r1**4)
    
    # ------ Update values (Euler method) -------
    dm = m2 - m1
    r2 = r1 + dr_dm * dm
    P2 = P1 + dP_dm * dm

    # -------- Check if data makes sense --------
    if P2 < 0.0:
        print('>'*25, 'ALERT', '<'*25)
        print(f'negative pressure at {i+1}/{N}')
        return False

    if r2 < 0.0:
        print(f' aborted at {i+1}/{N}')
        return False
    
    # --------- Save data for next step ---------
    if i < len(data)-1:
        data[i+1,0] = r2
        data[i+1,1] = m2
        data[i+1,2] = P2

    return True

def solve_ivp_with_events(system_func, r_grid, y0, method, N, data):
    """
    Wrapper for solve_ivp that handles events and integration.
    This is reusable for different EoS systems.
    
    Parameters:
        system_func: function defining the ODEs
        r_grid: radial grid points
        y0: initial conditions [m0, P0, ...]
        method: ODE solver method
        N: number of steps
        data: (N, 6) array to save the data in (passed by reference)
    
    Returns:
        data: (N, 6) array with R, M, and P filled in
    """
    
    # Define events to stop integration
    
    def negative_mass_event(r, y):
        return y[0]  # Stop when mass becomes negative
    negative_mass_event.terminal = True
    negative_mass_event.direction = -1

    def negative_pressure_event(r, y):
        return y[1]  # Stop when pressure becomes negative
    negative_pressure_event.terminal = True
    negative_pressure_event.direction = -1
    
    r_span = (r_grid[0], r_grid[-1])
    
    # Set tolerances based on method
    if method == 'DOP853':
        rtol = 1e-10
        atol = 1e-12
    else:
        rtol = 1e-8
        atol = 1e-10
    
    sol = solve_ivp(
        system_func,
        r_span,
        y0,
        method=method,
        t_eval=r_grid,
        events=[negative_mass_event, negative_pressure_event],
        rtol=rtol,
        atol=atol,
        max_step=np.inf  # Let the solver choose step size
    )
    
    if sol.status == 1:  # Terminated by event
        if len(sol.t_events[0]) > 0:
            print(f' aborted at step {len(sol.t)}/{N}')
        elif len(sol.t_events[1]) > 0:
            print(f'>'*25, 'ALERT', '<'*25)
            print(f' negative pressure at step {len(sol.t)}/{N}')
    elif sol.status == -1:
        print(f' Integration failed at step {len(sol.t)}/{N}: {sol.message}')

    
    # -------- In any case save the data --------
    # Store the solution
    n_points = len(sol.t)

    # Trim data array to actual size
    data = data[:n_points]

    data[:,0] = sol.t     # radius
    data[:,1] = sol.y[0]  # mass
    data[:,2] = sol.y[1]  # pressure

    # Trim data array to actual size
    if data[:,0][-1] == 0.0:
        data = data[:-1]
    
    return data

def solve_ivp_with_events_m_grid(system_func, m_grid, y0, method, N, data):
    """
    Wrapper for solve_ivp that handles events and integration.
    This is reusable for different EoS systems.
    
    Parameters:
        system_func: function defining the ODEs
        m_grid: mass grid points
        y0: initial conditions [m0, P0, ...]
        method: ODE solver method
        N: number of steps
        data: (N, 6) array to save the data in (passed by reference)
    
    Returns:
        data: (N, 6) array with R, M, and P filled in
    """
    
    # Define events to stop integration
    
    def negative_radius_event(m, y):
        return y[0]  # Stop when radius becomes negative
    negative_radius_event.terminal = True
    negative_radius_event.direction = -1

    def negative_pressure_event(m, y):
        return y[1]  # Stop when pressure becomes negative
    negative_pressure_event.terminal = True
    negative_pressure_event.direction = -1

    # Solve from surface to center
    m_span = (m_grid[0], m_grid[-1])
    
    # Set tolerances based on method
    if method == 'DOP853':
        rtol = 1e-10
        atol = 1e-12
    else:
        rtol = 1e-8
        atol = 1e-10
    
    sol = solve_ivp(
        system_func,
        m_span,
        y0,
        t_eval=m_grid,
        method=method,
        events=[negative_radius_event, negative_pressure_event],
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=np.inf  # Let the solver choose step size
    )
    
    if sol.status == 1:  # Terminated by event
        if len(sol.t_events[0]) > 0:
            print(f' aborted at step {len(sol.t)}/{N}')
        elif len(sol.t_events[1]) > 0:
            print(f'>'*25, 'ALERT', '<'*25)
            print(f' negative pressure at step {len(sol.t)}/{N}')
    elif sol.status == -1:
        print(f' Integration failed at step {len(sol.t)}/{N}: {sol.message}')
    
    # -------- In any case save the data --------
    n_points = len(sol.t)

    # Trim data array to actual size
    data = data[:n_points]
    
    data[:,0] = sol.y[0]  # radii
    data[:,1] = sol.t     # mass
    data[:,2] = sol.y[1]  # pressure

    # Trim data array to actual size
    if data[:,0][-1] == 0.0:
        data = data[:-1]
    
    return data

def simulate_planet(planet, EoS, R=0, M=0, P_surface=0, T_surface=0, grid_type='r', solver_method='RK45', save_plot=False):
    '''
    Parameters:
        planet: name (Earth, Jupiter, Saturn and Uranus don't require parameters)
        EoS: list of int representing the Simulations that should be done.
            1: ideal gas
            2: polytropic
            3: analytical Fe
            4: analytical MgSiO3
            5: tabulated H
            6: tabulated H2O
        R: [cm] mean radius
        M: [g] mass
        P_surface: [dyne/cm^2] surface pressure
        T_surface: [K] surface temperature
    '''
    global C_ideal_gas, G
    # ----------- boundary conditions -----------
    if planet == 'Jupiter':
        R = 6.9911e9        # cm        [A]
        M = 1.898125e30     # g         [F]
        P_surface = 1e6     # dyne/cm^2 = 1 Bar
        T_surface = 165     # K

    elif planet == 'Saturn':
        R = 5.8232e9        # cm        [A]
        M = 5.6836e29       # g         [G]
        P_surface = 1e6     # dyne/cm^2 = 1 Bar
        T_surface = 134     # K

    elif planet == 'Uranus':
        R = 2.5362e9        # cm        [A][B]
        M = 8.68099e+28     # g         [H][C]
        P_surface = 1e6     # dyne/cm^2 = 1 Bar
        T_surface = 76      # K         [D]

    elif planet == 'Earth':
        R = 6.371e8         # cm        [A]
        M = 5.97217e27      # g         [E]
        P_surface = 1e6     # dyne/cm^2 = 1 Bar
        T_surface = 288     # K     

    # ----------- Physical Constants ------------
    G = 6.67430e-8  # cm^3 g^-1 s^-2

    R_gas = 8.314e7 # dyne cm^2 mol^-1 K-1
    rho_surface = P_surface/(R_gas*T_surface)
    gamma = 5/3
    C_ideal_gas = P_surface * (1/rho_surface)**gamma

    # ---------------- References ---------------
    # [A]   Archinal, B.A. et al. 2018. "Report of the IAU/IAG Working Group on cartographic coordinates and rotational elements: 2015" Celestial Mech. Dyn. Astr. 130:22.
    # [B]   https://doi.org/10.1007%2Fs10569-007-9072-y
    # [C]   https://ui.adsabs.harvard.edu/abs/1992AJ....103.2068J/abstract
    # [D]   https://www.sciencedirect.com/science/article/abs/pii/0032063395000615?via%3Dihub
    # [E]   Folkner, W.M. and Williams, J.G. 2008. "Mass parameters and uncertainties in planetary ephemeris DE421." Interoffice Memo. 343R-08-004 (internal document), Jet Propulsion Laboratory, Pasadena, CA.
    # [F]   https://ssd.jpl.nasa.gov/planets/phys_par.html
    # [G]   Jacobson, R.A., et al. 2006. "The gravity field of the Saturnian system from satellite observations and spacecraft tracking data" AJ 132(6):2520-2526.
    # [H]   Jacobson, R.A. 2014. "The Orbits of the Uranian Satellites and Rings, the Gravity Field of the Uranian System, and the Orientation of the Pole of Uranus" AJ 148:76-88.
    # [I]   Helled, R. & Guillot, T. (2018). Internal Structure of Giant and Icy Planets: Importance of Heavy Elements and Mixing. H.J. Deeg, J.A. Belmonte (eds.), Handbook of Exoplanets.

    if R == 0 or M == 0 or P_surface == 0:
        print('invalid boundary conditions')
        return
    
    if (5 in EoS or 6 in EoS) and T_surface == 0:
        print('invalid boundary conditions')
        return

    # --------------- simulations ---------------
    print('\n'+('='*10), planet, '='*10)
    if grid_type == 'r':
        if solver_method == 'Euler':
            if 1 in EoS:
                simulate_ideal_gas_Euler(
                    R, 
                    M,
                    P_surface,
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_01_ideal_gas_Euler_theta_{theta}',
                    save_plot=save_plot
                )
            if 2 in EoS:
                simulate_polytropic_Euler(
                    R, 
                    M,
                    P_surface,
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_02_polytropic_Euler_theta_{theta}',
                    save_plot=save_plot
                )
            if 3 in EoS:
                simulate_analytical_Euler(
                    R, 
                    M,
                    P_surface,
                    element='Fe',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_03_analytical_Fe_Euler_theta_{theta}',
                    save_plot=save_plot
                )
            if 4 in EoS:
                simulate_analytical_Euler(
                    R, 
                    M,
                    P_surface,
                    element='MgSiO3',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_04_analytical_MgSiO3_Euler_theta_{theta}',
                    save_plot=save_plot
                )
            if 5 in EoS:
                simulate_tabulated_Euler(
                    R, 
                    M,
                    P_surface,
                    T_surface,
                    element='H',
                    filename='data/EoS_H/TABLEEOS_2021_TP_Y0275_v1.csv',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_05_tabulated_H_Euler_theta_{theta}',
                    save_plot=save_plot
                )
            if 6 in EoS:
                simulate_tabulated_Euler(
                    R, 
                    M,
                    P_surface,
                    T_surface,
                    element='H2O',
                    filename='data/EoS_H2O/aqua_eos_pt_v1_0.dat',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_06_tabulated_H2O_Euler_theta_{theta}',
                    save_plot=save_plot
                )
        else:
            if 1 in EoS:
                simulate_ideal_gas_ivp(
                    R, 
                    M,
                    P_surface,
                    solver_method,
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_01_ideal_gas_{solver_method}_theta_{theta}',
                    save_plot=save_plot
                )
            if 2 in EoS:
                simulate_polytropic_ivp(
                    R, 
                    M,
                    P_surface,
                    solver_method,
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_02_polytropic_{solver_method}_theta_{theta}',
                    save_plot=save_plot
                )
            if 3 in EoS:
                simulate_analytical_ivp(
                    R, 
                    M,
                    P_surface,
                    solver_method,
                    element='Fe',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_03_analytical_Fe_{solver_method}_theta_{theta}',
                    save_plot=save_plot
                )
            if 4 in EoS:
                simulate_analytical_ivp(
                    R, 
                    M,
                    P_surface,
                    solver_method,
                    element='MgSiO3',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_04_analytical_MgSiO3_{solver_method}_theta_{theta}',
                    save_plot=save_plot
                )
            if 5 in EoS:
                simulate_tabulated_ivp(
                    R, 
                    M,
                    P_surface,
                    T_surface,
                    solver_method,
                    element='H',
                    filename='data/EoS_H/TABLEEOS_2021_TP_Y0275_v1.csv',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_05_tabulated_H_{solver_method}_theta_{theta}',
                    save_plot=save_plot
                )
            if 6 in EoS:
                simulate_tabulated_ivp(
                    R, 
                    M,
                    P_surface,
                    T_surface,
                    solver_method,
                    element='H2O',
                    filename='data/EoS_H2O/aqua_eos_pt_v1_0.dat',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_06_tabulated_H2O_{solver_method}_theta_{theta}',
                    save_plot=save_plot
                )
    elif grid_type == 'm':
        if solver_method == 'Euler':
            if 1 in EoS:
                simulate_ideal_gas_Euler_m_grid(
                    R, 
                    M,
                    P_surface,
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_01_ideal_gas_Euler_theta_{theta}',
                    save_plot=save_plot
                )
            if 2 in EoS:
                simulate_polytropic_Euler_m_grid(
                    R, 
                    M,
                    P_surface,
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_02_polytropic_Euler_theta_{theta}',
                    save_plot=save_plot
                )
            if 3 in EoS:
                simulate_analytical_Euler_m_grid(
                    R, 
                    M,
                    P_surface,
                    element='Fe',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_03_analytical_Fe_Euler_theta_{theta}',
                    save_plot=save_plot
                )
            if 4 in EoS:
                simulate_analytical_Euler_m_grid(
                    R, 
                    M,
                    P_surface,
                    element='MgSiO3',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_04_analytical_MgSiO3_Euler_theta_{theta}',
                    save_plot=save_plot
                )
            if 5 in EoS:
                simulate_tabulated_Euler_m_grid(
                    R, 
                    M,
                    P_surface,
                    T_surface,
                    element='H',
                    filename='data/EoS_H/TABLEEOS_2021_TP_Y0275_v1.csv',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_05_tabulated_H_Euler_theta_{theta}',
                    save_plot=save_plot
                )
            if 6 in EoS:
                simulate_tabulated_Euler_m_grid(
                    R, 
                    M,
                    P_surface,
                    T_surface,
                    element='H2O',
                    filename='data/EoS_H2O/aqua_eos_pt_v1_0.dat',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_06_tabulated_H2O_Euler_theta_{theta}',
                    save_plot=save_plot
                )
        else:
            if 1 in EoS:
                simulate_ideal_gas_ivp_m_grid(
                    R, 
                    M,
                    P_surface,
                    solver_method,
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_01_ideal_gas_{solver_method}_theta_{theta}',
                    save_plot=save_plot
                )
            if 2 in EoS:
                simulate_polytropic_ivp_m_grid(
                    R, 
                    M,
                    P_surface,
                    solver_method,
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_02_polytropic_{solver_method}_theta_{theta}',
                    save_plot=save_plot
                )
            if 3 in EoS:
                simulate_analytical_ivp_m_grid(
                    R, 
                    M,
                    P_surface,
                    solver_method,
                    element='Fe',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_03_analytical_Fe_{solver_method}_theta_{theta}',
                    save_plot=save_plot
                )
            if 4 in EoS:
                simulate_analytical_ivp_m_grid(
                    R, 
                    M,
                    P_surface,
                    solver_method,
                    element='MgSiO3',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_04_analytical_MgSiO3_{solver_method}_theta_{theta}',
                    save_plot=save_plot
                )
            if 5 in EoS:
                simulate_tabulated_ivp_m_grid(
                    R, 
                    M,
                    P_surface,
                    T_surface,
                    solver_method,
                    element='H',
                    filename='data/EoS_H/TABLEEOS_2021_TP_Y0275_v1.csv',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_05_tabulated_H_{solver_method}_theta_{theta}',
                    save_plot=save_plot
                )
            if 6 in EoS:
                simulate_tabulated_ivp_m_grid(
                    R, 
                    M,
                    P_surface,
                    T_surface,
                    solver_method,
                    element='H2O',
                    filename='data/EoS_H2O/aqua_eos_pt_v1_0.dat',
                    N=N,
                    theta=theta,
                    output_name=f'{planet}_06_tabulated_H2O_{solver_method}_theta_{theta}',
                    save_plot=save_plot
                )
    else:
        print('invalid grid type:', grid_type)

if __name__ == '__main__':
    warning_tracker = WarningTracker()
    warning_tracker.suppress_spam = True

    N = 1000000
    theta = 2 # If for high N: ValueError: Values in `t_eval` are not properly sorted. -> reduce theta
    method = 'RK45' # 'Euler', 'RK45', 'DOP853', 'Radau' 

    for planet in ['Earth', 'Jupiter', 'Saturn', 'Uranus']:
        for meth in ['RK45']:
            for gt in ['r']:
                simulate_planet(planet,  [4], solver_method=meth, grid_type=gt, save_plot=False)

    # for meth in ['RK45']:
    #     for gt in ['r']:
    #         simulate_planet('Jupiter',  [1,2,3,4,5,6], solver_method=meth, grid_type=gt)

