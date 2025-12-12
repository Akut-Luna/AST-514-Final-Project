import numpy as np
import matplotlib.pylab as plt

def EoS_Vinet(eta, K0, K0_prime):
    P = 3*K0 * eta**(2/3) * (1-eta**(-1/3)) * np.exp(3/2 * (K0_prime - 1)*(1-eta**(-1/3)))
    return P

def EoS_BME(eta, K0, K0_prime):
    P = 3/2*K0 * (eta**(7/3) - eta**(5/3)) * (1 + 3/4 * (K0_prime - 4) * (eta**(2/3) - 1))
    return P

def EoS_BME4(eta, K0, K0_prime, K0_double_prime):
    # P = EoS_BME(eta, K0, K0_prime)
    # term1 = 3/2*K0 * (eta**(7/3) - eta**(5/3))
    # term2 = 3/8 * K0 * (eta**(2/3) - 1)**2
    # term3 = K0*K0_double_prime + K0_prime*(K0_prime - 7) + 143/9
    # P +=  term1 * term2 * term3 # <--- PROBLEM
    # return P

    K_0 = K0
    K_1 = K0_prime
    K_2 = K0_double_prime

    test0 = 1.5*K_0*(eta**(7./3.) - eta**(5./3.))
    test1 = (1 + 0.75*(K_1 - 4.)*(eta**(2./3.) - 1))
    test2 = (3./8.)*(eta**(2./3.) - 1)*(eta**(2./3.) - 1)*(K_0*K_2 + K_1*(K_1 - 7.) + 143./9.)
    
    return test0*(test1 + test2)


if __name__ == '__main__':

    N = 100000
    rho_min = 1e2   # kg/m^3
    rho_max = 1e8   # kg/m^3 
    rho_grid = np.logspace(2, 8, N)  # Using log spacing for better coverage
    rho_grid = rho_grid * 1e-3       # kg/m^3 -> g/cm^3
    
    # Iron
    K0 = 156.2 # GPa
    K0_prime = 6.08
    rho0 = 8.30 # Mg/m^3 = g/cm^3
    data = np.zeros((N,2))
    for i, rho in enumerate(rho_grid):
        eta = rho/rho0
        P = EoS_Vinet(eta, K0, K0_prime)
        data[i,0] = P
        data[i,1] = rho

    data = data[data[:, 0].argsort()] # sort by pressure
    data = data[data[:, 0] > 0]       # disregard negative pressures

    header = 'P [GPa], rho [Mg/m^3]'
    np.savetxt(f'data/EoS_Fe/EoS_Fe.csv', data, delimiter=',', header=header)

    data_fe = data

    # Silicat
    K0 = 247 # GPa
    K0_prime = 3.97 
    K0_double_prime = -0.016 # 1/GPa
    rho0 = 4.10 # Mg/m^3
    data = np.zeros((N,2))
    for i, rho in enumerate(rho_grid):
        eta = rho/rho0
        P = EoS_BME4(eta, K0, K0_prime, K0_double_prime)
        data[i,0] = P
        data[i,1] = rho

    data = data[data[:, 0].argsort()] # sort by pressure
    
    data_mg = data

    # filter out everthing above 3*10^4 kg/m^3
    data_new = []
    for datapoint in data:
        if datapoint[1] < 3e4 * 1e-3:
            data_new.append(datapoint)
    data = np.array(data_new)

    header = 'P [GPa], rho [g/m^3]'
    np.savetxt(f'data/EoS_MgSiO3/EoS_MgSiO3.csv', data, delimiter=',', header=header)

    data_filt = data

    # # plot
    # plt.figure(figsize=(9, 7))

    # Use a shared y-axis
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    (ax1, ax2, ax3) = axes

    # ---- Fe ----
    ax1.loglog(data_fe[:,0]*1e9, data_fe[:,1]*1e3)
    ax1.text(0.02, 0.92, "Fe", transform=ax1.transAxes, fontsize=12, va='top')
    ax1.grid(True, alpha=0.3)

    # ---- MgSiO3 ----
    ax2.loglog(data_mg[:,0]*1e9, data_mg[:,1]*1e3)
    ax2.axvline(x=1.35e4*1e9, color='red', linestyle='--', linewidth=1)
    ax2.set_ylim([1e3, 1e5])
    ax2.text(0.02, 0.92, "MgSiO$_3$", transform=ax2.transAxes, fontsize=12, va='top')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('$\\rho$ [kg/m$^3$]')

    # ---- MgSiO3 filtered ----
    ax3.loglog(data_filt[:,0]*1e9, data_filt[:,1]*1e3)
    ax3.axvline(x=1.35e4*1e9, color='red', linestyle='--', linewidth=1)
    ax3.text(0.02, 0.92, "MgSiO$_3$ (filtered)", transform=ax3.transAxes, fontsize=12, va='top')
    ax3.set_ylim([1e3, 1e5])
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('P [Pa]')

    plt.tight_layout()
    plt.savefig('plots/plot_0_look_up_tables.pdf')
    plt.show()