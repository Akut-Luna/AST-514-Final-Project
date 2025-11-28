import numpy as np
import matplotlib.pylab as plt

def EoS_Vinet(eta, K0, K0_prime):
    p = 3*K0 * eta**(2/3) * (1-eta**(-1/3)) * np.exp(3/2 * (K0_prime - 1)*(1-eta**(-1/3)))
    return p

def EoS_BME(eta, K0, K0_prime):
    p = 3/2*K0 * (eta**(7/3) - eta**(5/3)) * (1 + 3/4 * (K0_prime - 4) * (eta**(2/3) - 1))
    return p

def EoS_BME4(eta, K0, K0_prime, K0_double_prime):
    p = EoS_BME(eta, K0, K0_prime)
    term1 = 3/2*K0 * (eta**(7/3) - eta**(5/3))
    term2 = 3/8 * K0 * (eta**(2/3) - 1)**2
    term3 = K0*K0_double_prime + K0_prime*(K0_prime - 7) + 143/9
    p +=  term1 * term2 * term3 # <--- PROBLEM
    return p

if __name__ == '__main__':

    N = 100000
    rho0 = 1e-6 # Mg/m^3
    rho1 = 100  # Mg/m^3

    rho_grid = np.linspace(rho0, rho1, N)
    
    theta = 1
    rho_grid = np.linspace(0, 1, N)
    rho_grid = rho1 * (1 - rho_grid**theta) + 1e-6  # power-law stretched coordinates

    # Iron
    K0 = 156.2 # GPa
    K0_prime = 6.08
    rho0 = 8.30 # Mg/m^3
    data = np.zeros((N,2))
    for i, rho in enumerate(rho_grid):
        eta = rho/rho0
        p = EoS_Vinet(eta, K0, K0_prime)
        data[i,0] = p
        data[i,1] = rho

    data = data[data[:, 0].argsort()] # sort by pressure
    data = data[data[:, 0] > 0]       # disregard negative pressures

    header = 'p [GPa], rho [Mg/m^3]'
    np.savetxt(f'data/EoS_Fe/EoS_Fe.csv', data, delimiter=',', header=header)

    plt.figure(figsize=(12,4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(data[:,0], data[:,1])
    ax1.set_xlabel('p [GPa]')
    ax1.set_ylabel('rho [Mg/m^3]')
    ax1.set_title('Iron')
    ax1.grid(True, alpha=0.3)

    # Silicat
    K0 = 247 # GPa
    K0_prime = 3.97 
    K0_double_prime = -0.016 # 1/GPa
    rho0 = 4.10 # Mg/m^3
    data = np.zeros((N,2))
    for i, rho in enumerate(rho_grid):
        eta = rho/rho0
        p = EoS_BME4(eta, K0, K0_prime, K0_double_prime)
        data[i,0] = p
        data[i,1] = rho

    data = data[data[:, 0].argsort()] # sort by pressure
    data = data[data[:, 0] > 0]       # disregard negative pressures

    header = 'p [GPa], rho [Mg/m^3]'
    np.savetxt(f'data/EoS_Si/EoS_Si.csv', data, delimiter=',', header=header)

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(data[:,0], data[:,1], '.')
    ax2.set_xlabel('p [GPa]')
    ax2.set_ylabel('rho [Mg/m^3]')
    ax2.set_title('Silicat')
    ax2.grid(True, alpha=0.3)

    plt.show()
