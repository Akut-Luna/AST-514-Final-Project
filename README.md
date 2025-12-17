Final project for AST 514 Planet Formation
Note that the data is not included due to their size. The file structur for the data is expected to look like this:
Root
└── data
    ├── EoS_Fe
    │   └── EoS_Fe.csv
    ├── EoS_H
    │   └── TABLEEOS_2021_TP_Y0275_v1.csv
    ├── EoS_H2O
    │   └── aqua_eos_pt_v1_0.dat
    ├── EoS_MgSiO3
    │   └── EoS_MgSiO3.csv
    ├── Literature
    │   ├── Earth
    │   │   └── PREM500.csv
    │   ├── Jupiter
    │   │   └── density.csv
    │   ├── Saturn
    │   │   └── density.csv
    │   └── Uranus
    │       └── gen_4_comp_for_014_025_009_012_011_024_020_004.npz
    └── simulation_results
        ├── plot_1
        │   ├── Euler
        │   │   ├── 1e2.csv
        │   │   ├── 1e3.csv
        │   │   ├── 1e4.csv
        │   │   ├── 1e5.csv
        │   │   └── 1e6.csv
        │   └── RK45
        │       ├── 1e2.csv
        │       ├── 1e3.csv
        │       ├── 1e4.csv
        │       ├── 1e5.csv
        │       └── 1e6.csv
        ├── plot_2
        │   ├── m.csv
        │   └── r.csv
        ├── plot_3
        │   ├── 38_Jupiter_06_tabulated_H2O_DOP853_theta_2.csv
        │   ├── 38_Jupiter_06_tabulated_H2O_Euler_theta_2.csv
        │   ├── 38_Jupiter_06_tabulated_H2O_RK45_theta_2.csv
        │   ├── 38_Jupiter_06_tabulated_H2O_Radau_theta_2.csv
        │   ├── Jupiter_06_tabulated_H2O_DOP853_theta_2.csv
        │   ├── Jupiter_06_tabulated_H2O_Euler_theta_2.csv
        │   ├── Jupiter_06_tabulated_H2O_RK45_theta_2.csv
        │   └── Jupiter_06_tabulated_H2O_Radau_theta_2.csv
        ├── plot_4
        │   ├── Earth
        │   │   ├── Earth_01_ideal_gas_RK45_theta_2.csv
        │   │   ├── Earth_02_polytropic_RK45_theta_2.csv
        │   │   ├── Earth_03_analytical_Fe_RK45_theta_2.csv
        │   │   ├── Earth_04_analytical_MgSiO3_RK45_theta_2.csv
        │   │   ├── Earth_05_tabulated_H_RK45_theta_2.csv
        │   │   └── Earth_06_tabulated_H2O_RK45_theta_2.csv
        │   ├── Jupiter
        │   │   ├── Jupiter_01_ideal_gas_RK45_theta_2.csv
        │   │   ├── Jupiter_02_polytropic_RK45_theta_2.csv
        │   │   ├── Jupiter_03_analytical_Fe_RK45_theta_2.csv
        │   │   ├── Jupiter_04_analytical_MgSiO3_RK45_theta_2.csv
        │   │   ├── Jupiter_05_tabulated_H_RK45_theta_2.csv
        │   │   └── Jupiter_06_tabulated_H2O_RK45_theta_2.csv
        │   ├── Saturn
        │   │   ├── Saturn_01_ideal_gas_RK45_theta_2.csv
        │   │   ├── Saturn_02_polytropic_RK45_theta_2.csv
        │   │   ├── Saturn_03_analytical_Fe_RK45_theta_2.csv
        │   │   ├── Saturn_04_analytical_MgSiO3_RK45_theta_2.csv
        │   │   ├── Saturn_05_tabulated_H_RK45_theta_2.csv
        │   │   └── Saturn_06_tabulated_H2O_RK45_theta_2.csv
        │   └── Uranus
        │       ├── Uranus_01_ideal_gas_RK45_theta_2.csv
        │       ├── Uranus_02_polytropic_RK45_theta_2.csv
        │       ├── Uranus_03_analytical_Fe_RK45_theta_2.csv
        │       ├── Uranus_04_analytical_MgSiO3_RK45_theta_2.csv
        │       ├── Uranus_05_tabulated_H_RK45_theta_2.csv
        │       └── Uranus_06_tabulated_H2O_RK45_theta_2.csv
        ├── plot_5
        │   ├── 01
        │   │   ├── Earth_01_ideal_gas_RK45_theta_2.csv
        │   │   ├── Jupiter_01_ideal_gas_RK45_theta_2.csv
        │   │   ├── Saturn_01_ideal_gas_RK45_theta_2.csv
        │   │   └── Uranus_01_ideal_gas_RK45_theta_2.csv
        │   ├── 02
        │   │   ├── Earth_02_polytropic_RK45_theta_2.csv
        │   │   ├── Jupiter_02_polytropic_RK45_theta_2.csv
        │   │   ├── Saturn_02_polytropic_RK45_theta_2.csv
        │   │   └── Uranus_02_polytropic_RK45_theta_2.csv
        │   ├── 03
        │   │   ├── Earth_03_analytical_Fe_RK45_theta_2.csv
        │   │   ├── Jupiter_03_analytical_Fe_RK45_theta_2.csv
        │   │   ├── Saturn_03_analytical_Fe_RK45_theta_2.csv
        │   │   └── Uranus_03_analytical_Fe_RK45_theta_2.csv
        │   ├── 04
        │   │   ├── Earth_04_analytical_MgSiO3_RK45_theta_2.csv
        │   │   ├── Jupiter_04_analytical_MgSiO3_RK45_theta_2.csv
        │   │   ├── Saturn_04_analytical_MgSiO3_RK45_theta_2.csv
        │   │   └── Uranus_04_analytical_MgSiO3_RK45_theta_2.csv
        │   ├── 05
        │   │   ├── Earth_05_tabulated_H_RK45_theta_2.csv
        │   │   ├── Jupiter_05_tabulated_H_RK45_theta_2.csv
        │   │   ├── Saturn_05_tabulated_H_RK45_theta_2.csv
        │   │   └── Uranus_05_tabulated_H_RK45_theta_2.csv
        │   └── 06
        │       ├── Earth_06_tabulated_H2O_RK45_theta_2.csv
        │       ├── Jupiter_06_tabulated_H2O_RK45_theta_2.csv
        │       ├── Saturn_06_tabulated_H2O_RK45_theta_2.csv
        │       └── Uranus_06_tabulated_H2O_RK45_theta_2.csv
        └── plot_MoI
            ├── Earth_04_analytical_MgSiO3_RK45_theta_2.csv
            ├── Jupiter_02_polytropic_RK45_theta_2.csv
            ├── Saturn_02_polytropic_RK45_theta_2.csv
            └── Uranus_05_tabulated_H_RK45_theta_2.csv