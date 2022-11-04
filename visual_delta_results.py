from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import scenario.common as cmn
from environment import command_parser, OUTPUT_DIR, DATADIR
from test_theta import file_suffixes

if __name__ == '__main__':
    render = command_parser()[0]

    # Var delta plot
    ## Generate axis data
    delta_vec = np.arange(0.1, 1, 0.1)
    Psi_Gau = [0, 45, -45]
    num_points = len(delta_vec)

    # Load values
    power_pc = np.zeros((num_points, 3))
    power_opt = np.zeros((num_points, 3))
    errorp_pc = np.zeros((num_points, 3))
    errorp_opt = np.zeros((num_points, 3))

    for s in range(len(file_suffixes)):
        try:
            filename = path.join(DATADIR, f'delta_{file_suffixes[s]}.npz')
            data = np.load(filename)
            power_pc[:, s] = data['power_pc']
            power_opt[:, s] = data['power_opt']
            errorp_pc[:, s] = data['errorp_pc']
            errorp_opt[:, s] = data['errorp_opt']
        except FileNotFoundError:
            pass

    # Plotting
    e_fig, e_ax = plt.subplots()
    p_fig, p_ax = plt.subplots()
    colors = ['red', 'blue', 'magenta']
    for s in range(3):
        # Smoothing power data
        p_pc = savgol_filter(cmn.watt2dbm(power_pc[:, s]), 5, 3)

        p_ax.plot(delta_vec, p_pc, label=f'PC {Psi_Gau[s]}', marker='^', c=colors[s], ls='dashed')
        e_ax.semilogy(delta_vec, errorp_pc[:, s], label=f'PC {Psi_Gau[s]}', marker='^', c=colors[s], ls='dashed')

    cmn.printplot(p_fig, p_ax, render=render,
                  filename=f'linear_power_consumption', dirname=OUTPUT_DIR,
                  title=f'Power needed', labels=[r'$\alpha$', r'$P$ [dBm]'])

    cmn.printplot(e_fig, e_ax, render=render,
                  filename=f'linear_outage', dirname=OUTPUT_DIR,
                  title=f'Outage achieved', labels=[r'$\alpha$ [deg]', r'$1 - p_s$'])