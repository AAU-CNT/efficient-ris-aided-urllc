from os import path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import scenario.common as cmn
from environment import command_parser, OUTPUT_DIR, RISRoofEnv, DATADIR, SIDE, H
from test_grid import dx, dy
from test_theta import file_suffixes, title_suffixes

if __name__ == '__main__':
    render = command_parser()[0]

    # TAKE VALUES
    side_x = SIDE
    side_y = SIDE

    ## SCATTER PLOTS
    # Covariance parameter
    Psi_Gau = [0, 45, -45]
    Psi = np.deg2rad(Psi_Gau[2])

    # sample drawing (ellipses added in latex later)
    el = np.deg2rad(30)
    az = np.deg2rad(45)
    env = RISRoofEnv(sides=np.array([side_x, side_y, H]), bs_position=np.array([[-5, -5, 5]]))
    xu_hat = env.pointing(az, el).get()
    Sigma = 1 ** 2 * np.array([[1 / np.cos(Psi) ** 2, np.sin(Psi)], [np.sin(Psi), 1 / np.cos(Psi) ** 2]])
    samples = np.random.multivariate_normal(xu_hat[0, :2], Sigma, size=1000)
    pos = np.hstack((samples, np.repeat(H, samples.shape[0])[np.newaxis].T))
    try:
        beta = cmn.lin2dB(env.pos2beta(pos).get())
    except AttributeError:
        beta = cmn.lin2dB(env.pos2beta(pos).get())
    print(np.ceil(np.max(beta)), np.floor(np.min(beta)))
    norm_beta = (beta - np.min(beta)) / (np.max(beta) - np.min(beta))

    # Scatter plot position
    sfig, sax = plt.subplots()
    sim = sax.scatter(samples[:, 0], samples[:, 1], c=norm_beta, cmap='Oranges')
    sfig.colorbar(sim, ax=sax, label=r'$\beta$')
    cmn.printplot(sfig, sax, render, 'scatterplot', OUTPUT_DIR, 'scatter plot position',
                  labels=[r'$x$ [m]', r'$y$ [m]'], grid=False)

    # Scatter plot (saving only png)
    sfig, sax = plt.subplots(figsize=(7, 7))
    sax.scatter(samples[:, 0], samples[:, 1], c=norm_beta, cmap='Oranges')
    sax.set_xlim(5, 16.5)
    sax.set_ylim(5, 16.5)
    sax.axis('tight')
    sax.axis('off')
    if render:
        sfig.savefig(path.join(OUTPUT_DIR, 'scatterplot_2.png'), dpi=300, transparent=True)
    else:
        cmn.printplot(sfig, sax, False, grid=False)


    # HEATMAP PLOT
    _, num_points_x, num_points_y = cmn.gridmesh_2d((0, side_x), (0, side_y), dx, dy, H)
    power_pc_grid = np.zeros((num_points_x, num_points_y, len(file_suffixes)))
    # Load data
    for s in range(len(file_suffixes)):
        # Grid heatmap
        try:
            filename = path.join(DATADIR, f'heatmap_delta9_{file_suffixes[s]}.npz')
            data = np.load(filename)
            power_pc_grid[:,:, s] = data['power_pc'].reshape(num_points_x, num_points_y)
        except FileNotFoundError:
            pass

    # Plot definitions
    P_fig, P_ax = plt.subplots(1, len(file_suffixes), figsize=(9, 3))

    P_visual = cmn.watt2dbm(power_pc_grid)
    P_min = max(np.min(P_visual), 12)
    P_max = np.max(P_visual)

    basic_cmap = mpl.colormaps['Oranges']
    for s in range(len(file_suffixes)):
        tmp = P_ax[s].imshow(P_visual[:,:, s], cmap=basic_cmap, vmin=P_min, vmax=P_max, extent=[0, 15, 0, 15],
                       interpolation='nearest', origin='lower', aspect='auto')
        P_ax[s].axis('scaled')
    P_fig.colorbar(tmp, ax=P_ax, label=r'$P$ [dBm]')


    cmn.printplot(P_fig, P_ax, render,
                  f'heatmap_power_consumption', OUTPUT_DIR, orientation='horizontal',
                  title=[f'Power needed' + f'{tit}' for tit in title_suffixes],
                  labels=[r'$x$ [m]', r'$y$ [m]'], grid=False)


    # Var Theta plot
    ## Generate axis data
    diagonal = np.sqrt(side_x ** 2 + side_y ** 2)
    dd = 0.5
    points_d = np.arange(0, diagonal, dd)
    num_points = len(points_d)
    azimuth = np.deg2rad(45)
    elevation = np.arccos(H / np.sqrt(H ** 2 + points_d ** 2))

    # Load values
    power_pc = np.zeros((num_points, 3))
    power_avg = np.zeros((num_points, 3))
    power_opt = np.zeros((num_points, 3))
    errorp_pc = np.zeros((num_points, 3))
    errorp_avg = np.zeros((num_points, 3))
    errorp_opt = np.zeros((num_points, 3))

    for s in range(len(file_suffixes)):
        try:
            filename = path.join(DATADIR, f'theta_delta9_{file_suffixes[s]}.npz')
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
    elevation = np.rad2deg(elevation)
    colors = ['red', 'blue', 'magenta']
    for s in range(3):
        # Smoothing power data
        p_opt = savgol_filter(cmn.watt2dbm(power_opt[:, s]), 5, 3)
        p_pc = savgol_filter(cmn.watt2dbm(power_pc[:, s]), 11, 3)


        p_ax.plot(elevation, p_opt, label=f'OPT  {Psi_Gau[s]}', marker='o', c=colors[s])
        p_ax.plot(elevation, p_pc, label=f'PC {Psi_Gau[s]}', marker='^', c=colors[s], ls='dashed')


        e_ax.semilogy(elevation, errorp_opt[:, s], label=f'OPT {Psi_Gau[s]}', marker='o', c=colors[s])
        e_ax.semilogy(elevation, errorp_pc[:, s], label=f'PC {Psi_Gau[s]}', marker='^', c=colors[s], ls='dashed')


    cmn.printplot(p_fig, p_ax, render=render,
                  filename=f'linear_power_consumption', dirname=OUTPUT_DIR,
                  title=f'Power needed', labels=[r'$\hat{\theta}$ [deg]', r'$P$ [dBm]'])

    cmn.printplot(e_fig, e_ax, render=render,
                  filename=f'linear_outage', dirname=OUTPUT_DIR,
                  title=f'Outage achieved', labels=[r'$\hat{\theta}$ [deg]', r'$1 - p_s$'])