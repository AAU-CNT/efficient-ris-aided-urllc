from copy import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as graphicElli

import scenario.common as cmn
from environment import RISRoofEnv, command_parser, OUTPUT_DIR, H, SIDE

try:
    import cupy as np
except ImportError:
    import numpy as np


def ind2coord(ind, dx: float, xlim: tuple, dy: float, ylim: tuple):
    try:
        return np.asnumpy(ind[0] * dx - xlim[1]), np.asnumpy(ind[1] * dy - ylim[1])
    except AttributeError:
        return ind[0] * dx - xlim[1], ind[1] * dy - xlim[1]

# rendering
render = command_parser()[0]
prefix = ''
# Set parameters
elli_color = 'gray'

# For mesh grid points
dx = 0.1
dy = 0.1

# bs positioning
bs_pos = np.array([[-5, -5, 5]])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Evaluate mesh grid points
    limits_x = (5, 16.5)
    limits_y = (5, 16.5)
    points, num_points_x, num_points_y = cmn.gridmesh_2d(limits_x, limits_y, dx, dy, H)

    # Build environment
    env = RISRoofEnv(sides=np.array([SIDE, SIDE, H]), bs_position=bs_pos, ue_position=points)

    # set configuration
    az = np.deg2rad(45)
    el_vec = np.deg2rad(np.arange(30, 31))
    A0 = np.arange(0.1, 1.0, 0.2)

    for el in el_vec:
        # pointing
        xu_hat, _ = env.load_conf(az, el)

        # Array factor
        A = np.abs(np.squeeze(env.compute_array_factor()).reshape(num_points_x, num_points_y)) ** 2

        # AF + path loss
        h_ris = np.abs(np.squeeze(env.build_ris_channel()).reshape(num_points_x, num_points_y)) ** 2
        h_ris_norm = h_ris / np.max(h_ris)

        # For visualization purpose
        try:
            x_visual = np.asnumpy(xu_hat)
            a_visual = np.asnumpy(A.T)
            h_visual = np.asnumpy(10 * np.log10(h_ris.T))
        except AttributeError:
            x_visual = xu_hat
            a_visual = A.T
            h_visual = 10 * np.log10(h_ris.T)


        # Suffix for title and filenames of plots
        el_deg = int(np.round(np.rad2deg(el)))
        filename_suffix = f'_el_{el_deg}'
        title_suffix = r' $\hat{\theta} = $' + f'{el_deg}°'

        # Plot definitions
        af_fig, af_ax = plt.subplots()
        afq_fig, afq_ax = plt.subplots()
        h_fig, h_ax = plt.subplots()

        for i, factor in enumerate(A0):
            # Project ellipse on the floor in O1
            proj_elli = env.evaluate_ellipse(el, az, factor)
            proj_center = proj_elli.center()
            proj_axis = proj_elli.axes()
            proj_angle = proj_elli.angle()
            max_pl_pos = proj_center + proj_axis[0] * np.array([np.cos(proj_angle), np.sin(proj_angle)])

            # Visualization purpose
            graph_elli = graphicElli(proj_center, 2 * proj_axis[0], 2 * proj_axis[1], np.rad2deg(proj_angle), ec=elli_color, fc='none')
            for ax in (af_ax, afq_ax, h_ax):
                patch = copy(graph_elli)
                ax.add_patch(patch)
                ax.text(*max_pl_pos.get(), f'{factor:.1f}', fontsize='small', color=elli_color)
                ax.scatter(x_visual[:, 0], x_visual[:, 1], marker='.', c='red')

        # Array factor plots
        # Unquantized
        basic_cmap = mpl.colormaps['Oranges']
        af_im = af_ax.imshow(a_visual, cmap=basic_cmap, vmin=0, vmax=1, extent=[*limits_x, *limits_y],
                             interpolation='nearest', origin='lower', aspect='auto')
        af_fig.colorbar(af_im, ax=af_ax, label=r'$|A(\phi)|^2$')

        cmn.printplot(af_fig, af_ax, render, f'{prefix}af_floor{filename_suffix}', OUTPUT_DIR,
                      title=f'Array factor on the floor' + title_suffix,
                      labels=[r'$x$ [m]', r'$y$ [m]'], grid=False)

        # Quantized
        bounds = np.arange(0, 1.1, 0.1).get()
        norm = mpl.colors.BoundaryNorm(bounds, basic_cmap.N)
        smap = mpl.cm.ScalarMappable(norm=norm, cmap=basic_cmap)
        colors = smap.to_rgba(bounds)
        cmap = mpl.colors.ListedColormap(colors)

        afq_ax.imshow(a_visual, cmap=cmap, vmin=0, vmax=1, extent=[*limits_x, *limits_y],
                      interpolation='nearest', origin='lower', aspect='auto')
        afq_fig.colorbar(smap, ax=afq_ax, label=r'$|A(\phi)|^2$')

        cmn.printplot(afq_fig, afq_ax, render, f'{prefix}af_floor_quant{filename_suffix}', OUTPUT_DIR,
                      title=f'Array factor on the floor' + title_suffix,
                      labels=[r'$x$ [m]', r'$y$ [m]'], grid=False)

        # Channel gain plot
        h_im = h_ax.imshow(h_visual, cmap='viridis', extent=[*limits_x, *limits_y],
                           interpolation='nearest', origin='lower', aspect='auto')

        cbar = h_fig.colorbar(h_im, ax=h_ax, label=r'$|\beta A(\phi)|^2$')
        h_file = prefix + f'channel_floor' + filename_suffix
        title = f'Channel gain on the floor' + title_suffix
        cmn.printplot(h_fig, h_ax, render, h_file, OUTPUT_DIR, title=title, labels=[r'$x$ [m]', r'$y$ [m]'], grid=False)
