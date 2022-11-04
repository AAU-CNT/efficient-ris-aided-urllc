try:
    import cupy as np
except ImportError:
    import numpy as np
from os import path

import scenario.common as cmn
from environment import RISRoofEnv, command_parser

# rendering
prefix = 'heatmap_'
dirname = path.join(path.dirname(__file__), 'data')
save_every = 10
# Suffix for title and filenames
file_suffixes = ['0deg', '45deg', '-45deg']
title_suffixes = [r' $\Psi$ 0°', r' $\Psi$ 45°', r' $\Psi$ -45°']

# For grid mesh
dx = 1
dy = 1

# bs positioning
bs_pos = np.array([[-5, -5, 5]])

# Position uncertainty values
positioning_std = 0.3

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # The following parser is used to impose some data without the need of changing the script (run with -h flag for help)
    # Render bool needs gto be True to save the data
    # If no arguments are given the standard value are loaded (see environment)
    render, side_x, h, name, dirname = command_parser()
    side_y = side_x
    prefix = prefix + name

    # Grid mesh of estimated position for evaluating the power needed
    limits_x = (0, side_x)
    limits_y = (0, side_y)
    points, num_points_x, num_points_y = cmn.gridmesh_2d(limits_x, limits_y, dx, dy, h)

    # Statistic of the position uncertainty (follow eq. (26) in the paper)
    Psi_Gau = [0, np.deg2rad(45), np.deg2rad(-45)]
    Sigma_list = [positioning_std ** 2 * np.array([[1 / np.cos(Psi) ** 2, np.sin(Psi)], [np.sin(Psi), 1 / np.cos(Psi) ** 2]]) for Psi in Psi_Gau]

    # Build environment
    env = RISRoofEnv(sides=np.array([side_x, side_y, h]), bs_position=bs_pos, delta_fraction=0.9)
    # Delta fraction is used to compute \delta and \varepsilon from the reliability p_s as
    # \delta = delta_fraction * (1 - p_s)
    # \varepsilon = (1 - delta_fraction) * (1 - p_s)

    # Looping through the covariance matrices
    for s, Sigma in enumerate(Sigma_list):
        if render:
            # Load previous data if they exist
            filename = path.join(dirname, f'{prefix}{file_suffixes[s]}.npz')
        else:
            filename = ''
        try:
            data = np.load(filename)
            start = int(data['last'] + 1)
            power_pc = data['power_pc']
            power_opt = data['power_opt']
            errorp_pc = data['errorp_pc']
            errorp_opt = data['errorp_opt']
        except FileNotFoundError:
            start = 0
            power_pc = np.zeros(len(points))
            power_opt = np.zeros(len(points))
            errorp_pc = np.zeros(len(points))
            errorp_opt = np.zeros(len(points))

        # Announcing start
        print(f'Power control \Psi {s+1}/{len(Sigma_list)} START')
        for i, xu_hat in enumerate(cmn.std_progressbar(points[start:]), start):
            # Added one dimension for retro-compatibility reason
            xu_hat = xu_hat[np.newaxis]
            # Angles pointing to the user estimated position
            tmp = cmn.cart2spher(xu_hat)
            el = tmp[:, 1]
            az = tmp[:, 2]
            # load configuration pointing to the UE estimated position
            env.load_conf(az, el)
            # Computer power
            power_pc[i] = env.power_control(xu_hat, Sigma)
            if render:  # saving data every save_every simulation
                if np.mod(i, save_every) == 0:
                    np.savez(filename, last=i, power_pc=power_pc, power_opt=power_opt, errorp_pc=errorp_pc, errorp_opt=errorp_opt )
        print('\t...DONE')

        if render: # Last save
            np.savez(filename, last=num_points_x * num_points_y - 1, power_pc=power_pc, power_opt=power_opt, errorp_pc=errorp_pc, errorp_opt=errorp_opt)