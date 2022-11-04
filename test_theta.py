try:
    import cupy as np
except ImportError:
    import numpy as np
from os import path

from scipy.stats import rice

import scenario.common as cmn
from environment import RISRoofEnv, command_parser

# rendering
prefix = 'theta_'
save_every = 10
# Suffix for title and filenames
file_suffixes = ['0deg', '45deg', '-45deg']
title_suffixes = [r' $\Psi$ 0', r' $\Psi$ 45', r' $\Psi$ -45']

# bs positioning
bs_pos = np.array([[-5, -5, 5]])

# Position uncertainty values
positioning_std = 0.3

if __name__ == '__main__':
    # The following parser is used to impose some data without the need of changing the script (run with -h flag for help)
    # Render bool needs gto be True to save the data
    # If no arguments are given the standard value are loaded (see environment)
    render, side_x, h, name, dirname = command_parser()
    side_y = side_x
    prefix = prefix + name

    # check pathloss computation method
    # This is done if instead of taking \max \beta in eq. (16) you want to use another heuristic value as
    # the \beta(x_c) for the 'ellicenter' kind
    # the \beta(xu_hat) for 'x_hat' kind
    if name in ['minbeta', 'xhat', 'ellicenter']:
        kind = name
    else:
        kind = 'minbeta'

    # Number of points of estimated position for evaluating the power needed
    diagonal = np.sqrt(side_x ** 2 + side_y ** 2)
    dd = 0.5
    points_d = np.arange(0, diagonal, dd)
    num_points = len(points_d)
    azimuth = np.deg2rad(45)
    elevation = np.arccos(h / np.sqrt(h ** 2 + points_d ** 2))

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
            power_pc = np.zeros(num_points)
            power_opt = np.zeros(num_points)
            errorp_pc = np.zeros(num_points)
            errorp_opt = np.zeros(num_points)

        # Generate samples for benchmark evaluation to speed up the procedure
        # position
        samples_x = np.random.multivariate_normal(np.zeros(2), Sigma, int(100 / (1 - env.reliability)))
        # rice
        rice_shape = cmn.db2lin(env.rice_shape)
        try:
            nu = np.sqrt(rice_shape / (1 + rice_shape)).get()
            sigma = np.sqrt(1 / 2 / (rice_shape + 1)).get()
        except AttributeError:
            nu = np.sqrt(rice_shape / (1 + rice_shape))
            sigma = np.sqrt(1 / 2 / (rice_shape + 1))
        # distribution and sampling
        g = rice.rvs(nu / sigma, scale=sigma, size=int(100 /(1 - env.reliability))) * rice.rvs(nu / sigma, scale=sigma, size=int(100 / (1 - env.reliability)))
        # Generate the value of interest
        samples_g = np.abs(np.array(g)) ** 2
        del g, nu, sigma, rice_shape

        # Announcing start
        print(f'Power control \Psi {s+1}/{len(Sigma_list)} START')
        for i, el in enumerate(cmn.std_progressbar(elevation[start:]), start):
            # load configuration pointing to the UE estimated position
            xu_hat, _ = env.load_conf(azimuth, el)
            # Compute power
            power_pc[i] = env.power_control(xu_hat, Sigma, kind=kind)
            # Compute oracle power and actual error probability
            power_opt[i], errorp_opt[i], errorp_pc[i] = env.benchmark(power_pc[i], xu_hat, Sigma, samples_x=samples_x, samples_g=samples_g)
            if render:  # saving data every save_every simulation
                if np.mod(i, save_every) == 0:
                    np.savez(filename, last=i, power_pc=power_pc, power_opt=power_opt, errorp_pc=errorp_pc, errorp_opt=errorp_opt )
        print('\t...DONE')
        if render: # Last save
            np.savez(filename, last=num_points-1, power_pc=power_pc, power_opt=power_opt, errorp_pc=errorp_pc, errorp_opt=errorp_opt)

