try:
    import cupy as np
except ImportError:
    import numpy as np
from os import path

from environment import RISRoofEnv, command_parser
from scenario.common import std_progressbar

# rendering
prefix = 'delta_'
save_every = 1
# Suffix for filenames
file_suffixes = ['0deg', '45deg', '-45deg']

# bs positioning
bs_pos = np.array([[-5, -5, 5]])

# Position uncertainty values
positioning_std = 0.3

if __name__ == '__main__':
    # Render
    render, side_x, h, name, dirname = command_parser()
    side_y = side_x
    prefix = prefix + name

    # check pathloss computation method
    if name in ['minbeta', 'xhat', 'ellicenter']:
        kind = name
    else:
        kind = 'minbeta'

    # Number of points of delta epsilon
    delta_vec = np.arange(0.1, 1, 0.1)
    num_points = len(delta_vec)
    # Keeping the point ixed
    azimuth = np.deg2rad(45)
    elevation = np.deg2rad(30)

    # Statistic of the position uncertainty
    Psi_Gau = [0, np.deg2rad(45), np.deg2rad(-45)]
    Sigma_list = [positioning_std ** 2 * np.array([[1 / np.cos(Psi) ** 2, np.sin(Psi)], [np.sin(Psi), 1 / np.cos(Psi) ** 2]]) for Psi in Psi_Gau]

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

        # Announcing start
        print(f'Power control \Psi {s}/{len(Sigma_list)} START')
        for i, delta in enumerate(std_progressbar(delta_vec[start:]), start):
            # Build environment
            env = RISRoofEnv(sides=np.array([side_x, side_y, h]), bs_position=bs_pos, delta_fraction=delta)
            # load configuration pointing to the UE estimated position
            xu_hat, _ = env.load_conf(azimuth, elevation)
            # Computer power
            power_pc[i] = env.power_control(xu_hat, Sigma, kind=kind)
            power_opt[i], errorp_opt[i], errorp_pc[i] = env.benchmark(power_pc[i], xu_hat, Sigma, opt_flag=False)
            if render:
                if np.mod(i, save_every) == 0:
                    np.savez(filename, last=i, power_pc=power_pc, power_opt=power_opt, errorp_pc=errorp_pc, errorp_opt=errorp_opt )
        print(f'\t...DONE')
        if render:# Last save
            np.savez(filename, last=num_points-1, power_pc=power_pc, power_opt=power_opt, errorp_pc=errorp_pc, errorp_opt=errorp_opt)

