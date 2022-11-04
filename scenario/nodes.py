#!/usr/bin/env python3
# filename "nodes.py"

try:
    import cupy as np
except ImportError:
    import numpy as np
from scenario.common import Position



class Node:
    """Construct a communication entity."""
    def __init__(self,
                 n: int,
                 pos: np.ndarray,
                 gain: float or np.ndarray = None,
                 max_pow: float or np.ndarray = None):
        """
        Parameters
        ---------
        :param n: int, number of nodes to place
        :param pos: ndarray of shape (n,3), position of the node in rectangular coordinates.
        :param gain : float, antenna gain of the node.
        :param max_pow : float, max power available on transmission in linear scale.
        """

        # Control on INPUT
        if pos.shape != (n, 3):
            raise ValueError(f'Illegal positioning: for Node, pos.shape must be ({n}, 3), instead it is {pos.shape}')

        # Set attributes
        self.n = n
        self.pos = Position(pos)
        self.gain = gain
        self.max_pow = max_pow


class BS(Node):
    """Base station class"""

    def __init__(self,
                 n: int = None,
                 pos: np.ndarray = None,
                 gain: float = None,
                 max_pow: float = None):
        """
        Parameters
        ---------
        :param n: number of BS.
        :param pos: ndarray of shape (n,3), position of the BS in rectangular coordinates.
        :param gain: float, BS antenna gain. Default is 13.85 dB.
        :param max_pow : float, BS max power. Default is 46 dBm.
        """
        if n is None:
            n = 1
        if gain is None:
            gain = 12.85    # [dB]
        if max_pow is None:
            max_pow = 46  # [dBm]

        # Init parent class
        super().__init__(n, pos, gain, max_pow)

    def __repr__(self):
        return f'BS-{self.n}'


class UE(Node):
    """User class
    """

    def __init__(self,
                 n: int,
                 pos: np.ndarray,
                 gain: float = None,
                 max_pow: float = None):
        """
        Parameters
        ---------
        :param n: number of UE.
        :param pos: ndarray of shape (n,3), position of the BS in rectangular coordinates.
        :param gain: float, BS antenna gain. Default is 2.15 dB.
        :param max_pow : float, BS max power. Default is 23 dBm.
        """
        if gain is None:
            gain = 2.15     # [dB]
        if max_pow is None:
            max_pow = 23  # [dBm]

        # Init parent class
        super().__init__(n, pos, gain, max_pow)

    def __repr__(self):
        return f'UE-{self.n}'


class RIS(Node):
    """Reflective Intelligent Surface class"""

    def __init__(self,
                 n: int,
                 pos: np.ndarray,
                 num_els_h: int,
                 dist_els_h: float,
                 num_els_v: int = None,
                 dist_els_v: float = None,
                 orientation: str = None):
        """
        Parameters
        ---------
        :param n: number of RIS to consider # TODO: not all methods works for multi-RIS environment
        :param pos: ndarray of shape (n, 3), position of the RIS in rectangular coordinates.
        :param num_els_v: int, number of elements along z-axis.
        :param num_els_h: int, number of elements along x-axis.
        :param dist_els_v: float, size of each element.
        """
        # Default values
        if num_els_v is None:
            num_els_v = num_els_h
        if dist_els_v is None:
            dist_els_v = dist_els_h
        if orientation is None:
            orientation = 'xz'

        # Initialize the parent, having zero gain, max_pow is -np.inf,
        super().__init__(n, pos, 0.0, -np.inf)

        # Instance variables
        self.num_els = num_els_v * num_els_h  # total number of elements
        self.num_els_h = num_els_h  # horizontal number of elements
        self.num_els_v = num_els_v  # vertical number of elements
        self.dist_els_h = dist_els_h
        self.dist_els_v = dist_els_v

        # Compute RIS sizes
        self.size_h = num_els_h * self.dist_els_h  # horizontal size [m]
        self.size_v = num_els_v * self.dist_els_v  # vertical size [m]
        self.area = self.size_v * self.size_h   # area [m^2]

        # Element positioning
        self.m = np.tile(1 + np.arange(self.num_els_h), (self.num_els_v,))
        self.n = np.repeat(1 + np.arange(self.num_els_v), (self.num_els_h,))
        if orientation == 'xz':
            self.el_pos = np.vstack((self.dist_els_h * (self.m - (self.num_els_h + 1) / 2), np.zeros(self.num_els), self.dist_els_v * (self.n - (self.num_els_v + 1) / 2)))
        elif orientation == 'xy':
            self.el_pos = np.vstack((self.dist_els_h * (self.m - (self.num_els_h + 1) / 2), self.dist_els_v * (self.n - (self.num_els_v + 1) / 2), np.zeros(self.num_els)))
        elif orientation == 'yz':
            self.el_pos = np.vstack((np.zeros(self.num_els), self.dist_els_h * (self.m - (self.num_els_h + 1) / 2), self.dist_els_v * (self.n - (self.num_els_v + 1) / 2)))
        else:
            raise TypeError('Wrong orientation of the RIS')
        self.orientation = orientation

        # Configure RIS
        self.actual_conf = np.ones(self.num_els)    # initialized with attenuation and phase 0
        self.std_configs = None
        self.num_std_configs = None
        self.std_config_angles = None
        self.std_config_limits_plus = None
        self.std_config_limits_minus = None
        self.angular_resolution = None

    def ff_dist(self, wavelength):
        return 2 / wavelength * max(self.size_h, self.size_v) ** 2


    def init_std_configurations(self, wavelength: float):
        """Set configurations offered by the RIS having a coverage of -3dB beamwidth on all direction from 0 to 180 degree

        :returns set_configs : ndarray, discrete set of configurations containing all possible angles (theta_s) in radians in which the RIS can steer the incoming signal.
        """
        a = 1.391
        self.num_std_configs = int(np.ceil(self.num_els_h * self.dist_els_h * np.pi / wavelength / a))
        self.std_configs = 1 - (2 * np.arange(1, self.num_std_configs + 1) - 1) * wavelength * a / self.num_els_h / self.dist_els_h / np.pi
        if np.any(self.std_configs < - 1):
            self.std_configs = self.std_configs[:-1]
            self.num_std_configs -= 1
        self.std_config_limits_plus = 1 - (2 * np.arange(1, self.num_std_configs + 1)) * wavelength * a / self.num_els_h / self.dist_els_h / np.pi
        self.std_config_limits_minus = 1 - (2 * np.arange(1, self.num_std_configs + 1) - 2) * wavelength * a / self.num_els_h / self.dist_els_h / np.pi
        self.std_config_angles = np.arccos(self.std_configs)

    def array_factor(self, pos_transmitters: np.array, pos_receivers: np.array, wavelength: float, ):
        pass

    def set_std_configuration(self, wavenumber, index, bs_pos: Position = Position(np.array([0, 10, 0]))):
        """Create the phase profile from codebook compensating the bs position and assuming attenuation 0"""
        # compensating bs
        phase_bs_h = np.cos(bs_pos.sph[:, 2]) * np.sin(bs_pos.sph[:, 1])
        phase_bs_v = np.cos(bs_pos.sph[:, 1])
        # Set standard configuration
        phase_conf_h = self.std_configs[index]
        phase_conf_v = 0
        # Compensating the residual phase
        phase_conf_tot = (self.num_els_h + 1) / 2 * self.dist_els_h * (phase_conf_h + phase_bs_h) + (self.num_els_v + 1) / 2 * self.dist_els_v * (phase_conf_v + phase_bs_v)
        # Put all together
        self.actual_conf = np.exp(1j * wavenumber * (phase_conf_tot - self.m * self.dist_els_h * (phase_conf_h + phase_bs_h) - self.n * self.dist_els_v * (phase_conf_v + phase_bs_v)))
        return self.actual_conf     #, phase_conf_h + phase_bs_h, phase_conf_v + phase_bs_v

    def set_configuration(self, wavenumber, configuration_angle, bs_pos: Position = Position(np.array([0, 10, 0]))):
        """Create the phase profile from codebook compensating the bs position and assuming attenuation 0"""
        # compensating bs
        phase_bs_h = np.cos(bs_pos.sph[:, 2]) * np.sin(bs_pos.sph[:, 1])
        phase_bs_v = np.cos(bs_pos.sph[:, 1])
        # Set specific configuration
        phase_c_h = np.cos(configuration_angle)
        phase_c_v = 0
        # Put all together
        self.actual_conf = np.exp(1j * wavenumber * (- self.m * self.dist_els_h * (phase_c_h + phase_bs_h) - self.n * self.dist_els_v * (phase_c_v + phase_bs_v)))
        return self.actual_conf

    def load_conf_xy(self, wavenumber, azimuth_angle, elevation_angle, bs_pos: Position = Position(np.array([0, 10, 0]))):
        """Create the phase profile from codebook compensating the bs position and assuming attenuation 0"""
        # compensating bs
        phase_bs_h = np.cos(bs_pos.sph[:, 2]) * np.sin(bs_pos.sph[:, 1])
        phase_bs_v = np.sin(bs_pos.sph[:, 2]) * np.sin(bs_pos.sph[:, 1])
        # Set specific configuration
        phase_c_h = np.cos(azimuth_angle) * np.sin(elevation_angle)
        phase_c_v = np.sin(azimuth_angle) * np.sin(elevation_angle)
        # Put all together
        self.actual_conf = np.exp(1j * wavenumber * (- self.m * self.dist_els_h * (phase_c_h + phase_bs_h) - self.n * self.dist_els_v * (phase_c_v + phase_bs_v)))
        return self.actual_conf

    def __repr__(self):
        return f'RIS-{self.n}'
