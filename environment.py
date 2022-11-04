import scenario.common as cmn
from scenario.cluster import Cluster

try:
    import cupy as np
except ImportError:
    import numpy as np
import argparse
import montecarlo as mc
from ellipse import Ellipse
from scipy.constants import speed_of_light
from scipy.stats import rice
from os import path

# GLOBAL STANDARD PARAMETERS
OUTPUT_DIR = cmn.standard_output_dir('ris-oneshot-urllc')
DATADIR = path.join(path.dirname(__file__), 'data')
# Set parameters
NUM_EL_X = 10
CARRIER_FREQ = 0.9e9            # [Hz]
BANDWIDTH = 360e3               # [Hz]
PACKET_LENGTH = 256             # [bit]
TOLERABLE_LATENCY = .5e-3       # [s]
TARGET_RELIABILITY = 1 - 1e-5   # reliability of the communication
RICE_SHAPE = 6                  # [dB]
LINESEARCH_PRECISION = 2e-2     # Precision of the line search method
NOISE_POWER = -94               # [dBm]
SIDE = 15                       # [m] side of the room
H = 25.                         # [m] height of the room


# Parser for the test files
def command_parser():
    """Parse command line using arg-parse and get user data to run the render.
        If no argument is given, no data is saved  and the default values are used.
        
        :return: the parsed arguments
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", action="store_true", default=False)
    parser.add_argument("-D", type=float, default=SIDE)
    parser.add_argument("-H", type=float, default=H)
    parser.add_argument("-f", "--filename", default='')
    parser.add_argument("-d", "--directory", default=DATADIR)
    args: dict = vars(parser.parse_args())
    return list(args.values())


## Classes
class RISRoofEnv(Cluster):
    """General environment class for the setting at hand"""
    def __init__(self,
                 sides: np.ndarray,
                 bs_position: np.array,
                 ue_position: np.array = np.zeros((1, 3)),
                 ris_num_els: int = NUM_EL_X,
                 carrier_frequency: float = CARRIER_FREQ,
                 bandwidth: float = BANDWIDTH,
                 noise_power: float = NOISE_POWER,
                 packet_length: int = PACKET_LENGTH,
                 max_latency: float = TOLERABLE_LATENCY,
                 reliability: float = TARGET_RELIABILITY,
                 rice_shape: float = RICE_SHAPE,
                 precision: float = LINESEARCH_PRECISION,
                 delta_fraction : float = 0.9,
                 rbs: int = 1,
                 rng: np.random.RandomState = None):
        # Init parent class
        super().__init__(shape='box',
                         sizes=sides,
                         carrier_frequency=carrier_frequency,
                         bandwidth=bandwidth,
                         noise_power=noise_power,
                         direct_channel='LoS',
                         reflective_channel='LoS',
                         rbs=rbs,
                         rng=rng)
        self._int_tested = 25       # attribute related to the number of integers under frequency_scheduling
        try:
            bs_position = np.asarray(bs_position)
            ue_position = np.asarray(ue_position)
        except AttributeError:
            pass
        # Geometry and scenario
        self.place_bs(1, bs_position)
        self.place_ue(ue_position.shape[0], ue_position, gain = 0)
        self.place_ris(1, np.array([[0, 0, 0]]), num_els_x=ris_num_els, dist_els_x=self.wavelength/2, orientation='xy')
        self.compute_distances()
        self.x_hat = None

        # Channel characteristics
        self.rice_shape = rice_shape

        # URLLC data
        self.packet_length = packet_length
        self.max_latency = max_latency
        self.reliability = reliability

        # Errors and precisions
        self.delta = (1 - self.reliability) * delta_fraction
        self.epsilon = (1 - self.reliability) * (1 - delta_fraction)
        self.precision = precision  # line search precision \nu

        # Rice CDF values
        try:
            iCDF, CDF = np.load('data/CDF.npy')
            ind = 0     # only K = 6 is available in the table at the moment
            self.G0 = iCDF[np.nonzero(CDF[:, ind] <= self.delta)[0][-1], ind]
        except FileNotFoundError:
            self.G0 = 1e-6      # Random value below the one for K = 6 and delta 0.5e-5

        # Load AF data for A0 and G(A0) values
        try:
            temp = np.load('data/sinc_argument_precise.npy')
            self.af_argument = temp[0, 1::2]
            self.af_gain = temp[1, 1::2]
        except FileNotFoundError:
            self.af_argument = np.array([1.391])
            self.af_gain = np.array([0.5])


    def load_conf(self, azimuth_angle: float, elevation_angle: float) -> tuple:
        """Load the configuration pointing towards azimuth and elevation given as input when the for the current setting
        (i.e. when the RIS is oriented in the x-y plane).
        
        ---- Inputs:
        :param azimuth_angle: float, azimuth angle \varphi in rad
        :param elevation_angle: float, elevation angle \theta in rad
        ---- Output
        :return: tuple, containing the point on the floor where the RIS is pointing to and
                       the loaded configuration as a vector
        """
        self.x_hat = self.pointing(float(azimuth_angle), float(elevation_angle))
        return self.x_hat, self.ris.load_conf_xy(self.wavenumber, np.array(azimuth_angle), np.array(elevation_angle), self.bs.pos)

    def pointing(self, azimuth_angle: float, elevation_angle: float, k_max=1):
        """Return the point on the floor corresponding to the input azimyth and elevation.

        ---- Inputs:
        :param azimuth_angle: float, azimuth angle \varphi in rad
        :param elevation_angle: float, elevation angle \theta in rad
        :param k_max: int, DEPRECATED used for testing the grating lobes
        ---- Output
        :return: np.ndarray (1,3), corresponding point on the floor of the scenario
        """
        k = np.arange(0, k_max)
        x_pointing = k * 2 * self.wavelength / np.sqrt(self.ris.num_els_h) / self.ris.dist_els_h + np.cos(
            azimuth_angle) * np.sin(elevation_angle)
        y_pointing = k * 2 * self.wavelength / np.sqrt(self.ris.num_els_h) / self.ris.dist_els_h + np.sin(
            azimuth_angle) * np.sin(elevation_angle)
        z_pointing = np.sqrt(1 - x_pointing ** 2 - y_pointing ** 2)
        return self.z_size / z_pointing[:, np.newaxis] * np.array([x_pointing, y_pointing, z_pointing]).T

    ## Power control part!
    def required_power(self, target_snr, min_pl, af_gain, g0):
        """Compute the required power according to Theorem 2

        ---- Inputs:
        :param target_snr: float, minimum SNR satisfying the KPI of the transmission
        :param min_pl: float, value of the minimum path loss gain (linear scale) min \beta
        :param af_gain: float, desired array factor gain A_0
        :param g0: float, inverse of the eCDF giving the desired error probability due to fading
        ---- Output
        :return: float, value of the power obtained
        """
        power = target_snr * self.N0B / self.ris.num_els ** 2 / af_gain / min_pl / g0
        return power

    def pos2beta(self, x):
        """Compute the value of the pathloss (linear scale) given position in space

        ---- Inputs:
        :param x: np.ndarray (K, 3), K position to compute \beta for
        ---- Output
        :return  np.ndarray (K,), value of the path loss gain (linear scale)
        """
        pl = 10 * self.pl_exponent * np.log10(self.dist_br * np.linalg.norm(np.array(x), axis=-1))
        pl += -(self.bs.gain + self.ue.gain)
        pl += - 40 * np.log10(self.wavelength / 4 / np.pi / self.ref_dist)
        pl += - 20 * self.pl_exponent * np.log10(self.ref_dist)
        return 10 ** (-pl / 10)

    def power_control(self, x, Sigma, kind: str = 'minbeta'):
        """Algorithm implementing the proposed power control scheme.

        ---- Inputs:
        :param x: np.ndarray (1,3), estimated position of the user,
        :param Sigma: np.ndarray (3,3), covariance matrix of the Gaussian position uncertainty
        :param kind: str, way of computing the path loss in the region. Default is min \beta as in eq. (16)
        ---- Output
        :return: float, obtained power
        """
        # Minimum SNR to satisfy the URLLC requirements
        gamma_min = 2 ** (self.packet_length / (self.BW * self.max_latency)) - 1

        # Compute reliable gain (considering position uncertainty)
        montecarlo = mc.Montecarlo(self)
        # Move into 2d only
        x2d = x[0, :2]
        reliable_af_gain = montecarlo.min_gain(x2d, Sigma, self.precision, False)

        # Re-find position
        elevation = np.array(cmn.cart2spher(x)[:, 1])
        azimuth = np.array(cmn.cart2spher(x)[:, 2])
        #
        if kind == 'xhat':
            pos = x
        else:
            pos = self.afgain2pos(elevation, azimuth, reliable_af_gain, kind)
        # compute power
        power_pc = self.required_power(gamma_min, self.pos2beta(pos), reliable_af_gain, self.G0)
        return power_pc

    def benchmark(self, power_pc, x, Sigma, samples_x = None, samples_g = None, opt_flag = True):
        """Algorithm implementing the oracle power control scheme.

        ---- Inputs:
        :param power_pc: float, power obtained by the power_control scheme
        :param x: np.ndarray (1,3), estimated position of the user,
        :param Sigma: np.ndarray (3,3), covariance matrix of the Gaussian position uncertainty
        :param samples_x: np.ndarray (2, K) samples of the K realization of the position of the users
        :param samples_g: np.ndarray (K,) samples of the K realization of the fading
        :param opt_flag: bool, if True the optimal power is returned
        ---- Outputs:
        :return: tuple, containing the power of the oracle, its outage probability and the actual outage
                        probability of when employing power_pc
        """
        # Minimum SNR to satisfy the URLLC requirements
        gamma_min = 2 ** (self.packet_length / (self.BW * self.max_latency)) - 1

        # Move into 2d only
        x2d = x[0, :2]
        # Find benchmark values
        if samples_x is None:
            samples_x = np.random.multivariate_normal(x2d, Sigma, int(100 / (1 - self.reliability)))
        else:
            samples_x = x2d + samples_x
        pos_bm = np.hstack((samples_x, np.repeat(self.z_size, samples_x.shape[0])[np.newaxis].T))

        # Pathloss and noise
        gamma_real = self.ris.num_els ** 2 * self.pos2beta(pos_bm) / self.N0B
        # Array factor
        gamma_real *= self.compute_afgain(pos_bm)
        del pos_bm

        # Fading
        # Compute parameters
        if samples_g is None:
            rice_shape = cmn.db2lin(self.rice_shape)
            try:
                nu = np.asnumpy(np.sqrt(rice_shape / (1 + rice_shape)))
                sigma = np.asnumpy(np.sqrt(1 / 2 / (rice_shape + 1)))
            except AttributeError:
                nu = np.sqrt(rice_shape / (1 + rice_shape))
                sigma = np.sqrt(1 / 2 / (rice_shape + 1))
            # distribution and sampling
            gm = rice.rvs(nu / sigma, scale=sigma, size=len(gamma_real))
            gu = rice.rvs(nu / sigma, scale=sigma, size=len(gamma_real))
            # Generate the value of interest
            samples_g = np.abs(np.array(gm) * np.array(gu)) ** 2
            del gm, gu, nu, sigma, rice_shape

        # Insert fading
        gamma_real *= samples_g
        del samples_g

        # Optimal power
        if opt_flag:
            iCDF, CDF = ecdf(gamma_real)
            power_opt = gamma_min / iCDF[np.nonzero(CDF >= (1 - self.reliability))[0][0]]
            errorp_opt = np.sum(power_opt * gamma_real <= gamma_min) / gamma_real.shape[0]
        else:
            power_opt = 0.
            errorp_opt = 1.

        # Compute probability
        errorp_pc = np.sum(power_pc * gamma_real <= gamma_min) / gamma_real.shape[0]
        return power_opt, errorp_opt, errorp_pc



    def afgain2pos(self, elevation, azimuth, target_af_gain, kind = 'minbeta'):
        """Given the AF gain chosen returns the position of the path loss of interest on the floor.

        ---- Inputs:
        :param elevation: float, elevation angle \theta in rad
        :param azimuth: float, azimuth angle \varphi in rad
        :param target_af_gain: float, chosen target AF gain
        :param kind: str, way of computing the path loss in the region. Default is min \beta as in eq. (16)
        ---- Outputs:
        :return: np.ndarray (1, 3), position on the floor where the path loss chosen should be computed
        """
        proj_elli = self.evaluate_ellipse(elevation, azimuth, target_af_gain)
        proj_center = proj_elli.center()
        if kind == 'ellicenter':
            return np.hstack((proj_center, self.z_size))
        else:
            proj_axis = proj_elli.axes()
            proj_angle = proj_elli.angle()
            return np.hstack((proj_center + proj_axis[0] * np.hstack((np.cos(proj_angle), np.sin(proj_angle))), self.z_size))

    def evaluate_ellipse(self, elevation, azimuth, target_af_gain):
        """Method to project the ellipse on the floor

        ---- Inputs:
        :param elevation: float, elevation angle \theta in rad
        :param azimuth: float, azimuth angle \varphi in rad
        :param target_af_gain: float, chosen target AF gain
        ---- Outputs:
        :return: Ellipse, the ellipse class
        """
        # Find the nearest argument and the shape of the generating ellipse in O2
        argument = self.af_argument[np.argmin(np.abs(self.af_gain - target_af_gain))]
        ULA_DT = 2 * np.arcsin(argument * self.wavelength / np.pi / self.ris.dist_els_h / self.ris.num_els_h)
        Delta_phi = ULA_DT
        Delta_theta = ULA_DT / np.cos(elevation)
        # Axis
        a = np.tan(Delta_theta / 2)
        b = np.tan(Delta_phi / 2)

        # Rotation matrix for ellipse in O1
        try:
            rotation_mat = cmn.euler_rotation_matrix(np.pi / 2, elevation.get(), np.pi / 2 - azimuth.get())
        except AttributeError:
            rotation_mat = cmn.euler_rotation_matrix(np.pi / 2, elevation, np.pi / 2 - azimuth)

        # Project ellipse on the floor in O1
        param = param_evaluation(rotation_mat, a, b, self.z_size)
        return Ellipse(param)

    def compute_afgain(self, x):
        """ Utils function to compute AF gain given a position in space.

        ---- Input:
        :param x: np.ndarray (K, 3), position of the K points to estimate the AF gain for
        ---- Output:
        :return: np.ndarray (K,), computed AF gain
        """
        # Preprocessing
        N = x.shape[0]
        pos_dist = np.linalg.norm(x, axis=-1)
        pos_versor = x / pos_dist[np.newaxis].T
        # Compute the array on a subset of points for RAM reason
        af_gain = np.zeros(N)
        # max test per iteration
        n = int(1e4)
        # iterations
        iter = int(np.floor(N / n))
        # Phase bs ris is always the same
        phase_shift_br = self.freqs[np.newaxis].T * np.tile((self.dist_br - self.bs.pos.cartver @ self.ris.el_pos)[np.newaxis].T, (1, self.RBs, n))

        # Iterating to smaller set of data to avoid RAM or GPU memory limits
        for i in np.arange(iter):
            phase_shift_ru = self.freqs[np.newaxis].T * (pos_dist[i*n:(i+1)*n] - (pos_versor[i*n:(i+1)*n] @ self.ris.el_pos).T)[np.newaxis].reshape((self.ris.num_els, 1, n))
            af_gain[i*n:(i+1)*n] = np.abs(np.sum(self.ris.actual_conf[np.newaxis, np.newaxis].T * np.exp(- 1j * 2 * np.pi / speed_of_light * (phase_shift_ru + phase_shift_br)), axis=0) / self.ris.num_els) ** 2
        # deal with non integer division N / n
        n2 = N - iter * n
        if n2 > 0:
            phase_shift_ru = self.freqs[np.newaxis].T * (pos_dist[iter * n:] - (pos_versor[iter * n:] @ self.ris.el_pos).T)[np.newaxis].reshape((self.ris.num_els, 1, n2))
            phase_shift_br = self.freqs[np.newaxis].T * np.tile((self.dist_br - self.bs.pos.cartver @ self.ris.el_pos)[np.newaxis].T, (1, self.RBs, n2))
            af_gain[iter * n:] = np.abs(np.sum(self.ris.actual_conf[np.newaxis, np.newaxis].T * np.exp(- 1j * 2 * np.pi / speed_of_light * (phase_shift_ru + phase_shift_br)), axis=0) / self.ris.num_els) ** 2
        del phase_shift_ru, phase_shift_br, pos_versor, pos_dist
        return af_gain


def param_evaluation(r: np.ndarray, major_ax: float, minor_axis: float, height: float):
    """  Evaluate the parameters of the ellipse on the floor.

    ---- Input:
    :param r: np.ndarray (3,3), rotation matrix from O1 to O2
    :param major_ax: float, length of the major axis of the elliptic cone at reference distance w = 1
    :param minor_axis: float, length of the minor axis of the elliptic cone at reference distance w = 1
    :param height: float, height of the room
    ---- Output:
    :return: np.ndarray (6,), capital letter parameters of the ellipses (eq. (24))
    """
    assert major_ax >= minor_axis
    r = np.array(r)
    return np.array([(r[0, 0] / major_ax) ** 2 + (r[1, 0] / minor_axis) ** 2 - r[2, 0] ** 2,
                     2 * (r[0, 0] * r[0, 1] / major_ax ** 2 + r[1, 0] * r[1, 1] / minor_axis ** 2 - r[2, 0] * r[2, 1]),
                     (r[0, 1] / major_ax) ** 2 + (r[1, 1] / minor_axis) ** 2 - r[2, 1] ** 2,
                     2 * height * (r[0, 0] * r[0, 2] / major_ax ** 2 + r[1, 0] * r[1, 2] / minor_axis ** 2 - r[2, 0] * r[2, 2]),
                     2 * height * (r[0, 1] * r[0, 2] / major_ax ** 2 + r[1, 1] * r[1, 2] / minor_axis ** 2 - r[2, 1] * r[2, 2]),
                     height ** 2 * ((r[0, 2] / major_ax) ** 2 + (r[1, 2] / minor_axis) ** 2 - r[2, 2] ** 2)])


def ecdf(a):
    """Empirical CDF evaluation of a rv.

    ---- Input:
    :param a: np.ndarray (K,), realization of a rv
    ---- Outputs:
    :return: tuple, collecting the inverse eCDF and the eCDF of the rv
    """
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]
