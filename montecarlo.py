try:
    import cupy as np
except ImportError:
    import numpy as np


class Montecarlo:
    height = 0              # Room height
    N = 0                   # RIS elements
    loss = 0                # Path loss exponent
    theta = 0               # Elevation angle
    phi = 0                 # Azimuth angle

    # Point the RIS
    def __point(self, x):
        dist = np.sqrt(x[0] ** 2 + x[1] ** 2)
        self.theta = np.arctan(dist / self.height)
        if x[0] != 0:
            self.phi = np.arctan(x[1] / x[0])
        else:
            self.phi = np.pi / 2 * np.sign(x[1])
        if x[0] < 0:
            self.phi += np.pi

    # Define the ellipse projection
    def __project(self, gain, verbose = False):
        # Find the beamwidth associated to the given gain
        return self.env.evaluate_ellipse(self.theta, self.phi, gain)

    def __beam_reliability(self, x, Sigma, gain, path_correction, verbose = False):
        beam = self.__project(gain, verbose)
        # Generate points (Using cupy for speed)
        samples = np.random.multivariate_normal(x, Sigma, size=self.samples_feasible)
        remaining_samples = self.num_samples
        if verbose:
            print('Beam center: ', beam.center())
            print('Beam axes: ', beam.axes())
        if path_correction:     # Not used
            dist = x[0] ** 2 + x[1] ** 2 + self.height ** 2
            # Algorithm idea
            # close = []
            # far = []
            # i = 0
            # for point in samples:
            #     if point[0] ** 2 + point[1] ** 2 + self.height ** 2 > dist:
            #         far.append(i)
            #     else:
            #         close.append(i)
            #     i += 1
            # if len(close) > 0:
            # Numpy acceleration
            far = (samples[:, 0] ** 2 + samples[:, 1] ** 2 + self.env.z_size ** 2) > dist
            close = (samples[:, 0] ** 2 + samples[:, 1] ** 2 + self.env.z_size ** 2) <= dist
            if np.any(close):
                prob_close = beam.inside_probability(samples[close])
            else:
                prob_close = 0
            # if len(far) > 0:
            if np.any(far):
                corr_gain = gain * ((np.sqrt(dist) + beam.axes()[0]) / np.sqrt(dist))  ** self.loss
                corr_beam = self.__project(corr_gain, verbose)
                prob_far = corr_beam.inside_probability(samples[far])
                if verbose:
                    print('Corrected gain: ', corr_gain)
                    print('Corrected beam center: ', beam.center())
                    print('Corrected beam axes: ', beam.axes())
                    print('Close beam probability:', prob_close)
                    print('Far beam probability:', prob_far)
            else:
                prob_far = 0
            return (prob_far * np.sum(far) + prob_close * np.sum(close)) / (np.sum(far) + np.sum(close))

        else:
            prob = 0
            while remaining_samples > 0:
                prob += beam.inside_probability(samples) * samples.shape[0] / self.num_samples
                remaining_samples -= samples.shape[0]
                # If it is the last the following returns an empty array
                samples = np.random.multivariate_normal(x, Sigma, size=min(remaining_samples, self.samples_feasible))
            return prob

    def min_gain(self, x, Sigma, precision, path_correction, verbose = False):
        # Point the beam and iterate over possible gains
        self.__point(x)
        gain_low = 0.1
        gain_high = 1 - precision
        # Low gain test
        prob_low = self.__beam_reliability(x, Sigma, gain_low, path_correction, verbose)
        if  prob_low < (1 - self.epsilon):
            if verbose:
                print('No valid gain, maximum probability: ', prob_low)
            return -1
        # High gain test
        prob_hi = self.__beam_reliability(x, Sigma, gain_high, path_correction, verbose)
        if prob_hi >= (1 - self.epsilon):
            if verbose:
                print('Maximum gain, probability: ', prob_hi)
            return gain_high
        # Line search
        while gain_high - gain_low > precision:
            mid_gain = (gain_high + gain_low) / 2
            prob = self.__beam_reliability(x, Sigma, mid_gain, path_correction, verbose)
            if verbose:
                print('Trying gain: ', mid_gain)
                print('Success probability: ', prob)
            if prob >= (1 - self.epsilon):
                gain_low = mid_gain
            else:
                gain_high = mid_gain
        return gain_low

    # Constructor
    def __init__(self, env):
        self.env = env
        self.height = env.z_size
        self.N = env.ris.num_els
        self.loss = env.pl_exponent
        self.epsilon = env.epsilon
        # Check if the points are too much for a single computation (RAM or GPU issue)
        self.num_samples = int(100 / self.epsilon)
        if self.num_samples > 1e7:
            self.samples_feasible = int(1e7) # int(min(self.num_samples / 100, 1e7))
        else:
            self.samples_feasible = self.num_samples
