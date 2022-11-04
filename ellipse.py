try:
    import cupy as np
except ImportError:
    import numpy as np

class Ellipse:
    coeff = np.zeros(6)             # Ellipse equation coefficients

    # Find the coordinates of the center
    def center(self) -> np.ndarray:
        delta = self.coeff[1] ** 2 - 4 * self.coeff[0] * self.coeff[2]
        x = (2 * self.coeff[2] * self.coeff[3] - self.coeff[1] * self.coeff[4]) / delta
        y = (2 * self.coeff[0] * self.coeff[4] - self.coeff[1] * self.coeff[3]) / delta
        return np.hstack((x, y))

    # Find the axes of the ellipse
    def axes(self) -> np.ndarray:
        delta = self.coeff[1] ** 2 - 4 * self.coeff[0] * self.coeff[2]
        mult = 2 * self.coeff[0] * self.coeff[4] ** 2
        mult += 2 * self.coeff[2] * self.coeff[3] ** 2
        mult -= 2 * self.coeff[1] * self.coeff[3] * self.coeff[4]
        mult += 2 * delta * self.coeff[5]

        diff = np.sqrt((self.coeff[0] - self.coeff[2]) ** 2 + self.coeff[1] ** 2)

        ax1 = -np.sqrt(mult * (self.coeff[0] + self.coeff[2] + diff)) / delta
        ax2 = -np.sqrt(mult * (self.coeff[0] + self.coeff[2] - diff)) / delta

        return np.hstack((ax1, ax2))

    def angle(self):
        if self.coeff[1] != 0:
            return np.arctan((self.coeff[2] - self.coeff[0] - np.sqrt((self.coeff[0] - self.coeff[2]) ** 2 + self.coeff[1] ** 2)) / self.coeff[1])
        elif self.coeff[0] < self.coeff[2]:
            return 0
        else:
            return np.pi / 2


    # Return ellipse parameters
    def params(self):
        return self.coeff

    # Return the fraction of points inside the ellipse
    def inside_probability(self, points):
        # Algorithm idea
        # prob = 0
        # for point in points:
        #     if self.equation(point) <= 0:
        #         prob += 1
        # prob /= len(points)
        # Numpy acceleration
        prob = np.sum(self.eq_tensor(points) <= np.zeros(points.shape[0])) / points.shape[0]
        return prob

    def eq_tensor(self, points):
        value = self.coeff[0] * points.T[0] ** 2
        value += self.coeff[1] * points.T[0] * points.T[1]
        value += self.coeff[2] * points.T[1] ** 2
        value += self.coeff[3] * points.T[0]
        value += self.coeff[4] * points.T[1]
        value += self.coeff[5]
        return value

    # Return the value of the ellipse equation in a given point
    def equation(self, point):
        value = self.coeff[0] * point[0] ** 2
        value += self.coeff[1] * point[0] * point[1]
        value += self.coeff[2] * point[1] ** 2
        value += self.coeff[3] * point[0]
        value += self.coeff[4] * point[1]
        value += self.coeff[5]
        return value

    # Initialize the ellipse
    def __init__(self, params):
        if params[0] < 0:
            params = -params
        self.coeff = params
