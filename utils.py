import numpy as np
import pywt
from scipy import optimize


def detect_regimes(data, wavelet="mexh", max_level=12, threshold=1000):
    """Detects regime changes in 1D signal using wavelet transform.

    Args:
        data (numpy.ndarray): 1D input signal array.
        wavelet (str, optional): Wavelet type to use. Defaults to "mexh".
        max_level (int, optional): Maximum decomposition level. Defaults to 12.
        threshold (float, optional): Minimum amplitude threshold for peaks. Defaults to 1000.

    Returns:
        tuple: A pair of indices (first_change, last_change) representing the
            boundaries of the detected regime.

    Raises:
        AttributeError: If the specified wavelet is not directly available and needs
            to be created as a ContinuousWavelet object.
    """
    scales = np.arange(1, max_level + 1)

    try:
        coeffs, _ = pywt.cwt(data, scales, wavelet)
    except AttributeError:
        wavelet = pywt.ContinuousWavelet(wavelet)  # pyright: ignore
        coeffs, _ = pywt.cwt(data, scales, wavelet)

    # Calculate energy
    energy = np.sum(np.abs(coeffs) ** 2, axis=0)

    # Find local maxima in energy
    peaks = np.where((energy[1:-1] > energy[:-2]) & (energy[1:-1] > energy[2:]))[0] + 1
    peaks = peaks[data[peaks] > threshold]

    return peaks[1], peaks[-2]


def get_rotated_coords(y, center, angle_deg):
    """Converts 1D index to 2D coordinates after rotation.

    Args:
        y (int): Y-coordinate before rotation.
        center (int): Center point of rotation.
        angle_deg (float): Rotation angle in degrees.

    Returns:
        tuple: A pair of integers (x, y) representing the rotated coordinates.
    """
    # Convert angle to radians
    angle = np.radians(angle_deg)

    # Calculate coordinates relative to center
    x_center = 0  # Since we're always using the center column
    y_center = y - center

    # Rotate coordinates
    x_rot = x_center * np.cos(angle) - y_center * np.sin(angle)
    y_rot = x_center * np.sin(angle) + y_center * np.cos(angle)

    # Translate back and round to nearest integer
    x = int(round(x_rot + center))
    y = int(round(y_rot + center))

    return x, y


def fit_circle(points):
    """Fits a circle to a set of 2D points using least squares optimization.

    Args:
        points (list): List of (x, y) coordinate pairs.

    Returns:
        tuple: Three floats (xc, yc, R) representing the center coordinates
            and radius of the fitted circle.
    """

    def calc_R(xc, yc):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    x, y = np.array(points).T
    center_estimate = np.mean(x), np.mean(y)
    center, _ = optimize.leastsq(f_2, center_estimate)

    xc, yc = center
    R = calc_R(xc, yc).mean()
    return xc, yc, R


def create_circular_mask(image_shape, center, radius):
    """Creates a boolean mask for circular region in an image.

    Args:
        image_shape (tuple): Shape of the image (height, width).
        center (tuple): (x, y) coordinates of circle center.
        radius (float): Radius of the circle.

    Returns:
        numpy.ndarray: Boolean array where True indicates points inside the circle.
    """
    Y, X = np.ogrid[: image_shape[0], : image_shape[1]]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return dist_from_center <= radius
