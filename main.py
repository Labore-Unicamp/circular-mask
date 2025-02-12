import sys

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import optimize
from scipy.ndimage import rotate


def detect_regimes_wavelet(data, wavelet="mexh", max_level=12, threshold=1000):
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


def main():
    """Main function to process core images and visualize circular masking.

    Reads a binary file containing core image data, processes a single slice
    to detect and fit a circle, creates a mask, and visualizes the results
    using matplotlib.

    Command line arguments:
        sys.argv[1]: Path to the binary file containing core image data.
    """
    with open(sys.argv[1], "rb") as f:
        core = np.fromfile(f, dtype=np.int16)
        core = core.reshape((core.size // 512**2, 512, 512))

    image = core[100].astype(np.float32)
    center = image.shape[0] // 2

    # Store image coordinates for sign changes
    points = []

    # Rotate image and find sign changes
    for angle in range(0, 180, 5):
        rotated = rotate(image, angle, reshape=False)
        cord = rotated[:, center]
        first, last = detect_regimes_wavelet(cord)

        # Convert both points to image coordinates
        rotated_first = get_rotated_coords(first, center, angle)
        rotated_last = get_rotated_coords(last, center, angle)
        points.extend((rotated_first, rotated_last))

    x, y = zip(*points)
    xc, yc, R = fit_circle(points)

    mask = create_circular_mask(image.shape, (xc, yc), R)
    masked_image = np.where(mask, image, np.nan)
    values_inside_circle = image[mask]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    # Original image with circle
    ax1.imshow(image, cmap="gray")
    circle = plt.Circle((xc, yc), R, fill=False, color="blue", label="Fitted Circle")
    ax1.add_patch(circle)
    ax1.scatter(x, y, c="red", s=20)
    ax1.set_title("Original Image with Fitted Circle")

    # Masked image
    ax2.imshow(masked_image, cmap="gray")
    ax2.set_title("Masked Image")

    ax3.hist(values_inside_circle, bins=50, color="blue", alpha=0.7)
    ax3.set_title("Histogram of Values Inside Circle")
    ax3.set_xlabel("Pixel Value")
    ax3.set_ylabel("Frequency")

    plt.show()


if __name__ == "__main__":
    main()
