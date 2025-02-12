import sys

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import optimize
from scipy.ndimage import rotate


def detect_regimes_wavelet(data, wavelet="mexh", max_level=12, threshold=1000):
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
    """Convert 1D index to 2D coordinates after rotation"""
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
    """
    Fit a circle to a set of 2D points using least squares optimization.
    Returns the center coordinates (x, y) and radius r.
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
    """Create a circular mask for the image"""
    Y, X = np.ogrid[: image_shape[0], : image_shape[1]]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return dist_from_center <= radius


def main():
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
