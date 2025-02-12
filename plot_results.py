import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate

from utils import create_circular_mask, detect_regimes, fit_circle, get_rotated_coords


def main():
    parser = argparse.ArgumentParser(
        description="Plot results from a binary file with circular mask."
    )
    parser.add_argument("--input", type=str, help="Path to the input binary file")
    parser.add_argument(
        "--slice-index",
        type=int,
        default=100,
        help="Index of the slice to plot (default: 100)",
    )
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        core = np.fromfile(f, dtype=np.int16)
        core = core.reshape((core.size // 512**2, 512, 512))

    image = core[args.slice_index].astype(np.float32)
    center = image.shape[0] // 2

    # Store image coordinates for sign changes
    points = []

    # Rotate image and find sign changes
    for angle in range(0, 180, 5):
        rotated = rotate(image, angle, reshape=False)
        cord = rotated[:, center]
        first, last = detect_regimes(cord)

        # Convert both points to image coordinates
        rotated_first = get_rotated_coords(first, center, angle)
        rotated_last = get_rotated_coords(last, center, angle)
        points.extend((rotated_first, rotated_last))

    x, y = zip(*points)
    xc, yc, R = fit_circle(points)

    mask = create_circular_mask(image.shape, (xc, yc), R)
    masked_image = np.where(mask, image, np.nan)
    values_inside_circle = image[mask]

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
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

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
