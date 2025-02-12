import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm

from utils import create_circular_mask, detect_regimes, fit_circle, get_rotated_coords


def process_slice(slice_data):
    """Processes a 2D core slice to detect and create a circular mask.

    This function performs the following steps:
    1. Rotates the image at different angles (0-180°)
    2. Detects regime changes along the center line for each rotation
    3. Collects boundary points from the regime changes
    4. Fits a circle to the collected boundary points
    5. Creates a circular mask based on the fitted circle

    Args:
        slice_data (np.ndarray): 2D numpy array representing a single core slice image.
            Expected to be a grayscale image with numeric values.

    Returns:
        np.ndarray: A 2D boolean array of the same shape as input, where True values
            represent the circular mask. Returns an array of zeros if no circle
            could be detected.

    Note:
        The function assumes the core is roughly centered in the image and uses
        5-degree rotation steps for boundary detection.
    """
    image = slice_data.astype(np.float32)
    center = image.shape[0] // 2
    points = []

    # Rotate image and find sign changes
    for angle in range(0, 180, 5):
        rotated = rotate(image, angle, reshape=False)
        cord = rotated[:, center]
        try:
            first, last = detect_regimes(cord)
            rotated_first = get_rotated_coords(first, center, angle)
            rotated_last = get_rotated_coords(last, center, angle)
            points.extend((rotated_first, rotated_last))
        except (IndexError, ValueError):
            continue

    if not points:
        return np.zeros_like(image, dtype=bool)

    # Fit circle and create mask
    xc, yc, R = fit_circle(points)
    return create_circular_mask(image.shape, (xc, yc), R)


def main():
    """Processes core image data to create circular masks for each slice.

    This script reads a binary file containing 3D core image data, processes each
    slice to detect and create circular masks, and saves the results. The process
    includes:
    1. Loading the binary file as a 3D numpy array
    2. Processing each slice to create circular masks
    3. Saving the resulting masks as a binary file

    The input file is expected to be a raw binary file containing int16 values,
    which will be reshaped to (N, 512, 512) dimensions.

    Command-line Arguments:
        --input: Path to the input binary file containing core image data

    Output:
        Creates a new binary file with '_mask' suffix containing the generated
        masks as uint8 values.
    """
    parser = argparse.ArgumentParser(
        description="Plot results from a binary file with circular mask."
    )
    parser.add_argument("--input", type=Path, help="Path to the input binary file")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        core = np.fromfile(f, dtype=np.int16)
        core = core.reshape((core.size // 512**2, 512, 512))

    masks = np.zeros_like(core, dtype=np.uint8)

    print("Processing slices...")
    for i in tqdm(range(core.shape[0])):
        masks[i] = process_slice(core[i])

    masks.tofile(f"{args.input.stem}_mask.raw")


if __name__ == "__main__":
    main()
