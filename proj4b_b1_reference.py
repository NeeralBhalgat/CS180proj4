"""Part B.1: single-scale Harris corners and adaptive non-maximal suppression.

Run: python3 proj4b.py
Coordinates returned and saved by this module are (x, y), as in proj4a.py.
Dependencies: numpy, Pillow, scipy, scikit-image, matplotlib.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_peaks

from proj4a import IMAGE_DIR, RESULTS_DIR, load_image


def get_harris_corners(image, edge_discard=20, threshold_rel=0.01):
    """Return response image, local-maximum points (N, 2), and scores (N,).

    Harris uses det(M) - 0.04 * trace(M)**2 at one scale (sigma=1).
    A 20-pixel border reserves space for the later 40x40 descriptors.
    threshold_rel filters weak peaks relative to the maximum response.
    """
    if edge_discard < 0 or not 0 <= threshold_rel <= 1:
        raise ValueError("Require edge_discard >= 0 and 0 <= threshold_rel <= 1.")
    gray = rgb2gray(image) if image.ndim == 3 else np.asarray(image, dtype=float)
    response = corner_harris(gray, method="k", k=0.04, sigma=1)
    # min_distance=1 gives local peak suppression, not the wider ANMS step.
    rows_cols = corner_peaks(
        response, min_distance=1, threshold_abs=0,
        threshold_rel=threshold_rel, exclude_border=False,
    )
    rows, cols = rows_cols.T
    height, width = gray.shape
    valid = ((rows >= edge_discard) & (rows < height - edge_discard)
             & (cols >= edge_discard) & (cols < width - edge_discard))
    rows_cols = rows_cols[valid]
    points = rows_cols[:, ::-1].copy()  # (row, col) -> (x, y)
    scores = response[rows_cols[:, 0], rows_cols[:, 1]]
    return response, points, scores


def adaptive_non_maximal_suppression(points, scores, num_points=500,
                                     c_robust=0.9, batch_size=128):
    """Return selected indices and the suppression radius of EVERY candidate.

    r_i = min_j ||p_i-p_j|| subject to score_i < c_robust * score_j.
    Candidates without a sufficiently stronger neighbor have infinite radius.
    Rank by decreasing radius, breaking ties by decreasing corner score.
    Exact O(N**2) work; O(batch_size*N) temporary memory.
    """
    points = np.asarray(points, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or scores.shape != (len(points),):
        raise ValueError("Expected points with shape (N, 2) and scores with shape (N,).")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(scores)):
        raise ValueError("Points and scores must be finite.")
    if np.any(scores <= 0):
        raise ValueError("ANMS requires positive corner scores.")
    if num_points < 0 or batch_size < 1 or not 0 < c_robust < 1:
        raise ValueError("Require num_points >= 0, batch_size >= 1, and 0 < c_robust < 1.")
    radii_squared = np.full(len(points), np.inf)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        dx = points[start:stop, 0, None] - points[None, :, 0]
        dy = points[start:stop, 1, None] - points[None, :, 1]
        distances_squared = dx * dx + dy * dy
        stronger = scores[start:stop, None] < c_robust * scores[None, :]
        distances_squared[~stronger] = np.inf
        radii_squared[start:stop] = distances_squared.min(axis=1)
    # lexsort uses the last key as primary. Index is the final tie-breaker.
    order = np.lexsort((np.arange(len(points)), -scores, -radii_squared))
    return order[:num_points], np.sqrt(radii_squared)


def save_corner_comparison(image, points, scores, selected, filename):
    """Show all candidates and equal-count score-only / ANMS selections."""
    strongest = np.argsort(-scores, kind="stable")[:len(selected)]
    panels = [
        (points, f"All Harris candidates ({len(points)})"),
        (points[strongest], f"Strongest Harris scores ({len(strongest)})"),
        (points[selected], f"ANMS ({len(selected)}, c = 0.9)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
    for ax, (corners, title) in zip(axes, panels):
        ax.imshow(image)
        ax.scatter(corners[:, 0], corners[:, 1], s=9,
                   facecolors="none", edgecolors="lime", linewidths=0.55)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", default=[
        f"{scene}_{view}.jpg"
        for scene in ("posters", "hallway", "bathroom") for view in (1, 2)
    ], help="Image filenames inside images/.")
    parser.add_argument("--num-points", type=int, default=500)
    parser.add_argument("--threshold-rel", type=float, default=0.01)
    args = parser.parse_args()
    output_dir = RESULTS_DIR / "part_b"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in args.images:
        image = load_image(filename)
        _, points, scores = get_harris_corners(image, threshold_rel=args.threshold_rel)
        selected, radii = adaptive_non_maximal_suppression(
            points, scores, num_points=args.num_points)
        stem = Path(filename).stem
        figure_path = output_dir / f"{stem}_corners.png"
        save_corner_comparison(image, points, scores, selected, figure_path)
        np.savez(output_dir / f"{stem}_corners.npz", points=points,
                 scores=scores, radii=radii, selected_indices=selected,
                 selected_points=points[selected])
        print(f"{filename}: {len(points)} Harris candidates -> {len(selected)} ANMS points")
        print(f"Saved {figure_path.relative_to(IMAGE_DIR.parent)}")


if __name__ == "__main__":
    main()
