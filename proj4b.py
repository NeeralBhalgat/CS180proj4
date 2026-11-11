import argparse
import json
from pathlib import Path

from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from harris import get_harris_corners as course_harris_corners

from proj4a import (
    IMAGE_DIR,
    RESULTS_DIR,
    computeH,
    load_correspondences,
    load_image,
    makeMosaic,
    save_image,
)


def get_harris_corners(image, edge_discard=20, threshold_rel=0.01):
    if not 0 <= threshold_rel <= 1:
        raise ValueError("threshold_rel must be between 0 and 1.")
    gray = rgb2gray(image) if image.ndim == 3 else np.asarray(image, dtype=float)
    response, coords = course_harris_corners(gray, edge_discard=edge_discard)
    points = coords[::-1].T.copy()
    scores = response[coords[0], coords[1]]
    cutoff = max(0.0, threshold_rel * response.max())
    valid = scores > cutoff
    return response, points[valid], scores[valid]


def adaptive_non_maximal_suppression(points, scores, num_points=500, c_robust=0.9):
    points = np.asarray(points, dtype=float)
    scores = np.asarray(scores, dtype=float)
    radii = np.full(len(points), np.inf)

    for i in range(len(points)):
        keep = scores[i] < c_robust * scores
        pts = points[keep]

        if len(pts) > 0:
            dx = pts[:, 0] - points[i, 0]
            dy = pts[:, 1] - points[i, 1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            radii[i] = dist.min()

    order = sorted(range(len(points)), key=lambda i: (radii[i], scores[i]), reverse=True)
    selected = np.array(order[:num_points], dtype=int)
    return selected, radii


def save_corner_comparison(image, points, scores, selected, filename):
    strongest = np.argsort(-scores, kind="stable")[:len(selected)]
    panels = [
        (points, f"Harris candidates ({len(points)})"),
        (points[strongest], f"Strongest scores ({len(strongest)})"),
        (points[selected], f"After ANMS ({len(selected)})"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
    for ax, (corners, title) in zip(axes, panels):
        ax.imshow(image)
        ax.scatter(corners[:, 0], corners[:, 1], s=9, facecolors="none",
                   edgecolors="lime", linewidths=0.55)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def extract_feature_descriptors(image, points, patch_size=40,
                                descriptor_size=8, sigma=2.0):
    if (patch_size <= 0 or descriptor_size <= 0 or patch_size % 2
            or patch_size % descriptor_size):
        raise ValueError("patch_size must be positive, even, and divisible by descriptor_size.")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative.")
    gray = rgb2gray(image) if image.ndim == 3 else np.asarray(image, dtype=float)
    descriptors = np.zeros((len(points), descriptor_size, descriptor_size))
    half = patch_size // 2
    step = patch_size // descriptor_size
    start = step // 2
    for i, (x,y) in enumerate(points):
        x = round(x)
        y = round(y)
        x,y = int(x), int(y)

        if not (half <= x <= gray.shape[1] - half
                and half <= y <= gray.shape[0] - half):
            raise ValueError("Feature points must have a full patch inside the image.")
        patch = gray[y - half : y + half, x - half : x + half]
        blur = gaussian_filter(patch, sigma=sigma)
        # pick one value every five pixels for the default 40 -> 8 patch
        for r in range(descriptor_size):
            for c in range(descriptor_size):
                descriptors[i, r, c] = blur[start + r * step, start + c * step]

    return normalize_descriptors(descriptors)

def normalize_descriptors(descriptors, epsilon=1e-8):
    out = np.zeros_like(descriptors, dtype=float)
    for i, patch in enumerate(descriptors):
        m = patch.mean()
        s = patch.std()
        if s < epsilon:
            s = epsilon
        out[i] = (patch - m) / s

    return out


def save_descriptor_examples(image, points, descriptors, filename,
                             num_examples=8):
    count = min(num_examples, len(points))
    if count == 0:
        return

    indices = np.linspace(0, len(points) - 1, count, dtype=int)
    fig, axes = plt.subplots(
        2, count, figsize=(2 * count, 4),
        squeeze=False, constrained_layout=True,
    )

    for column, i in enumerate(indices):
        x, y = np.round(points[i]).astype(int)
        patch = image[y - 20:y + 20, x - 20:x + 20]

        axes[0, column].imshow(patch, cmap="gray")
        axes[0, column].set_title(f"Corner {i}: 40×40")

        axes[1, column].imshow(
            descriptors[i], cmap="gray", interpolation="nearest",
        )
        axes[1, column].set_title("Normalized 8×8")

        axes[0, column].axis("off")
        axes[1, column].axis("off")

    fig.savefig(filename, dpi=160)
    plt.close(fig)


def match_features(descriptors1, descriptors2, ratio_threshold=0.8):
    matches = []
    if len(descriptors2) < 2:
        return np.empty((0, 2), dtype=int)
    for i in range(len(descriptors1)):
        dist = []
        for j in range(len(descriptors2)):
            d = descriptors1[i] - descriptors2[j]
            dist.append(np.sum(d ** 2))
        order = np.argsort(dist)
        best = order[0]
        second = order[1]
        if dist[best] < ratio_threshold * dist[second]:
            matches.append([i, best])
    return np.array(matches, dtype=int).reshape(len(matches), 2)


def save_feature_matches(image1, image2, points1, points2, matches,
                         filename, inliers=None):
    height = max(image1.shape[0], image2.shape[0])
    width1 = image1.shape[1]
    canvas = np.zeros((height, width1 + image2.shape[1], 3))
    canvas[:image1.shape[0], :width1] = image1
    canvas[:image2.shape[0], width1:] = image2
    fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=True)
    ax.imshow(canvas)
    for k, (i, j) in enumerate(matches):
        x1, y1 = points1[i]
        x2, y2 = points2[j]
        color = plt.cm.hsv((k * 0.61803398875) % 1)
        if inliers is not None:
            color = "lime" if inliers[k] else "red"
        ax.plot([x1, x2 + width1], [y1, y2], color=color,
                 linewidth=0.7, alpha=0.65)
        ax.scatter([x1, x2 + width1], [y1, y2], color=[color], s=9)
    title = f"{len(matches)} candidate matches (before RANSAC)"
    if inliers is not None:
        title = f"RANSAC: {int(inliers.sum())}/{len(matches)} inliers (green); rejected matches (red)"
    ax.set_title(title)
    ax.axis("off")
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def project_points(points, H):
    pts3 = np.column_stack((points, np.ones(len(points))))
    result = pts3 @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return result[:, :2] / result[:, 2:3]


def ransac_homography(points1, points2, num_iterations=2000,
                      inlier_threshold=3.0, seed=0):
    points1 = np.asarray(points1, dtype=float)
    points2 = np.asarray(points2, dtype=float)
    if (points1.ndim != 2 or points1.shape[1] != 2
            or points2.shape != points1.shape
            or not np.isfinite(points1).all() or not np.isfinite(points2).all()):
        raise ValueError("Matched points must be finite arrays of equal shape (N, 2).")
    if num_iterations <= 0 or inlier_threshold <= 0:
        raise ValueError("Iterations and inlier threshold must be positive.")
    if len(points1) < 4:
        raise ValueError("RANSAC needs at least four matched points.")
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(len(points1), dtype=bool)
    errbest = np.inf

    for _ in range(num_iterations):
        sample = rng.choice(len(points1), size=4, replace=False)
        source = points1[sample]
        target = points2[sample]
        # Every triple must be noncollinear for a four-point homography.
        degenerate = False
        for triple in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            for points in (source, target):
                if np.linalg.matrix_rank(np.column_stack((points[list(triple)], np.ones(3)))) < 3:
                    degenerate = True
        if degenerate:
            continue
        try:
            H = computeH(source, target)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
            continue
        predicted = project_points(points1, H)
        errors = np.sqrt(np.sum((predicted - points2) ** 2, axis=1))
        inliers = errors < inlier_threshold
        count = inliers.sum()
        error = errors[inliers].sum()
        if count > best_inliers.sum() or (count == best_inliers.sum() and error < errbest):
            best_inliers = inliers
            errbest = error

    if best_inliers.sum() < 4:
        raise ValueError("RANSAC could not find four consistent matches.")
    best_H = None
    for _ in range(3):
        H = computeH(points1[best_inliers], points2[best_inliers])
        errors = np.sqrt(np.sum((project_points(points1, H) - points2) ** 2, axis=1))
        inliers = errors < inlier_threshold
        if not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3 or inliers.sum() < 4:
            break
        best_H = H
        final_inliers = inliers
        if np.array_equal(inliers, best_inliers):
            break
        best_inliers = inliers
    if best_H is None:
        raise ValueError("RANSAC refinement could not produce a valid homography.")
    return best_H, final_inliers


def stitch_automatically(image1, image2, num_points=500,
                         ratio_threshold=0.8, inlier_threshold=3.0):
    _, points1, scores1 = get_harris_corners(image1)
    _, points2, scores2 = get_harris_corners(image2)
    selected1, _ = adaptive_non_maximal_suppression(points1, scores1, num_points)
    selected2, _ = adaptive_non_maximal_suppression(points2, scores2, num_points)
    points1, points2 = points1[selected1], points2[selected2]
    descriptors1 = extract_feature_descriptors(image1, points1)
    descriptors2 = extract_feature_descriptors(image2, points2)
    matches = match_features(descriptors1, descriptors2, ratio_threshold)
    H, inliers = ransac_homography(points1[matches[:, 0]], points2[matches[:, 1]],
                                   inlier_threshold=inlier_threshold)
    mosaic = makeMosaic(image1, image2, H)
    return mosaic, H, points1, points2, matches, inliers


def save_mosaic_comparison(manual_mosaic, automatic_mosaic, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    for ax, image, title in zip(axes, [manual_mosaic, automatic_mosaic],
                                ["Manual correspondences", "Automatic matches + RANSAC"]):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def run_b1(image_filename, num_points=500):
    image = load_image(image_filename)
    _, points, scores = get_harris_corners(image)
    selected, radii = adaptive_non_maximal_suppression(points, scores, num_points)
    out = RESULTS_DIR / "part_b"
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_filename).stem
    save_corner_comparison(image, points, scores, selected,
                           out / f"{stem}_corners.png")
    np.savez(out / f"{stem}_corners.npz", points=points, scores=scores,
             radii=radii, selected_indices=selected, selected_points=points[selected])
    print(f"{image_filename}: {len(points)} Harris candidates -> {len(selected)} ANMS points")
    return points, scores, selected, radii


def run_b2(image_filename, num_points=500, num_examples=8):
    image = load_image(image_filename)
    _, points, scores = get_harris_corners(image)
    selected, _ = adaptive_non_maximal_suppression(points, scores, num_points)
    points = points[selected]

    descriptors = extract_feature_descriptors(image, points)

    out = RESULTS_DIR / "part_b"
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_filename).stem

    save_descriptor_examples(
        image, points, descriptors,
        out / f"{stem}_descriptors.png",
        num_examples=num_examples,
    )

    np.savez(
        out / f"{stem}_descriptors.npz",
        points=points, descriptors=descriptors,
    )

    return points, descriptors


def run_b3(image1_filename, image2_filename, num_points=500, ratio_threshold=0.8):
    image1 = load_image(image1_filename)
    image2 = load_image(image2_filename)
    points1, descriptors1 = run_b2(image1_filename, num_points)
    points2, descriptors2 = run_b2(image2_filename, num_points)
    matches = match_features(descriptors1, descriptors2, ratio_threshold)
    out = RESULTS_DIR / "part_b"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(image1_filename).stem}_{Path(image2_filename).stem}"
    save_feature_matches(image1, image2, points1, points2, matches,
                         out / f"{stem}_matches.png")
    np.savez(out / f"{stem}_matches.npz", points1=points1,
             points2=points2, matches=matches, ratio_threshold=ratio_threshold)
    print(f"{stem}: {len(matches)} candidate matches at squared-distance ratio < {ratio_threshold}")
    return points1, points2, matches


def run_b4(image1_filename, image2_filename, correspondence_filename,
           num_points=500):
    image1 = load_image(image1_filename)
    image2 = load_image(image2_filename)
    mosaic, H, points1, points2, matches, inliers = stitch_automatically(
        image1, image2, num_points)
    scene = Path(image1_filename).stem.rsplit("_", 1)[0]
    out = RESULTS_DIR / "part_b"
    out.mkdir(parents=True, exist_ok=True)
    save_image(mosaic, f"part_b/mosaic_{scene}_automatic.png")

    manual_points1, manual_points2 = load_correspondences(correspondence_filename)
    manual_H = computeH(manual_points1, manual_points2)
    manual_path = RESULTS_DIR / f"mosaic_{scene}.png"
    if manual_path.exists():
        manual_mosaic = np.asarray(Image.open(manual_path).convert("RGB"), dtype=float) / 255
    else:
        manual_mosaic = makeMosaic(image1, image2, manual_H)
        save_image(manual_mosaic, f"mosaic_{scene}.png")
    save_mosaic_comparison(manual_mosaic, mosaic,
                           out / f"{scene}_mosaic_comparison.png")
    save_feature_matches(image1, image2, points1, points2, matches,
                         out / f"{scene}_ransac_matches.png", inliers)
    matched1 = points1[matches[:, 0]]
    matched2 = points2[matches[:, 1]]
    errors = np.sqrt(np.sum((project_points(matched1, H) - matched2) ** 2, axis=1))
    stats = {"scene": scene, "matches": len(matches), "inliers": int(inliers.sum()),
             "mean_inlier_error": float(errors[inliers].mean()),
             "median_inlier_error": float(np.median(errors[inliers])),
             "manual_points": len(manual_points1), "iterations": 2000,
             "threshold_pixels": 3.0, "seed": 0,
             "H": H.tolist(), "manual_H": manual_H.tolist()}
    (out / f"{scene}_ransac.json").write_text(json.dumps(stats, indent=2) + "\n")
    np.savez(out / f"{scene}_ransac.npz", H=H, points1=points1,
             points2=points2, matches=matches, inliers=inliers, errors=errors)
    print(f"{scene}: {inliers.sum()}/{len(matches)} inliers; mean error {errors[inliers].mean():.2f} px", flush=True)
    return H, inliers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["b1", "b2", "b3", "b4", "all"], default="all")
    args = parser.parse_args()
    for scene in ("posters", "hallway", "bathroom"):
        if args.part in ("b1", "all"):
            for view in (1, 2):
                run_b1(f"{scene}_{view}.jpg")
        if args.part in ("b2", "all"):
            for view in (1, 2):
                run_b2(f"{scene}_{view}.jpg")
        if args.part in ("b3", "all"):
            run_b3(f"{scene}_1.jpg", f"{scene}_2.jpg")
        if args.part in ("b4", "all"):
            correspondence = "poster_correspondences.json" if scene == "posters" else f"{scene}_correspondences.json"
            run_b4(f"{scene}_1.jpg", f"{scene}_2.jpg", correspondence)


if __name__ == "__main__":
    main()
