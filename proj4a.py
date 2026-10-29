import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "images"
RESULTS_DIR = PROJECT_DIR / "results"


def load_image(filename):
    path = IMAGE_DIR / filename
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(image, dtype=float) / 255.0


def save_image(image, filename):
    path = RESULTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = (np.clip(image, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(pixels).save(path)


def load_correspondences(filename="poster_correspondences.json"):
    path = IMAGE_DIR / filename
    with path.open() as file:
        points = json.load(file)
    return (
        np.asarray(points["im1Points"], dtype=float),
        np.asarray(points["im2Points"], dtype=float),
    )


def computeH(im1_pts, im2_pts):
    im1_pts = np.asarray(im1_pts, dtype=float)
    im2_pts = np.asarray(im2_pts, dtype=float)

    if len(im1_pts) < 4 or len(im2_pts) < 4:
        raise ValueError("Need at least four points in each image.")

    if len(im1_pts) != len(im2_pts):
        raise ValueError("BAD DATA")

    A = []
    b = []
    for (x1,y1), (x2,y2) in zip(im1_pts, im2_pts):
        A.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2])
        b.append(x2)

        A.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2])
        b.append(y2)

    h = np.linalg.lstsq(A, b, rcond=None)[0]
    h = np.append(h,1.0).reshape(3,3)
    return h





def warpImageNearestNeighbor(im, H):
    height, width = im.shape[0], im.shape[1]
    corners = [
        (0,0),
        (width - 1, 0),
        (width - 1, height - 1),
        (0, height - 1)
    ]
    warpedX = []
    warpedY = []

    for x, y in corners:
        point = H @ np.array([x,y,1])
        warpedX.append(point[0]/ point[2])
        warpedY.append(point[1]/ point[2])

    xmin = int(np.floor(min(warpedX)))
    xmax = int(np.ceil(max(warpedX)))
    ymin = int(np.floor(min(warpedY)))
    ymax = int(np.ceil(max(warpedY)))

    outwidth = xmax - xmin + 1
    outheight = ymax - ymin + 1
    warped = np.zeros(
        (outheight, outwidth, 3),
        dtype = im.dtype
    )

    invH = np.linalg.inv(H)
    for y in range(outheight):
        for x in range(outwidth):
            point = invH @ np.array([x + xmin, y + ymin, 1])
            sx = round(point[0] / point[2])
            sy = round(point[1] / point[2])

            if 0 <= sx < width and 0 <= sy < height:
                warped[y, x] = im[sy, sx]
                
    return warped


def warpImageBilinear(im, H, returnInfo=False):
    height, width = im.shape[0], im.shape[1]
    corners = [
        (0,0),
        (width - 1, 0),
        (width - 1, height - 1),
        (0, height - 1)
    ]
    warpedX = []
    warpedY = []

    for x, y in corners:
        point = H @ np.array([x,y,1])
        warpedX.append(point[0]/ point[2])
        warpedY.append(point[1]/ point[2])

    xmin = int(np.floor(min(warpedX)))
    xmax = int(np.ceil(max(warpedX)))
    ymin = int(np.floor(min(warpedY)))
    ymax = int(np.ceil(max(warpedY)))

    outwidth = xmax - xmin + 1
    outheight = ymax - ymin + 1
    warped = np.zeros(
        (outheight, outwidth, 3),
        dtype = im.dtype
    )

    if returnInfo:
        weights = np.zeros_like(warped)

    invH = np.linalg.inv(H)
    for y in range(ymax - ymin + 1):
        for x in range(xmax - xmin + 1):
            point = invH @ np.array([x + xmin, y + ymin, 1])
            sx = point[0] / point[2]
            sy = point[1] / point[2]

            if 0 <= sx <= width - 1 and 0 <= sy <= height - 1:
                x0 = int(np.floor(sx))
                y0 = int(np.floor(sy))
                x1 = min(x0 + 1, width - 1)
                y1 = min(y0 + 1, height - 1)

                dx = sx - x0
                dy = sy - y0
                warped[y, x] = (
                    (1 - dx) * (1 - dy) * im[y0, x0] #top pixel
                    + dx * (1 - dy) * im[y0, x1] 
                    + (1 - dx) * dy * im[y1, x0]
                    + dx * dy * im[y1, x1]
                )
                if returnInfo:
                    weights[y, x] = featherWeight(sx, sy, width, height)
                
    if returnInfo:
        return warped, weights, xmin, ymin
    return warped


def featherWeight(x, y, width, height):
    wx = 1 - abs(2 * x / (width - 1) - 1)
    wy = 1 - abs(2 * y / (height - 1) - 1)
    return max(min(wx, wy), 1e-6)


def makeMosaic(im1, im2, H12):
    a, wa, ax, ay = warpImageBilinear(im1, np.eye(3), returnInfo=True)
    b, wb, bx, by = warpImageBilinear(im2, np.linalg.inv(H12), returnInfo=True)
    xmin, ymin = min(ax, bx), min(ay, by)
    width = max(ax + a.shape[1], bx + b.shape[1]) - xmin
    height = max(ay + a.shape[0], by + b.shape[0]) - ymin
    mosaic = np.zeros((height, width, 3))
    weights = np.zeros_like(mosaic)

    for im, weight, left, top in [(a, wa, ax, ay), (b, wb, bx, by)]:
        h, w = im.shape[:2]
        x, y = left - xmin, top - ymin
        mosaic[y:y+h, x:x+w] += im * weight
        weights[y:y+h, x:x+w] += weight

    valid = weights > 0
    mosaic[valid] /= weights[valid]
    return mosaic

def mosaicPosters():
    im1 = load_image('posters_1.jpg')
    im2 = load_image('posters_2.jpg')
    pts1, pts2 = load_correspondences()
    H12 = computeH(pts1, pts2)
    mosaic = makeMosaic(im1, im2, H12)
    save_image(mosaic, "mosaic_posters.png")  
    print("Saved results/mosaic_posters.png")

def mosaicHallway():
    im1 = load_image('hallway_1.jpg')
    im2 = load_image('hallway_2.jpg')
    pts1, pts2 = load_correspondences('hallway_correspondences.json')
    H12 = computeH(pts1, pts2)
    mosaic = makeMosaic(im1, im2, H12)
    save_image(mosaic, "mosaic_hallway.png")
    print("Saved results/mosaic_hallway.png")


def mosaicBathroom():
    im1 = load_image('bathroom_1.jpg')
    im2 = load_image('bathroom_2.jpg')
    pts1, pts2 = load_correspondences('bathroom_correspondences.json')
    H12 = computeH(pts1, pts2)
    mosaic = makeMosaic(im1, im2, H12)
    save_image(mosaic, "mosaic_bathroom.png")
    print("Saved results/mosaic_bathroom.png")


def rectifyStarWars():
    im = load_image("starwars_poster.jpeg")
    pts1 = np.array([[291, 714], [1916, 451], [1947, 3520], [268, 3242]])
    pts2 = np.array([[0, 0], [799, 0], [799, 1199], [0, 1199]])
    H = computeH(pts1, pts2)
    print("Star Wars homography:")
    print(H)

    start = perf_counter()
    warped = warpImageNearestNeighbor(im, H)
    print(f"Star Wars nearest-neighbor: {perf_counter() - start:.2f} seconds")
    print("Output shape:", warped.shape)
    save_image(warped, "starwars_rectified_nearest.png")
    print("Saved results/starwars_rectified_nearest.png")

    start = perf_counter()
    warped = warpImageBilinear(im, H)
    print(f"Star Wars bilinear: {perf_counter() - start:.2f} seconds")
    save_image(warped, "starwars_rectified_bilinear.png")
    print("Saved results/starwars_rectified_bilinear.png")


def rectifyWave():
    im = load_image("wave_poster.jpg")
    pts1 = np.array([[636, 600], [2637, 349], [2673, 3896], [551, 3717]])
    pts2 = np.array([[0, 0], [799, 0], [799, 1279], [0, 1279]])
    H = computeH(pts1, pts2)
    print("Wave poster homography:")
    print(H)

    start = perf_counter()
    warped = warpImageNearestNeighbor(im, H)
    print(f"Wave nearest-neighbor: {perf_counter() - start:.2f} seconds")
    print("Output shape:", warped.shape)
    save_image(warped, "wave_rectified_nearest.png")
    print("Saved results/wave_rectified_nearest.png")

    start = perf_counter()
    warped = warpImageBilinear(im, H)
    print(f"Wave bilinear: {perf_counter() - start:.2f} seconds")
    save_image(warped, "wave_rectified_bilinear.png")
    print("Saved results/wave_rectified_bilinear.png")


if __name__ == "__main__":
    rectifyStarWars()
    rectifyWave()
    mosaicPosters()
    mosaicHallway()
    mosaicBathroom()

