# B.3 and B.4 code walkthrough

## match_features

A descriptor is one normalized 8x8 patch. For every descriptor in image 1, subtract it from all descriptors in image 2. Square the pixel differences and sum across rows and columns, leaving one error per candidate. Sort the errors and compare the best with the runner-up. Keep the pair only when the best squared error is less than 0.8 times the second-best error. The output is an M-by-2 array of indices, not coordinates: `[i, j]` pairs corner i in image 1 with corner j in image 2. Equal zero errors are rejected because the comparison is strict.

## project_points

`np.column_stack((points, np.ones(len(points))))` adds a 1 to each `(x, y)` coordinate, producing `(x, y, 1)`. Homographies use these homogeneous coordinates.

`homogeneous @ H.T` applies the same homography to every point. We transpose H because each point is stored as a row; this is equivalent to multiplying H by an individual column vector.

`projected[:, :2] / projected[:, 2:3]` divides both transformed coordinates by the third component, returning ordinary `(x, y)` coordinates. This returns an N-by-2 array. Invalid projections may produce infinities or NaNs; they cannot pass the RANSAC pixel-error threshold.

## ransac_homography

1. Convert both matched-coordinate arrays to floating-point arrays. At least four pairs are needed to estimate a homography.
2. Create a random-number generator. The fixed seed reproduces the same random samples.
3. Start with no inliers and an infinite tie-breaking error.
4. Repeat 2,000 times. Choose four distinct match indices without replacement. The same indices select the source and destination points, preserving each correspondence.
5. Skip samples whose four source or destination points all lie on a line. Such a sample cannot determine a valid planar transformation.
6. Call the Part A `computeH` function on those four pairs. Skip singular homographies, which cannot be inverted for warping.
7. Project ALL matched source points with the proposed H, not only the four sampled pairs.
8. Subtract the actual destination coordinates and calculate ordinary Euclidean distance: square x and y differences, add them, and take a square root.
9. Mark a pair as an inlier if its reprojection error is below three pixels. Here `inliers` is a Boolean array, one entry per candidate match.
10. Keep the mask with the most inliers. When two masks have the same count, prefer the one with lower total inlier error.
11. After the random trials, fit a new H using ALL inliers, rather than trusting only four points. Recompute the inlier mask and repeat this refinement up to three times.
12. Return H and the final mask. H maps image 1 coordinates to image 2 coordinates.

RANSAC does not directly decide whether two patches look alike. B.3 already proposed those pairs. RANSAC asks whether many proposed pairs agree with ONE geometric transformation.

Example: if 150 candidate pairs agree on one H and 100 do not, the 150 form a consensus. Refitting H on them usually gives a more stable estimate than fitting all 250 pairs or trusting one four-point sample.

## stitch_automatically

Detect Harris candidates in both images, use ANMS to select 500 in each, extract normalized descriptors, and match descriptors. Convert accepted index pairs into actual coordinates. Run RANSAC on those coordinates. Pass the estimated H to the Part A `makeMosaic` function. Return the mosaic and intermediate values so the runner can save visual evidence and statistics.

`makeMosaic` keeps image 1 fixed and warps image 2 using the inverse of H. Its existing bilinear interpolation and feather blending produce the final automatic mosaic. No manually selected correspondences are used to estimate the automatic H.

## save_mosaic_comparison

Create two plots, put the manual mosaic in the first and the automatic mosaic in the second, add labels, save the figure, and close it. The two mosaics can have different canvas bounds because their homographies differ slightly.

## run_b4

Load the photos and call the complete automatic pipeline. Save the automatic mosaic. Load the manually clicked correspondences and use the existing Part A mosaic as the baseline; if that image is missing, recreate it using the manual H. Save a side-by-side comparison and an overlay showing RANSAC inliers in green and rejected pairs in red. Finally, save the automatic H, matched coordinates, inlier mask, reprojection errors, and a JSON statistics file.

## main

Run all B sections by default, or select `--part b1`, `b2`, `b3`, or `b4`. Each section runs on the posters, hallway, and bathroom images. The poster correspondence JSON has the singular filename `poster_correspondences.json`.

## What the results can and cannot show

An inlier count measures agreement with the estimated homography, not independently verified matching accuracy. A single homography cannot perfectly align a scene with camera translation and objects at different depths. Feather blending hides brightness seams but cannot repair parallax or incorrect geometry. Residual double edges should therefore be discussed honestly on the website.
