import tempfile
import unittest
from pathlib import Path

import numpy as np

import proj4b as b


class Project4BTests(unittest.TestCase):
    def test_anms_uses_distance_to_sufficiently_stronger_points(self):
        selected, radii = b.adaptive_non_maximal_suppression(
            [[0, 0], [3, 4], [20, 0]], [10, 8, 7], num_points=2)
        np.testing.assert_array_equal(selected, [0, 2])
        self.assertTrue(np.isinf(radii[0]))
        self.assertEqual(radii[1], 5)
        self.assertAlmostEqual(radii[2], np.sqrt(305))

    def test_integer_grayscale_descriptors_use_float_blurring(self):
        image = np.random.default_rng(2).integers(0, 256, (80, 80), dtype=np.uint8)
        points = np.array([[40, 40]])
        actual = b.extract_feature_descriptors(image, points)
        expected = b.extract_feature_descriptors(image.astype(float), points)
        np.testing.assert_allclose(actual, expected)
        self.assertAlmostEqual(actual.mean(), 0)
        self.assertAlmostEqual(actual.std(), 1)

    def test_flat_descriptors_and_invalid_patch(self):
        image = np.ones((80, 80))
        np.testing.assert_array_equal(
            b.extract_feature_descriptors(image, [[40, 40]]), np.zeros((1, 8, 8)))
        with self.assertRaises(ValueError):
            b.extract_feature_descriptors(image, [[2, 2]])
        with self.assertRaises(ValueError):
            b.extract_feature_descriptors(image, [[40, 40]], patch_size=42)

    def test_ratio_matching_rejects_ambiguous_and_missing_neighbors(self):
        np.testing.assert_array_equal(
            b.match_features(np.array([[[0.]]]), np.array([[[1.]], [[3.]]])), [[0, 0]])
        self.assertEqual(b.match_features(np.zeros((1, 8, 8)), np.zeros((2, 8, 8))).shape, (0, 2))
        self.assertEqual(b.match_features(np.zeros((1, 8, 8)), np.zeros((1, 8, 8))).shape, (0, 2))

    def test_ransac_recovers_projective_transform_with_outliers(self):
        rng = np.random.default_rng(4)
        source = rng.uniform(0, 500, (100, 2))
        expected = np.array([[1.1, .02, 30], [-.03, .95, 15], [.0001, -.0002, 1.]])
        target = b.project_points(source, expected)
        target[:30] = rng.uniform(0, 500, (30, 2))
        H, inliers = b.ransac_homography(source, target, num_iterations=200)
        np.testing.assert_array_equal(inliers, np.arange(100) >= 30)
        np.testing.assert_allclose(b.project_points(source[30:], H), target[30:], atol=1e-6)

    def test_ransac_rejects_degenerate_and_mismatched_data(self):
        collinear = np.column_stack((np.arange(6), np.arange(6)))
        with self.assertRaises(ValueError):
            b.ransac_homography(collinear, collinear, num_iterations=10)
        with self.assertRaises(ValueError):
            b.ransac_homography(collinear, collinear[:4])
        with self.assertRaises(ValueError):
            b.ransac_homography(collinear, collinear, num_iterations=0)

    def test_descriptor_plot_saves(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'descriptors.png'
            b.save_descriptor_examples(np.zeros((80, 80)), np.array([[40, 40]]),
                                       np.zeros((1, 8, 8)), path)
            self.assertTrue(path.is_file())


if __name__ == '__main__':
    unittest.main()
