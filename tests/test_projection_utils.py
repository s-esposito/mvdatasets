import unittest
import numpy as np
import torch
from mvdatasets.geometry.projections import (
    local_perspective_projection,
    local_inv_perspective_projection,
    global_perspective_projection,
    global_inv_perspective_projection
)

class TestProjectionFunctions(unittest.TestCase):

    def test_local_perspective_projection(self):
        # Test with NumPy
        intrinsics = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        points_3d = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
        result = local_perspective_projection(intrinsics, points_3d)
        expected = np.array([[1320, 1240], [1320, 1240], [1320, 1240]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-4)

        # Test with PyTorch
        intrinsics = torch.tensor([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ], dtype=torch.float32)
        points_3d = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.float32)
        result = local_perspective_projection(intrinsics, points_3d)
        expected = torch.tensor([[1320, 1240], [1320, 1240], [1320, 1240]], dtype=torch.float32)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_local_inv_perspective_projection(self):
        # Test with NumPy
        intrinsics_inv = np.linalg.inv(
            np.array([
                [1000, 0, 320],
                [0, 1000, 240],
                [0, 0, 1]
            ], dtype=np.float32)
        )
        points_2d = np.array([[1320, 1240], [1320, 1240], [1320, 1240]], dtype=np.float32)
        result = local_inv_perspective_projection(intrinsics_inv, points_2d)
        expected = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-4)

        # Test with PyTorch
        intrinsics_inv = torch.inverse(
            torch.tensor([
                [1000, 0, 320],
                [0, 1000, 240],
                [0, 0, 1]
            ], dtype=torch.float32)
        )
        points_2d = torch.tensor([[1320, 1240], [1320, 1240], [1320, 1240]], dtype=torch.float32)
        result = local_inv_perspective_projection(intrinsics_inv, points_2d)
        expected = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_global_perspective_projection(self):
        # Test with NumPy
        intrinsics = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        points_3d_world = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
        result, mask = global_perspective_projection(intrinsics, c2w, points_3d_world)
        expected = np.array([[1320, 1240], [1320, 1240], [1320, 1240]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-4)
        np.testing.assert_array_equal(mask, [True, True, True])

        # Test with PyTorch
        intrinsics = torch.tensor([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ], dtype=torch.float32)
        c2w = torch.eye(4, dtype=torch.float32)
        points_3d_world = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.float32)
        result, mask = global_perspective_projection(intrinsics, c2w, points_3d_world)
        expected = torch.tensor([[1320, 1240], [1320, 1240], [1320, 1240]], dtype=torch.float32)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))
        self.assertTrue(torch.equal(mask, torch.tensor([True, True, True])))

    def test_global_inv_perspective_projection(self):
        # Test with NumPy
        intrinsics_inv = np.linalg.inv(
            np.array([
                [1000, 0, 320],
                [0, 1000, 240],
                [0, 0, 1]
            ], dtype=np.float32)
        )
        c2w = np.eye(4, dtype=np.float32)
        points_2d_screen = np.array([[1320, 1240], [1320, 1240], [1320, 1240]], dtype=np.float32)
        depth = np.array([1, 2, 3], dtype=np.float32)
        result = global_inv_perspective_projection(
            intrinsics_inv, c2w, points_2d_screen, depth
        )
        expected = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-4)

        # Test with PyTorch
        intrinsics_inv = torch.inverse(
            torch.tensor([
                [1000, 0, 320],
                [0, 1000, 240],
                [0, 0, 1]
            ], dtype=torch.float32)
        )
        c2w = torch.eye(4, dtype=torch.float32)
        points_2d_screen = torch.tensor([[1320, 1240], [1320, 1240], [1320, 1240]], dtype=torch.float32)
        depth = torch.tensor([1, 2, 3], dtype=torch.float32)
        result = global_inv_perspective_projection(
            intrinsics_inv, c2w, points_2d_screen, depth
        )
        expected = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.float32)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
