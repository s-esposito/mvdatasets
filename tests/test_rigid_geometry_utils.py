import unittest
import numpy as np
import torch
from mvdatasets.geometry.rigid import (
    apply_rotation_3d,
    apply_transformation_3d,
    pose_local_rotation,
    pose_global_rotation
)


class Test3DTransformations(unittest.TestCase):

    def test_apply_rotation_3d(self):
        # Test with NumPy
        points = np.array([[1, 0, 0]], dtype=np.float32)
        rot = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], np.float32)
        result = apply_rotation_3d(points, rot)
        expected = np.array([[0, 1, 0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

        # Test with batched rotation
        points = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        rot = np.array([
            [
                [0, -1, 0],
                [1,  0, 0],
                [0,  0, 1]
            ],
            [
                [0,  1, 0],
                [-1, 0, 0],
                [0,  0, 1]
            ]
        ], dtype=np.float32)
        result = apply_rotation_3d(points, rot)
        expected = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

        # Test with PyTorch
        points = torch.tensor([[1, 0, 0]], dtype=torch.float32)
        rot = torch.tensor([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=torch.float32)
        result = apply_rotation_3d(points, rot)
        expected = torch.tensor([[0, 1, 0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(result, expected))

    def test_apply_transformation_3d(self):
        # Test with NumPy
        points = np.array([[1, 0, 0]], dtype=np.float32)
        transform = np.array([
            [0, -1, 0, 1],
            [1,  0, 0, 2],
            [0,  0, 1, 3],
            [0,  0, 0, 1]
        ], dtype=np.float32)
        result = apply_transformation_3d(points, transform)
        expected = np.array([[1, 3, 3]], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

        # Test with PyTorch
        points = torch.tensor([[1, 0, 0]], dtype=torch.float32)
        transform = torch.tensor([
            [0, -1, 0, 1],
            [1,  0, 0, 2],
            [0,  0, 1, 3],
            [0,  0, 0, 1]
        ], dtype=torch.float32)
        result = apply_transformation_3d(points, transform)
        expected = torch.tensor([[1, 3, 3]], dtype=torch.float32)
        self.assertTrue(torch.allclose(result, expected))

    def test_pose_local_rotation(self):
        # Test with NumPy
        pose = np.eye(4, dtype=np.float32)
        rotation = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=np.float32)
        result = pose_local_rotation(pose, rotation)
        expected = np.array([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

        # Test with PyTorch
        pose = torch.eye(4, dtype=torch.float32)
        rotation = torch.tensor([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=torch.float32)
        result = pose_local_rotation(pose, rotation)
        expected = torch.tensor([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ], dtype=torch.float32)
        self.assertTrue(torch.allclose(result, expected))

    def test_pose_global_rotation(self):
        # Test with NumPy
        pose = np.eye(4, dtype=np.float32)
        rotation = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=np.float32)
        result = pose_global_rotation(pose, rotation)
        expected = np.array([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

        # Test with PyTorch
        pose = torch.eye(4, dtype=torch.float32)
        rotation = torch.tensor([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=torch.float32)
        result = pose_global_rotation(pose, rotation)
        expected = torch.tensor([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ], dtype=torch.float32)
        self.assertTrue(torch.allclose(result, expected))


if __name__ == "__main__":
    unittest.main()
