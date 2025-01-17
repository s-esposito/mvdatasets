import unittest
import torch
import numpy as np
from mvdatasets.geometry.quaternions import (
    quat_multiply,
    quat_invert,
    make_quaternion_deg,
    make_quaternion_rad,
    rots_to_quats,
    angular_distance,
    quats_to_rots
)

class TestQuaternionFunctions(unittest.TestCase):

    def test_quat_multiply(self):
        a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        b = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        result = quat_multiply(a, b)
        expected = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(result, expected))

        # Test associativity of quaternion multiplication
        c = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        result_abc = quat_multiply(quat_multiply(a, b), c)
        result_cba = quat_multiply(a, quat_multiply(b, c))
        self.assertTrue(torch.allclose(result_abc, result_cba))

    def test_quat_invert(self):
        q = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])
        result = quat_invert(q)
        expected = torch.tensor([[0.7071, -0.7071, 0.0, 0.0]])
        self.assertTrue(torch.allclose(result, expected))

        # Verify double inversion gives the original quaternion
        double_inverted = quat_invert(result)
        self.assertTrue(torch.allclose(double_inverted, q))

    def test_make_quaternion_deg(self):
        q = make_quaternion_deg(90, 0, 0)
        expected = torch.tensor([0.7071, 0.7071, 0.0, 0.0])
        self.assertTrue(torch.allclose(q, expected, atol=1e-4))

    def test_make_quaternion_rad(self):
        q = make_quaternion_rad(np.pi / 2, 0, 0)
        expected = torch.tensor([0.7071, 0.7071, 0.0, 0.0])
        self.assertTrue(torch.allclose(q, expected, atol=1e-4))

    def test_rots_to_quats(self):
        rots = torch.eye(3).unsqueeze(0)  # Single identity matrix
        result = rots_to_quats(rots)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(result, expected))

        # Test rotation matrix for a 90-degree rotation around Z-axis
        rot_z = torch.tensor([
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]
            ]
        ])
        result = rots_to_quats(rot_z)
        expected = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_angular_distance(self):
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q2 = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])
        result = angular_distance(q1, q2)
        expected = torch.tensor([0.2929])  # 1 - abs(dot_product)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

        # Test angular distance between identical quaternions
        result = angular_distance(q1, q1)
        expected = torch.tensor([0.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_quats_to_rots(self):
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        result = quats_to_rots(quats)
        expected = torch.eye(3).unsqueeze(0)
        self.assertTrue(torch.allclose(result, expected))

        # Test quaternion to rotation matrix for 90-degree rotation around Y-axis
        quats = torch.tensor([[0.7071, 0.0, 0.7071, 0.0]])
        result = quats_to_rots(quats)
        expected = torch.tensor([
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0]
            ]
        ])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
