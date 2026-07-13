import unittest
import torch
from models.helpers import ZeroMean
Atol = 1e-5

class TestZeroMean(unittest.TestCase):

    def test_shapes_correct(self):
        Z = ZeroMean(nrows=5, ndims=3)
        b = Z.value
        self.assertEqual(b.shape, (6, 3))
        self.assertEqual(Z.theta.shape, (5, 3))

    def test_mean_zero_property(self):
        Z = ZeroMean(5, 3)
        b = Z.value
        m = b.mean(dim=0)
        self.assertTrue(torch.allclose(m, torch.zeros_like(m), atol=Atol))

    def test_sum_zero_property(self):
        Z = ZeroMean(8, 2)
        b = Z.value
        s = b.sum(dim=0)
        print("theta:", Z.theta)
        print("b:", b)
        print("b.mean:", b.mean(0))
        print("b.sum:", b.sum(0))
        self.assertTrue(torch.allclose(s, torch.zeros_like(s), atol=Atol))

    def test_last_row_is_negative_sum(self):
        Z = ZeroMean(4, 3)
        theta_sum = Z.theta.sum(dim=0)
        b = Z.value
        last = b[-1]
        self.assertTrue(torch.allclose(last, -theta_sum, atol=Atol))

    def test_gradients_flow(self):
        Z = ZeroMean(10, 4)
        b = Z.value
        loss = b.pow(2).sum()  # simple differentiable loss
        loss.backward()

        # All θ should have gradients
        self.assertFalse((Z.theta.grad == 0).all())
        self.assertIsNotNone(Z.theta.grad)

    def test_project_from_b(self):
        Z = ZeroMean(4, 3)

        # create a random b (wrong shape allowed)
        b_new = torch.randn(5, 3)

        Z.project_from_b(b_new)
        b_proj = Z.value

        # Check centered
        self.assertTrue(torch.allclose(b_proj.mean(0),
                                       torch.zeros(3),
                                       atol=Atol))

        # Check first nrows match centered input
        b_centered = b_new - b_new.mean(dim=0, keepdim=True)
        self.assertTrue(torch.allclose(Z.theta, b_centered[:-1], atol=Atol))

    def test_equivalence_to_naive_centering(self):
        n, d = 7, 5
        Z = ZeroMean(n, d)

        # Generate many random θ and compare b
        for _ in range(20):
            with torch.no_grad():
                Z.theta.copy_(torch.randn(n, d))

            b = Z.value

            # directly compute b_naive
            b_naive = torch.cat(
                [
                    Z.theta,
                    -Z.theta.sum(0, keepdim=True)
                ],
                dim=0
            )

            self.assertTrue(torch.allclose(b, b_naive, atol=Atol))

            # check mean-zero
            self.assertTrue(torch.allclose(b.mean(0),
                                           torch.zeros(d),
                                           atol=Atol))

    def test_backward_consistency(self):
        Z = ZeroMean(6, 4)
        Z.theta.data = torch.randn(6, 4)

        # b will include last row = -sum(theta)
        b = Z.value
        loss = (b ** 3).sum()  # nonlinear to test backprop

        loss.backward()

        # Gradients exist
        self.assertIsNotNone(Z.theta.grad)

        # Make sure they are finite
        self.assertFalse(torch.isnan(Z.theta.grad).any())



if __name__ == "__main__":
    unittest.main()
