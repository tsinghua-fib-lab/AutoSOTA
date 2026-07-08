import jax.numpy as jnp

from smz.planning import impl


class TestTreeTDLambda:

    def test_single_trace_mc(self):
        # Check if valid sum of rewards under a single trace

        # Format into (time, batch)
        states = jnp.arange(3)[..., None] * 1.0
        rewards = discounts = jnp.ones((3, 1))
        values = jnp.zeros((3, 1))

        out = impl.tree_truncated_td_lambda(
            states, states + 1, rewards, discounts, values,
            gamma=1.0, lambda_=1.0
        )

        assert out.shape == states.shape, "Incorrect output shape"

        # For 3 steps of reward 1, we expect [3, 2, 1]
        expected = jnp.asarray([3., 2., 1.])[..., None]
        assert jnp.isclose(out, expected).all(), "Incorrect value estimated"

    def test_branched_trace_mc(self):
        # Check if valid sum of rewards under a branched trace

        # Connected path index = (:, 0), disconnected path index = (:, 1)
        states = jnp.asarray([[0, -1], [1, 2], [2, 4]])

        rewards = discounts = jnp.ones((3, 2))
        values = jnp.zeros((3, 2))

        out = impl.tree_truncated_td_lambda(
            states, states + 1, rewards, discounts, values,
            gamma=1.0, lambda_=1.0
        )

        assert out.shape == states.shape, "Incorrect output shape"

        # For connected path, [3, 2, 1]; disconnected path [1, 1, 1]
        expected = jnp.asarray([[3., 1.], [2., 1.], [1., 1.]])
        assert jnp.isclose(out, expected).all(), "Incorrect value estimated"

    def test_single_trace_bootstrap(self):
        # Check if valid backed up bootstrap under a single trace

        # Format into (time, batch)
        states = jnp.arange(3)[..., None] * 1.0
        discounts = jnp.ones((3, 1))
        rewards = jnp.zeros((3, 1))

        values = jnp.arange(1, 4)[..., None] * 1.0

        # 0.5^1 (0.0 + 1.0) + 0.5^2 (0.0 + 2.0) + 0.5^3 (0.0 + 3.0)
        out = impl.tree_truncated_td_lambda(
            states, states + 1, rewards, discounts, values,
            gamma=1.0, lambda_=0.5
        )

        assert out.shape == states.shape, "Incorrect output shape"

        # Backup connected path using TD-lambda
        expected = jnp.asarray([1.75, 2.5, 3.])
        assert jnp.isclose(out, expected[..., None]).all(), \
            "Incorrect value estimated"

    def test_branched_trace_bootstrap(self):
        # Check if valid backed up bootstrap under a branched trace

        # Connected path index = (:, 0), disconnected path index = (:, 1)
        states = jnp.asarray([[0, -1], [1, 2], [2, 4]])

        discounts = jnp.ones((3, 2))
        rewards = jnp.zeros((3, 2))

        values = jnp.asarray([[1., 1.], [2., 2.], [3., 3.]])

        out = impl.tree_truncated_td_lambda(
            states, states + 1, rewards, discounts, values,
            gamma=1.0, lambda_=.5
        )

        assert out.shape == states.shape, "Incorrect output shape"

        # Connected index (:, 0) is backed up; Disconnected index(:, 1) not
        expected = jnp.asarray([[1.75, 1.], [2.5, 2.], [3., 3.]])
        assert jnp.isclose(out, expected).all(), "Incorrect value estimated"

    def test_complete(self):
        """Manually designed test for checking correctness of tree TD lambda

        Setup:
         - constant rewards = 1.0
         - constant disocunts = 0.8
         - constant lambda = 0.7

        All paths:
        0 -> 1 (duplicated)
        0 -> 2
        0 -> 3 (dropped)
        1 -> 4 (dropped)
        1 -> 5
        2 -> 6 (duplicated)
        5 -> 8
        6 -> 7
        6 -> 9

        All values
        0: NA, 1: 2, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 1, 8, 2, 9: 3

        All backups (td-returns along the reverse paths):
        7 to 6: 1 + 0.8 * 1 = 1.8
        8 to 5: 1 + 0.8 * 2 = 2.6
        9 to 6: 1 + 0.8 * 3 = 3.4
        4 to 1: 1 + 0.8 * 0 = 1.0
        5 to 1: 1 + 0.8 * ((1 - 0.7) * 1 + 0.7 * 2.6) = 2.696
        6 to 2: 1 + 0.8 * ((1 - 0.7) * 2 + 0.7 * (1.8 + 3.4) / 2) = 2.936
        1 to 0: 1 + 0.8 * ((1 - 0.7) * 2 + 0.7 * (1.0 + 2.696) / 2) = 2.51488
        2 to 0: 1 + 0.8 * ((1 - 0.7) * 1 + 0.7 * 2.936) = 2.88416
        3 to 0: 1 + 0.8 * 0 = 1.0
        """

        states = jnp.asarray([
            [0, 0, 0],
            [1, 1, 2],
            [6, 5, 6]], jnp.float32
        )
        next_states = jnp.asarray([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], jnp.float32
        )
        values = jnp.asarray([
            [2, 1, 0],
            [0, 1, 2],
            [1, 2, 3]], jnp.float32
        )

        rewards = jnp.ones((3, 3)) * 1.
        discounts = jnp.ones((3, 3)) * 0.8

        out = impl.tree_truncated_td_lambda(
            states, next_states, rewards, discounts, values,
            gamma=1.0, lambda_=0.7
        )

        assert out.shape == states.shape, "Incorrect output shape"

        expected = jnp.asarray([
            [2.51488, 2.88416, 1.0],
            [1.0, 2.696, 2.936],
            [1.8, 2.6, 3.4]
        ])

        assert jnp.isclose(out, expected).all(), "Incorrect value estimated"
