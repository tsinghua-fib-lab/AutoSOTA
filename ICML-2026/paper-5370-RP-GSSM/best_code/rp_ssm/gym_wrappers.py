import math
from typing import Optional

import cv2

import numpy as onp

from gym import spaces, logger
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.classic_control import utils


class ExtendedCartpole(CartPoleEnv):
    """
    Wrapper around gym's cartpole that allows
    for no actions and image data.

    If `img_shape` is None, use standard physics
    observations. Otherwise use image observations
    with shape (W, H) = img_shape.
    """

    def __init__(
        self,
        img_shape: Optional[tuple[int, int]] = None
    ):
        super().__init__()
        self.img_shape = img_shape
        self.render_mode = "rgb_array"
        self.action_space = spaces.Discrete(3, start=-1)
        if img_shape is not None:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=img_shape+(3,),
                dtype="uint8",
            )

    def step(self, action):
        """Copied from gym and edited"""
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = float(action) * self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()

        if self.img_shape is not None:
            observation = cv2.resize(
                self.render(),
                self.img_shape[:2]
            )
        else:
            observation = self.state

        return onp.array(observation, dtype=self.observation_space.dtype), reward, terminated, False, {}
    
    def reset(self, seed=None, options=None):
        """Copied from gym and edited"""
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()

        if self.img_shape is not None:
            observation = cv2.resize(
                self.render(),
                self.img_shape[:2]
            )
        else:
            observation = self.state

        return onp.array(observation, dtype=self.observation_space.dtype), {}