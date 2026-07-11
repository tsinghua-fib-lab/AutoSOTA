import numpy as np
import random
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import torch


class RandomRotation:
    def __init__(self, angle_range=(-90, 90), fill=0):
        self.angle_range = angle_range
        self.fill = fill

    def __call__(self, pil_img):
        angle = float(np.random.uniform(*self.angle_range))
        out = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=self.fill)
        return out, {"angle": angle}


class RandomAffine:
    def __init__(self, angle_range=(-90, 90), translations=(-2, -1, 0, 1, 2), fill=0):
        self.angle_range = angle_range
        self.translations = translations
        self.fill = fill

    def __call__(self, pil_img):
        angle = float(np.random.uniform(*self.angle_range))
        tx = int(random.choice(self.translations))
        ty = int(random.choice(self.translations))

        transformed = TF.affine(
            pil_img,
            angle=angle,
            translate=(tx, ty),
            scale=1.0,
            shear=0.0,
            interpolation=InterpolationMode.BILINEAR,
            fill=self.fill,
        )

        return transformed, {
            "angle": angle,
            "translation": (tx, ty),
        }


# Transformations for ModelNet10
class RandomTurntableSO3:
    def __init__(self, yaw_range=(0, 360), pitch_range=(-10, 10), roll_range=(-10, 10)):
        self.yaw_range = yaw_range
        self.pitch_range = pitch_range
        self.roll_range = roll_range

    def __call__(self, points):
        # 1. Sample angles in radians
        yaw = np.radians(np.random.uniform(*self.yaw_range))
        pitch = np.radians(np.random.uniform(*self.pitch_range))
        roll = np.radians(np.random.uniform(*self.roll_range))

        # 2. Create rotation matrices for each axis
        # Rotation around Y (Yaw)
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_yaw = torch.tensor([
            [cy,  0, sy],
            [0,   1, 0],
            [-sy, 0, cy]
        ], dtype=torch.float32)

        # Rotation around X (Pitch)
        cp, sp = np.cos(pitch), np.sin(pitch)
        R_pitch = torch.tensor([
            [1, 0,   0],
            [0, cp, -sp],
            [0, sp,  cp]
        ], dtype=torch.float32)

        # Rotation around Z (Roll)
        cr, sr = np.cos(roll), np.sin(roll)
        R_roll = torch.tensor([
            [cr, -sr, 0],
            [sr,  cr, 0],
            [0,   0,  1]
        ], dtype=torch.float32)

        # 3. Combine them: R = R_yaw @ R_pitch @ R_roll
        R = R_yaw @ R_pitch @ R_roll

        # Apply transformation
        transformed = points @ R.T

        return transformed, {
            "R": R,
            "angles": {"yaw": yaw, "pitch": pitch, "roll": roll}
        }


class RandomScaling:
    def __init__(self, scale_range=(0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, points):
        s = float(np.random.uniform(*self.scale_range))
        transformed = points * s

        return transformed, {
            "group": "scaling",
            "scale": s,
        }

class RandomLorentz:

    def __init__(
        self,
        beta_range=(0.0, 0.8),
        rotation=True,
    ):
        self.beta_range = beta_range
        self.rotation = rotation

    def random_rotation(self):

        A = torch.randn(3, 3)

        Q, _ = torch.linalg.qr(A)

        if torch.det(Q) < 0:
            Q[:, 0] *= -1

        return Q

    def random_unit_vector(self):

        v = torch.randn(3)
        v = v / v.norm()

        return v

    def boost_matrix(self, beta_vec):

        beta2 = torch.dot(beta_vec, beta_vec)

        gamma = 1.0 / torch.sqrt(1.0 - beta2)

        Lambda = torch.eye(4)

        Lambda[0, 1:] = -gamma * beta_vec
        Lambda[1:, 0] = -gamma * beta_vec

        if beta2 > 0:

            outer = torch.outer(beta_vec, beta_vec)

            spatial = (
                torch.eye(3)
                + ((gamma - 1.0) / beta2) * outer
            )

            Lambda[1:, 1:] = spatial

        Lambda[0, 0] = gamma

        return Lambda

    def sample_lorentz(self):

        beta_mag = torch.empty(1).uniform_(
            *self.beta_range
        ).item()

        direction = self.random_unit_vector()

        beta_vec = beta_mag * direction

        B = self.boost_matrix(beta_vec)

        if self.rotation:

            R = self.random_rotation()

            R4 = torch.eye(4)
            R4[1:, 1:] = R

            Lambda = R4 @ B

        else:
            Lambda = B

        return Lambda, beta_vec

    def __call__(self, particles):

        Lambda, beta_vec = self.sample_lorentz()

        transformed = particles.clone()

        # valid particles only
        mask = particles.abs().sum(dim=-1) > 0

        transformed[mask] = (
            particles[mask] @ Lambda.T
        )

        return transformed, {
            "Lambda": Lambda,
            "beta": beta_vec,
        }