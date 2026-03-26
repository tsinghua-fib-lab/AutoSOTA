# acia\datasets

from sklearn.manifold import TSNE
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision import transforms
import torch
from torch.utils.data import TensorDataset
from scipy.ndimage import rotate, gaussian_filter
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


class ColoredMNIST(Dataset):
    def __init__(self, env: str, root='./data', train=True, intervention_type='none', intervention_strength=1.0):
        super().__init__()
        self.env = env
        self.intervention_type = intervention_type
        self.intervention_strength = intervention_strength
        mnist = datasets.MNIST(root=root, train=train, download=True)
        self.images = mnist.data.float() / 255.0
        self.labels = mnist.targets
        self.colored_images = self._color_images()
        self.env_labels = torch.full_like(self.labels, float(env == 'e2'))

    def _color_images(self) -> torch.Tensor:
        n_images = len(self.images)
        colored = torch.zeros((n_images, 3, 28, 28))

        for i, (img, label) in enumerate(zip(self.images, self.labels)):
            # Base probabilities - anti-causal relationship (Y → X)
            # In 'e1': P(red|even) = 0.75, P(red|odd) = 0.25
            # In 'e2': P(red|even) = 0.25, P(red|odd) = 0.75
            base_p_red = 0.75 if ((label % 2 == 0) == (self.env == 'e1')) else 0.25

            if self.intervention_type == 'none':
                p_red = base_p_red

            elif self.intervention_type == 'perfect':
                p_red = 1.0 if base_p_red > 0.5 else 0.0

            elif self.intervention_type == 'imperfect':
                perfect_p = 1.0 if base_p_red > 0.5 else 0.0
                p_red = (1 - self.intervention_strength) * base_p_red + self.intervention_strength * perfect_p

            is_red = torch.rand(1) < p_red
            if is_red:
                colored[i, 0] = img  # Red channel
            else:
                colored[i, 1] = img  # Green channel
        return colored

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.colored_images[idx], self.labels[idx], self.env_labels[idx]


class RotatedMNIST:
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, angles, root='./data', train=True, intervention_type='none', intervention_strength=1.0):

        self.angles = angles
        self.intervention_type = intervention_type
        self.intervention_strength = intervention_strength
        mnist = datasets.MNIST(root=root, train=train, download=True)
        self.images = mnist.data.numpy() / 255.0
        self.labels = mnist.targets.numpy()
        self.env_data = {}

        for env_name, base_angle in self.angles.items():
            digit_angles = self._get_rotation_angles(base_angle)
            rotated_images = np.zeros_like(self.images)
            env_labels = np.zeros_like(self.labels)

            for i, (img, label) in enumerate(zip(self.images, self.labels)):
                angle = digit_angles[label]
                rotated_images[i] = rotate(img, angle, reshape=False)
                env_labels[i] = list(self.angles.keys()).index(env_name)

            tensor_images = torch.tensor(rotated_images, dtype=torch.float32).unsqueeze(1)
            tensor_labels = torch.tensor(self.labels)
            tensor_envs = torch.tensor(env_labels)

            self.env_data[env_name] = {'images': tensor_images,
                'labels': tensor_labels,
                'envs': tensor_envs}
        self._flatten_data()

    def _get_rotation_angles(self, base_angle):
        if self.intervention_type == 'none':
            return {d: base_angle for d in range(10)}

        elif self.intervention_type == 'perfect':
            return {d: base_angle if d % 2 == 0 else (base_angle + 45) % 360
                    for d in range(10)}

        elif self.intervention_type == 'imperfect':
            perfect_angles = {d: base_angle if d % 2 == 0 else (base_angle + 45) % 360 for d in range(10)}

            return {d: (1 - self.intervention_strength) * base_angle +
                       self.intervention_strength * perfect_angles[d]
                    for d in range(10)}

    def _flatten_data(self):
        images = []
        labels = []
        envs = []

        for env_data in self.env_data.values():
            images.append(env_data['images'])
            labels.append(env_data['labels'])
            envs.append(env_data['envs'])

        self.images_flat = torch.cat(images)
        self.labels_flat = torch.cat(labels)
        self.envs_flat = torch.cat(envs)

    def __len__(self):
        return len(self.images_flat)

    def __getitem__(self, idx):
        return self.images_flat[idx], self.labels_flat[idx], self.envs_flat[idx]


class Camelyon17Dataset(Dataset):
    def __init__(self, root_dir, metadata_path, hospital_id, indices):
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata['center'] == hospital_id]
        self.metadata = self.metadata.loc[indices]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = f"patch_patient_{row['patient']:03d}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png"
        img_path = self.root_dir / f"patient_{row['patient']:03d}_node_{row['node']}" / img_name
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        tumor = torch.tensor(row['tumor'], dtype=torch.long)
        hospital = torch.zeros(5)
        hospital[row['center']] = 1
        return image, tumor, hospital



class BallAgentDataset:
    def __init__(self, n_balls=3, n_samples=10000, size=64, intervention_type='none', intervention_strength=1.0):
        self.n_balls = n_balls
        self.n_samples = n_samples
        self.size = size
        self.intervention_type = intervention_type
        self.intervention_strength = intervention_strength
        self.colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0.7, 0)]
        self.positions, self.images, self.interventions = self._generate_data()

    def _generate_data(self):
        positions = []
        images = []
        interventions = []

        while len(positions) < self.n_samples:
            pos = np.random.uniform(0.1, 0.9, (self.n_balls, 2))
            valid = True
            for i in range(self.n_balls):
                for j in range(i + 1, self.n_balls):
                    dist = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
                    if dist < 0.2:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                continue
            n_interventions = np.random.randint(0, self.n_balls * 2 + 1)
            intervention_idx = np.random.choice(self.n_balls * 2, n_interventions, replace=False)
            intervention = np.zeros(self.n_balls * 2, dtype=bool)
            intervention[intervention_idx] = True
            final_pos = self._apply_intervention(pos, intervention)
            positions.append(final_pos.flatten())
            images.append(self._render_image(final_pos))
            interventions.append(intervention)
        return np.array(positions), np.array(images), np.array(interventions)

    def _apply_intervention(self, pos, intervention):
        intervened_pos = pos.copy()

        if self.intervention_type == 'none':
            return pos

        if self.intervention_type == 'perfect':
            for i in range(len(intervention)):
                if intervention[i]:
                    ball_idx = i // 2
                    coord_idx = i % 2
                    intervened_pos[ball_idx, coord_idx] = (0.2 + 0.6 * (ball_idx / self.n_balls))

        elif self.intervention_type == 'imperfect':
            for i in range(len(intervention)):
                if intervention[i]:
                    ball_idx = i // 2
                    coord_idx = i % 2
                    new_val = np.random.uniform(0.1, 0.9)
                    orig_val = pos[ball_idx, coord_idx]
                    intervened_pos[ball_idx, coord_idx] = ((1 - self.intervention_strength) * orig_val +
                            self.intervention_strength * new_val)

                    attempts = 0
                    valid = self._check_constraints(intervened_pos)
                    while not valid and attempts < 10:
                        new_val = np.random.uniform(0.1, 0.9)
                        intervened_pos[ball_idx, coord_idx] = ((1 - self.intervention_strength) * orig_val +
                                self.intervention_strength * new_val)
                        valid = self._check_constraints(intervened_pos)
                        attempts += 1
        return intervened_pos

    def _check_constraints(self, pos):
        for i in range(self.n_balls):
            for j in range(i + 1, self.n_balls):
                dist = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
                if dist < 0.2:
                    return False
        return True

    def _render_image(self, positions):
        img = np.zeros((self.size, self.size, 3))

        for i, (x, y) in enumerate(positions):
            px, py = int(x * self.size), int(y * self.size)
            color = self.colors[i % len(self.colors)]

            y_grid, x_grid = np.ogrid[-8:9, -8:9]
            distances = np.sqrt(x_grid ** 2 + y_grid ** 2)
            intensity = np.exp(-distances ** 2 / (2 * 4.0 ** 2))
            for c in range(3):
                y_coords = np.clip(py + y_grid, 0, self.size - 1)
                x_coords = np.clip(px + x_grid, 0, self.size - 1)
                img[y_coords, x_coords, c] += intensity * color[c]
        return np.clip(img, 0, 1)


class BallAgentEnvironment(Dataset):
    def __init__(self, dataset, is_train=True):
        self.data = dataset
        self.is_train = is_train
        self.train_idx, self.test_idx = self._split_data()

    def _split_data(self):
        n = len(self.data.positions)
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        return indices[:split], indices[split:]

    def __len__(self):
        return len(self.train_idx) if self.is_train else len(self.test_idx)

    def __getitem__(self, idx):
        indices = self.train_idx if self.is_train else self.test_idx
        real_idx = indices[idx]
        x = torch.FloatTensor(self.data.images[real_idx])
        y = torch.FloatTensor(self.data.positions[real_idx])
        e = torch.FloatTensor(self.data.interventions[real_idx])
        return x.permute(2, 0, 1), y, e

def visualize_cmnist_results(model, loader, device, save_path):
    model.eval()
    z_L_all, z_H_all, labels, envs = [], [], [], []
    with torch.no_grad():
        for x, y, e in loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            z_L, z_H, _ = model(x)
            z_L_all.append(z_L.cpu())
            z_H_all.append(z_H.cpu())
            labels.append(y.cpu())
            envs.append(e.cpu())

    z_L_all = torch.cat(z_L_all).numpy()
    z_H_all = torch.cat(z_H_all).numpy()
    labels = torch.cat(labels).numpy()
    envs = torch.cat(envs).numpy()
    parity = np.array(['Even' if y % 2 == 0 else 'Odd' for y in labels])
    tsne = TSNE(n_components=2, random_state=42)
    z_L_2d = tsne.fit_transform(z_L_all)
    z_H_2d = tsne.fit_transform(z_H_all)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Plot 1: Low-level by digit
    scatter = axes[0].scatter(z_L_2d[:, 0], z_L_2d[:, 1], c=labels, cmap='tab10')
    axes[0].set_title('Low-level Representation by Digit')
    fig.colorbar(scatter, ax=axes[0])

    # Plot 2: High-level by digit
    scatter = axes[1].scatter(z_H_2d[:, 0], z_H_2d[:, 1], c=labels, cmap='tab10')
    axes[1].set_title('High-level Representation by Digit')
    fig.colorbar(scatter, ax=axes[1])

    # Plot 3: High-level by environment
    scatter = axes[2].scatter(z_H_2d[:, 0], z_H_2d[:, 1], c=envs, cmap='Set1')
    axes[2].set_title('High-level Representation by Environment')
    fig.colorbar(scatter, ax=axes[2])

    # Plot 4: High-level by parity
    parity_map = {'Even': 0, 'Odd': 1}
    parity_numeric = np.array([parity_map[p] for p in parity])
    scatter = axes[3].scatter(z_H_2d[:, 0], z_H_2d[:, 1], c=parity_numeric, cmap='RdYlBu')
    axes[3].set_title('High-level Representation by Parity')
    fig.colorbar(scatter, ax=axes[3])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

