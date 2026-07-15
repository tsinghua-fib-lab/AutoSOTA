import argparse
import numpy as np
import torch
from scipy.stats import wasserstein_distance
import os
import json


def get_calo_images(directory, num_samples):
    directory = os.fsencode(directory)
    images = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".npy"):
            image = np.load(os.path.join(
                directory, file))
            images.append(image)
    image_array = np.concat(images, axis=0)
    if num_samples != None:
        return image_array[:num_samples]
    return image_array


def calcEtaPhi(num_pixels, widthEta, widthPhi):
    # For Muon project
    # (num_pixels, widthEta, widthPhi) = (32, 0.025, 0.025) if we are sure physical range of image is -0.4, 0.4
    # else (num_pixels, widthEta, widthPhi) = (32, 0.025, pi/126)

    etaEdges = np.linspace(-(num_pixels/2.)*widthEta,
                           (num_pixels/2.)*widthEta, num_pixels+1)
    etagrid = 0.5 * (etaEdges[:-1] + etaEdges[1:])

    phiEdges = np.linspace(-(num_pixels/2.)*widthPhi,
                           (num_pixels/2.)*widthPhi, num_pixels+1)
    phigrid = 0.5 * (phiEdges[:-1] + phiEdges[1:])

    eta = np.tile(etagrid, (num_pixels, 1))
    phi = np.tile(phigrid[::-1].reshape(-1, 1), (1, num_pixels))

    return eta, phi


def discrete_mass_muon(jet_image):
    '''
    Calculates the jet mass from a pixelated jet image
    Args:
    -----
        jet_image: numpy ndarray of dim (1, num_pixels, num_pixels)
    Returns:
    --------
        M: float, jet mass
    '''
    if len(jet_image.shape) == 2:
        image = jet_image[np.newaxis, :]
    else:
        image = jet_image
    # eta, phi = generate_relative_eta_and_phi(np.squeeze(jet_image))
    eta, phi = calcEtaPhi(32, 0.025, 0.025)
    Px = np.sum(image * np.cos(phi), axis=(1, 2))
    Py = np.sum(image * np.sin(phi), axis=(1, 2))
    Pz = np.sum(image * np.sinh(eta), axis=(1, 2))
    E = np.sum(image * np.cosh(eta), axis=(1, 2))
    PT2 = np.square(Px) + np.square(Py)
    M2 = np.square(E) - (PT2 + np.square(Pz))
    M2 = M2.clip(min=0)
    M = np.sqrt(M2)
    return M


def discrete_pt_muon(jet_image):
    '''
    Calculates the jet transverse momentum from a pixelated jet image
    Args:
    -----
        jet_image: numpy ndarray of dim (1, num_pixels, num_pixels)
    Returns:
    --------
        float, jet transverse momentum
    '''
    if len(jet_image.shape) == 2:
        image = jet_image[np.newaxis, :]
    else:
        image = jet_image
    eta, phi = calcEtaPhi(32, 0.025, 0.025)
    Px = np.sum(image * np.cos(phi), axis=(1, 2))
    Py = np.sum(image * np.sin(phi), axis=(1, 2))
    return np.sqrt(np.square(Px) + np.square(Py))


def get_mass_pt_muon(samples):
    mass_values = []
    pt_values = []
    for i in range(samples.shape[0]):
        mass_values.append(float(discrete_mass_muon(samples[i])))
        pt_values.append(float(discrete_pt_muon(samples[i])))
    return mass_values, pt_values


def get_distance_MUON(image, samples):
    image_mean, image_std = image.mean(), image.std(ddof=0)
    image = (image-image_mean) / (image_std+1e-10)
    m0, p0 = get_mass_pt_muon(image)
    m1, p1 = get_mass_pt_muon(samples)

    m0, p0 = np.array(m0), np.array(p0)
    m1, p1 = np.array(m1), np.array(p1)
    pt_dist = wasserstein_distance(p0, p1)
    mass_dist = wasserstein_distance(m0, m1)
    return (pt_dist, mass_dist)


def global_standard_norm(x):
    x_mean, x_std = x.mean(), x.std(ddof=0)
    return (x-x_mean) / (x_std+1e-10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Sparsity')
    parser.add_argument('--dataset_path', type=str,
                        default=None, help='Path to images')
    parser.add_argument('--generated_path', type=str,
                        default=None, help='Path to images')
    parser.add_argument('--num_samples', type=int,
                        default=None, help='Number of samples')
    args = parser.parse_args()

    real_images = np.load(args.dataset_path)
    generated_images = get_calo_images(args.generated_path, args.num_samples)
    assert isinstance(generated_images, np.ndarray)

    metrics = {"Real sparsity": 1 -
               (np.count_nonzero(real_images)/real_images.size),
               "Generated sparsity": 1 -
               (np.count_nonzero(generated_images)/generated_images.size)}

    (pt, mass) = get_distance_MUON(real_images,
                                   generated_images)
    metrics["Pt"] = pt.item()
    metrics["Mass"] = mass.item()
    filename = "metrics.json"
    with open(os.path.join(args.generated_path, filename), "w") as outfile:
        json.dump(metrics, outfile)
