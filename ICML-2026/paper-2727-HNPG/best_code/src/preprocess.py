from sklearn.decomposition import PCA
import numpy as np


def image_pca_and_rescale(images, pca_dim, R):
    
    X_red = PCA(n_components=pca_dim).fit_transform(images)
    X_red = X_red / (np.linalg.norm(X_red, axis=1).max() + 1e-12) * R

    return X_red

def to_hyperboloid(geometry, points):

    dim = points.shape[-1]

    p0 = np.zeros(dim + 1)
    p0[0] = 1.0
    tangent_vecs = np.concatenate([np.zeros((len(points), 1)), points], axis=1)
    points = geometry.metric.exp(tangent_vec=tangent_vecs, base_point=p0)
    return points


def add_noise(geometry, noise_sigma, points, rng):

    ambient_noise = rng.normal(0, noise_sigma, size=points.shape)
    tangent_noise = geometry.to_tangent(ambient_noise, base_point=points)

    query_on_manifold = geometry.metric.exp(tangent_vec=tangent_noise, base_point=points)

    return query_on_manifold


def add_noise_euclidean(noise_sigma, points, rng):
    ambient_noise = rng.normal(0, noise_sigma, size=points.shape)
    return ambient_noise + points