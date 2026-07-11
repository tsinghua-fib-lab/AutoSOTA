import numpy as np
from skimage import measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

def sample_foreground_points(mask):
    # distance + peaks
    dist = distance_transform_edt(mask)
    coords = peak_local_max(dist, min_distance=5, labels=mask)
    markers = np.zeros_like(dist, int)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i

    # watershed
    labels = watershed(-dist, markers, mask=mask)
    props = measure.regionprops(labels)
    centroids = np.array([prop.centroid for prop in props])

    return centroids

def sample_background_points(mask, grid_size=24):
    h, w = mask.shape
    coords = []
    for i in range(0, h, grid_size):
        for j in range(0, w, grid_size):
            sub_mask = mask[i:i+grid_size, j:j+grid_size]
            if np.any(sub_mask):
                sub_coords = np.column_stack(np.nonzero(sub_mask))
                sub_coords[:, 0] += i
                sub_coords[:, 1] += j
                chosen = sub_coords[np.random.choice(len(sub_coords))]
                coords.append(chosen)
    return np.array(coords)
    

def merge_points(points, threshold=10):
    if len(points) == 0:
        return points

    tree = cKDTree(points)
    clusters = tree.query_ball_tree(tree, r=threshold)
    visited = set()
    merged = []

    for i, neighbors in enumerate(clusters):
        if i in visited:
            continue
        group = set(neighbors)
        visited |= group
        pts = points[list(group)]
        merged.append(pts.mean(axis=0))
    
    return np.vstack(merged)

