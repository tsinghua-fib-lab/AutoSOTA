"""
Mesh-based Functions.

"""
import torch


def triangle_area(mesh):
    """
    Compute area of each triangle in a mesh.
    
    Uses the cross product formula: Area = 0.5 * ||v1|| * ||h|| where (v1, v2) are two vectors defining the triangle and h is the height from v1 to v2.
    
    Args:
        mesh: Dictionary containing:
            - 'pos': Vertex positions, torch.Tensor of shape (N, 3)
            - 'face': Triangle indices, torch.Tensor of shape (F, 3)
    
    Returns:
        torch.Tensor: Area of each triangle, shape (F,)
    """
    pos = mesh['pos']  # Size (M, D)
    face = mesh['face']  # Size (F, 3)

    faces_pos = pos[face]  # Size (F, 3, 3) -- position of the triangle vertices.

    v1 = faces_pos[:, 1, :] - faces_pos[:, 0, :]  # Size (F, 3)
    v2 = faces_pos[:, 2, :] - faces_pos[:, 0, :]  # Size (F, 3)

    h = v2 - (v1 * v2).sum(dim=-1, keepdim=True) * v1 / torch.norm(v1, dim=-1, keepdim=True) ** 2  # Size (F, 3)

    return (torch.norm(h, dim=-1) * torch.norm(v1, dim=-1)) / 2  # Size (F,)


def lumped_mass(mesh):
    """
    Compute lumped mass matrix for mesh vertices.
    
    Distributes each triangle's area equally among its three vertices.
    
    Args:
        mesh: Dictionary containing:
            - 'pos': Vertex positions, torch.Tensor of shape (N, 3)
            - 'face': Triangle indices, torch.Tensor of shape (F, 3)
    
    Returns:
        torch.Tensor: Mass associated with each vertex, shape (N,)
    """
    pos = mesh['pos']  # Size (M, D)
    face = mesh['face']  # Size (F, 3)

    areas = triangle_area(mesh)

    M = torch.zeros(pos.shape[0], dtype=torch.float32, device=pos.device)  # Size (M,)
    M.scatter_add_(dim=0, index=face.flatten(), src=areas.repeat(3)) / 3.

    return M
