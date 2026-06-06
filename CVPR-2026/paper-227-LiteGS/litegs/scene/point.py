import torch
from simple_knn._C import distCUDA2
from ..utils import rgb_to_sh0
from . import cluster

@torch.no_grad()
def create_gaussians(xyz:torch.Tensor,color:torch.Tensor,sh_degree:int):
    dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
    xyz=xyz.transpose(0,1)
    color=color.transpose(0,1)

    sh_0 = rgb_to_sh0(color).unsqueeze(0)
    sh_rest = torch.zeros(((sh_degree + 1) ** 2 - 1, 3, xyz.shape[-1]),dtype=torch.float32,device=xyz.device)
    scale = torch.log(torch.sqrt(dist2)).unsqueeze(0).repeat(3, 1)
    rot = torch.zeros((4,xyz.shape[-1]),dtype=torch.float32,device=xyz.device)
    rot[0,:] = 1
    temp=0.1 * torch.ones((1,xyz.shape[-1]), dtype=torch.float,device=xyz.device)
    opacity = torch.log(temp/(1-temp))#inverse_sigmoid
    return xyz,scale,rot,sh_0,sh_rest,opacity

@torch.no_grad()
def create_gaussians_random(self,sh_degree:int):
    #todo
    return

@torch.no_grad()
def _gen_morton_code(positions: torch.Tensor, bits: int = 21) -> torch.Tensor:
    """
    Generate 3D Morton (Z-order) codes for a set of 3D points using PyTorch.

    Args:
        positions (torch.Tensor): A tensor of shape (3, N), where each row is (x, y, z).
        bits (int): Number of bits used to quantize each coordinate. Default is 10.
                    The final Morton code will use 3 * bits bits in total.

    Returns:
        torch.Tensor: A 1D tensor of size N (dtype=torch.long), where each element
                      is the Morton code corresponding to the input point.
    
    Steps:
        1. Compute the bounding box (min and max) of all points.
        2. Normalize coordinates into [0, 2^bits - 1].
        3. Convert to long type and clamp to avoid out-of-range values.
        4. Interleave bits (bit by bit) from X, Y, and Z to form the final Morton code.
    """
    assert positions.dim() == 2 and positions.shape[0] == 3, "positions must be a (3, N) tensor."

    # 1. Get min and max across each dimension
    min_vals = positions.min(dim=1,keepdim=True).values
    max_vals = positions.max(dim=1,keepdim=True).values
    scale = (2 ** bits) - 1

    # Avoid division by zero by clamping the denominator
    denom = (max_vals - min_vals).clamp_min(1e-12)

    # 2. Normalize positions into [0, 2^bits - 1]
    normalized = ((positions - min_vals) / denom) * scale

    # 3. Convert to long and clamp
    X = normalized[0].long().clamp_(0, scale)
    Y = normalized[1].long().clamp_(0, scale)
    Z = normalized[2].long().clamp_(0, scale)

    # 4. Interleave bits
    codes = torch.zeros_like(X, dtype=torch.long)  # Output Morton codes
    for i in range(bits):
        # Extract the i-th bit from X, Y, and Z
        x_i = (X >> i) & 1
        y_i = (Y >> i) & 1
        z_i = (Z >> i) & 1
        
        # Place these bits into (3*i), (3*i+1), and (3*i+2) of the final code
        codes |= (x_i << (3 * i)) | (y_i << (3 * i + 1)) | (z_i << (3 * i + 2))

    return codes

def get_morton_sorted_indices(xyz:torch.Tensor):
    '''
    TODO
    '''
    morton_code=_gen_morton_code(xyz[:,:3])
    _,indices=morton_code.sort()
    return indices

@torch.no_grad()
def spatial_refine(bClustered:bool,optimizer:torch.optim.Optimizer,xyz:torch.Tensor,*args)->list[torch.Tensor]:
    if bClustered:
        chunk_size=xyz.shape[-1]
        xyz,=cluster.uncluster(xyz)
        
    morton_code=_gen_morton_code(xyz)
    _,indices=morton_code.sort(stable=True)

    if optimizer is None:
        #tensor
        if bClustered:
            args=cluster.uncluster(args)
        refined_xyz=xyz[...,indices]
        refined_tensors=[refined_xyz,]
        for tensor in args:
            refined_tensors.append(tensor[...,indices])
        if bClustered:
            refined_tensors=cluster.cluster_points(chunk_size,refined_tensors)
        return *refined_tensors,
    else:
        #optimizer
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                #parameters
                if param.data is None:
                    continue
                if bClustered:
                    param_data, = cluster.uncluster(param.data)
                else:
                    param_data = param.data
                refined_data = param_data[..., indices]
                if bClustered:
                    refined_data, = cluster.cluster_points(chunk_size,refined_data)
                param.copy_(refined_data)
                #grads
                if param.grad is not None:
                        if bClustered:
                            grad_data, = cluster.uncluster(param.grad.data)
                        else:
                            grad_data = param.grad.data
                        refined_grad = grad_data[..., indices]
                        if bClustered:
                            refined_grad, = cluster.cluster_points(chunk_size,refined_grad)
                        param.grad.data=refined_grad
                #state
                state_dict = optimizer.state[param]
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor) and value.shape == param.data.shape:
                        if bClustered:
                            unclustered_value, = cluster.uncluster(value)
                        else:
                            unclustered_value = value
                        refined_value = unclustered_value[..., indices]
                        if bClustered:
                            refined_value, = cluster.cluster_points(chunk_size,refined_value)
                        value.data=refined_value
    return None
