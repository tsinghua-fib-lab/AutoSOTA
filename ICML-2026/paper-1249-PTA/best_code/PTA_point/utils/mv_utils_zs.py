from torch_scatter import scatter
import torch.nn as nn
import numpy as np
import torch

TRANS = -1.5

# realistic projection parameters
params = {'maxpoolz':1, 'maxpoolxy':7, 'maxpoolpadz':0, 'maxpoolpadxy':2,
            'convz':1, 'convxy':3, 'convsigmaxy':3, 'convsigmaz':1, 'convpadz':0, 'convpadxy':1,
            'imgbias':0., 'depth_bias':0.2, 'obj_ratio':0.8, 'bg_clr':0.0,
            'resolution': 112, 'depth': 8}


class Grid2Image(nn.Module):
    """A pytorch implementation to turn 3D grid to 2D image. 
       Maxpool: densifying the grid
       Convolution: smoothing via Gaussian
       Maximize: squeezing the depth channel
    """
    def __init__(self):
        super().__init__()
        torch.backends.cudnn.benchmark = False

        self.maxpool = nn.MaxPool3d((params['maxpoolz'], params['maxpoolxy'], params['maxpoolxy']), 
                                    stride=1, padding=(params['maxpoolpadz'], params['maxpoolpadxy'], 
                                    params['maxpoolpadxy']))
        self.conv = torch.nn.Conv3d(1, 1, kernel_size=(params['convz'], params['convxy'], params['convxy']),
                                    stride=1, padding=(params['convpadz'],params['convpadxy'],params['convpadxy']),
                                    bias=True)
        kn3d = get3DGaussianKernel(params['convxy'], params['convz'], sigma=params['convsigmaxy'], zsigma=params['convsigmaz'])
        self.conv.weight.data = torch.Tensor(kn3d).repeat(1,1,1,1,1)
        self.conv.bias.data.fill_(0)
            
    def forward(self, x):
        x = self.maxpool(x.unsqueeze(1))
        x = self.conv(x)
        img = torch.max(x, dim=2)[0]
        img = img / torch.max(torch.max(img, dim=-1)[0], dim=-1)[0][:,:,None,None]
        img = 1 - img
        img = img.repeat(1,3,1,1)
        return img


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """
    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    # zero = torch.zeros([b], requires_grad=False, device=angle.device)[0]
    # one = torch.ones([b], requires_grad=False, device=angle.device)[0]
    zero = z.detach()*0
    one = zero.detach()+1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    # print(rot_mat)
    return rot_mat


def points2grid(points, resolution=params['resolution'], depth=params['depth']):
    """Quantize each point cloud to a 3D grid.
    Args:
        points (torch.tensor): of size [B, _, 3]
    Returns:
        grid (torch.tensor): of size [B * self.num_views, depth, resolution, resolution]
    """
    
    batch, pnum, _ = points.shape

    pmax, pmin = points.max(dim=1)[0], points.min(dim=1)[0]
    pcent = (pmax + pmin) / 2
    pcent = pcent[:, None, :]
    prange = (pmax - pmin).max(dim=-1)[0][:, None, None]
    points = (points - pcent) / prange * 2.
    points[:, :, :2] = points[:, :, :2] * params['obj_ratio']
    
    depth_bias = params['depth_bias']
    _x = (points[:, :, 0] + 1) / 2 * resolution
    _y = (points[:, :, 1] + 1) / 2 * resolution
    _z = ((points[:, :, 2] + 1) / 2 + depth_bias) / (1+depth_bias) * (depth - 2)

    _x.ceil_()
    _y.ceil_()
    z_int = _z.ceil()

    _x = torch.clip(_x, 1, resolution - 2)
    _y = torch.clip(_y, 1, resolution - 2)
    _z = torch.clip(_z, 1, depth - 2)

    coordinates = z_int * resolution * resolution + _y * resolution + _x
    grid = torch.ones([batch, depth, resolution, resolution], device=points.device).view(batch, -1) * params['bg_clr']
    
    grid = scatter(_z, coordinates.long(), dim=1, out=grid, reduce="max")
    grid = grid.reshape((batch, depth, resolution, resolution)).permute((0,1,3,2))

    return grid


class Realistic_Projection:
    """For creating images from PC based on the view information.
    """
    def __init__(self):
        _views = np.asarray([
            [[1 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[3 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[5 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[7 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[1 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[2 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[3 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0, -np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0, np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
            ])
        
        # adding some bias to the view angle to reveal more surface
        _views_bias = np.asarray([
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
            ])

        self.num_views = _views.shape[0]

        angle = torch.tensor(_views[:, 0, :]).float().cuda()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        angle2 = torch.tensor(_views_bias[:, 0, :]).float().cuda()
        self.rot_mat2 = euler2mat(angle2).transpose(1, 2)

        self.translation = torch.tensor(_views[:, 1, :]).float().cuda()
        self.translation = self.translation.unsqueeze(1)

        self.grid2image = Grid2Image().cuda()

    def get_img(self, points):
        b, _, _ = points.shape
        v = self.translation.shape[0]

        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            rot_mat2=self.rot_mat2.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))

        grid = points2grid(points=_points, resolution=params['resolution'], depth=params['depth']).squeeze()
        img = self.grid2image(grid)
        return img

    @staticmethod
    def point_transform(points, rot_mat, rot_mat2, translation):
        """
        :param points: [batch, num_points, 3]
        :param rot_mat: [batch, 3]
        :param rot_mat2: [batch, 3]
        :param translation: [batch, 1, 3]
        :return:
        """
        rot_mat = rot_mat.to(points.device)
        rot_mat2 = rot_mat2.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = torch.matmul(points, rot_mat2)
        points = points - translation
        return points

def get2DGaussianKernel(ksize, sigma=0):
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center)
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel1d[..., None] @ kernel1d[None, ...] 
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()
    return kernel

def get3DGaussianKernel(ksize, depth, sigma=2, zsigma=2):
    kernel2d = get2DGaussianKernel(ksize, sigma)
    zs = (np.arange(depth, dtype=np.float32) - depth//2)
    zkernel = np.exp(-(zs ** 2) / (2 * zsigma ** 2))
    kernel3d = np.repeat(kernel2d[None,:,:], depth, axis=0) * zkernel[:,None, None]
    kernel3d = kernel3d / torch.sum(kernel3d)
    return kernel3d
        