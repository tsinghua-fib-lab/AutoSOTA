import pickle
import torch
import torch.nn.functional as F
from torch.utils import data
from pathlib import Path
import numpy as np
import pandas as pd
from einops import rearrange

def generalized_image_to_b_xy_c(tensor):
    """
    Transpose the tensor from [batch, channels, ..., pixel_x, pixel_y] to [batch, pixel_x*pixel_y, channels, ...]. We assume two pixel dimensions.
    """
    num_dims = len(tensor.shape) - 3  # Subtracting batch and pixel dimensions
    pattern = 'b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y -> b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)])
    return rearrange(tensor, pattern)

def generalized_b_xy_c_to_image(tensor, pixels_x=None, pixels_y=None):
    """
    Transpose the tensor from [batch, pixel_x*pixel_y, channels, ...] to [batch, channels, ..., pixel_x, pixel_y] using einops.
    """
    if pixels_x is None or pixels_y is None:
        pixels_x = pixels_y = int(np.sqrt(tensor.shape[1]))
    num_dims = len(tensor.shape) - 2  # Subtracting batch and pixel dimensions (NOTE that we assume two pixel dimensions that are FLATTENED into one dimension)
    pattern = 'b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)]) + f' -> b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y'
    return rearrange(tensor, pattern, x=pixels_x, y=pixels_y)

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Dataset(data.Dataset):
    def __init__(
        self,
        data_directories,
        use_double = False,
        return_img = True,
        gaussian_prior = False,
    ):
        super().__init__()

        # Assuming data_directories is a tuple of file paths
        self.data_paths = list(data_directories)
        channels = len(self.data_paths)            

        # load data
        for i in range(channels):
            if i == 0:
                self.data = pd.read_csv(self.data_paths[i], header=None)
            else:
                self.data = np.stack((self.data, pd.read_csv(self.data_paths[i], header=None)), axis=-1)

        # convert to torch tensor
        dtype = torch.float64 if use_double else torch.float32
        self.data = torch.tensor(self.data, dtype=dtype)
        self.num_datapoints = len(self.data)

        if return_img:
            assert len(self.data.shape) == 3, "Data must be of shape (num_datapoints, pixels_x*pixels_y, channels)"
            self.data = generalized_b_xy_c_to_image(self.data)

        if gaussian_prior:
            # instead consider no information at all
            self.data = torch.randn_like(self.data)

    def normalize(self, arr, min_val, max_val):
        return (arr - min_val) / (max_val - min_val)

    def unnorm(self, arr, min_val, max_val):
        return arr * (max_val - min_val) + min_val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if index >= self.num_datapoints:
            raise IndexError('index out of range')
        return self.data[index]

class Dataset_Paths(data.Dataset):
    def __init__(
        self,
        data_directories,
        use_double = False,
        return_img = True,
        gaussian_prior = False,
        exts = ['npy'],
    ):
        super().__init__()
        
        # load topo data
        self.paths = [p for ext in exts for p in Path(f'{data_directories}').glob(f'**/*.{ext}')]
        # sort paths by number of name
        self.paths = sorted(self.paths, key=lambda x: int(x.name.split('.')[0]))
        self.num_datapoints = len(self.paths)

        # convert to torch tensor in correct dtype
        self.dtype = torch.float64 if use_double else torch.float32

        self.return_img = return_img
        self.gaussian_prior = gaussian_prior

    def normalize(self, arr, min_val, max_val):
        return (arr - min_val) / (max_val - min_val)

    def unnorm(self, arr, min_val, max_val):
        return arr * (max_val - min_val) + min_val

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        if index >= self.num_datapoints:
            raise IndexError('index out of range')
                             
        data_np = np.load(self.paths[index], allow_pickle = True, encoding = 'latin1')
        data = torch.tensor(data_np.transpose(2, 0, 1), dtype = self.dtype) # vf_arr, strain_energy_density_fem, von_mises_stress, disp_x, disp_y, E_field, BC_node_x, BC_node_y, load_x_img, load_y_img x pixels x pixels
        return data


class DatasetTurbulentPickle(data.Dataset):
    """
    Turbulent channel-flow dataset from ``ch_2Dxysec.pickle``.

    The original data layout is ``[T, X, Y, 1]`` with shape ``[10000, 128, 48, 1]``.
    To stay compatible with the Darcy-style training path in ``last1``, each sample is:

    1. converted to channel-first image layout ``[1, X, Y]``
    2. globally normalized by max-abs value
    3. resized to ``[1, pixels_per_dim, pixels_per_dim]`` (default: ``64x64``)
    """

    def __init__(
        self,
        data_path,
        pixels_per_dim: int = 64,
        split: str = 'train',
        train_fraction: float = 0.9,
        use_double: bool = False,
        normalize_mode: str = 'maxabs',
    ):
        super().__init__()
        if split not in {'train', 'valid'}:
            raise ValueError("split must be 'train' or 'valid'")
        if not (0.0 < train_fraction < 1.0):
            raise ValueError('train_fraction must be in (0, 1).')

        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f'Turbulent dataset not found: {self.data_path}')

        self.pixels_per_dim = int(pixels_per_dim)
        self.split = split
        self.use_double = bool(use_double)
        self.dtype = torch.float64 if self.use_double else torch.float32

        with self.data_path.open('rb') as f:
            data_np = pickle.load(f)

        data_np = np.asarray(data_np, dtype=np.float32)
        if data_np.ndim != 4 or data_np.shape[-1] != 1:
            raise ValueError(
                f'Expected turbulent data shaped [T, X, Y, 1], got {data_np.shape}'
            )

        if normalize_mode == 'maxabs':
            scale = float(np.abs(data_np).max())
            if scale > 0.0:
                data_np = data_np / scale
        elif normalize_mode != 'none':
            raise ValueError("normalize_mode must be 'maxabs' or 'none'")

        data = torch.tensor(data_np, dtype=torch.float32).permute(0, 3, 1, 2)  # [T, 1, X, Y]
        if data.shape[-2:] != (self.pixels_per_dim, self.pixels_per_dim):
            data = F.interpolate(
                data,
                size=(self.pixels_per_dim, self.pixels_per_dim),
                mode='bilinear',
                align_corners=False,
            )

        n_total = data.shape[0]
        n_train = max(1, int(round(n_total * train_fraction)))
        if n_train >= n_total:
            n_train = n_total - 1

        if split == 'train':
            self.data = data[:n_train]
        else:
            self.data = data[n_train:]

        self.data = self.data.to(self.dtype)
        self.num_datapoints = len(self.data)

    def __len__(self):
        return self.num_datapoints

    def __getitem__(self, index):
        if index >= self.num_datapoints:
            raise IndexError('index out of range')
        return self.data[index]

def sample_images_with_squares(no_points, pixels_per_dim, dim, frame_dim = False, use_double = False):

    dtype = np.float64 if use_double else np.float32

    # Define the size of the square (e.g., a quarter of the image dimension)
    square_size = pixels_per_dim // 4

    # initialize an array to store the images
    # shape: (no_points, pixels_per_dim, pixels_per_dim, dim)
    if frame_dim:
        images = np.zeros((no_points, dim, 1, pixels_per_dim, pixels_per_dim), dtype=dtype)
    else:
        images = np.zeros((no_points, dim, pixels_per_dim, pixels_per_dim), dtype=dtype)

    for i in range(no_points):
        # randomly choose the top-left corner of the square
        x_start = np.random.randint(0, pixels_per_dim - square_size)
        y_start = np.random.randint(0, pixels_per_dim - square_size)

        for j in range(dim):
            # draw the square in each channel of the image
            if frame_dim:
                images[i, j, :, x_start:x_start + square_size, y_start:y_start + square_size] = 1.
            else:
                images[i, j, x_start:x_start + square_size, y_start:y_start + square_size] = 1.
                
    return images

class SquareImagesDataset(Dataset):
    def __init__(self, no_points, pixels_per_dim, dim, frame_dim=False, use_double=False):
        """
        Args:
            no_points (int): Number of images to generate.
            pixels_per_dim (int): The size of each image dimension.
            dim (int): Number of channels in the image.
            frame_dim (bool): Whether to include an additional frame dimension.
            use_double (bool): Whether to use double precision.
        """
        self.data = sample_images_with_squares(no_points, pixels_per_dim, dim, frame_dim, use_double)
        self.use_double = use_double

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError('Index out of range')

        # Convert numpy array to PyTorch tensor
        dtype = torch.float64 if self.use_double else torch.float32
        image = torch.tensor(self.data[index], dtype=dtype)

        return image

class Normalization:
    def __init__(self,data,dataType,strategy):
        self.mu = torch.mean(data,dim=0)
        self.std = torch.std(data,dim=0)
        self.min = torch.min(data,dim=0)[0]
        self.max = torch.max(data,dim=0)[0]
        self.globalmin = torch.min(data)
        self.globalmax = torch.max(data)
        self.dataType = dataType
        self.cols = data.size()[1]
        self.strategy = strategy
    
    def normalize(self, data):
        list_index_cat = []       
        temp = torch.zeros(data.shape,device=data.device)
        for i in range(0, self.cols):
            if self.dataType[i] == 'continuous':

                if(self.strategy == 'min-max-1'):
                    #scale to [0,1]
                    temp[:,i] = torch.div(data[:,i]-self.min[i], self.max[i]-self.min[i])

                elif(self.strategy == 'global-min-max-1'):
                    #scale to [-1,1] based on min max of full dataset
                    temp[:,i] = torch.div(data[:,i]-self.globalmin, self.globalmax-self.globalmin)

                elif(self.strategy == 'min-max-2'):
                    #scale to [-1,1]
                    temp[:,i] = 2.*torch.div(data[:,i]-self.min[i], self.max[i]-self.min[i])-1.

                elif(self.strategy == 'global-min-max-2'):
                    #scale to [-1,1] based on min max of full dataset
                    temp[:,i] = 2.*torch.div(data[:,i]-self.globalmin, self.globalmax-self.globalmin)-1.

                elif(self.strategy == 'mean-std'):
                    #scale s.t. mean=0, std=1
                    temp[:,i] = torch.div(data[:,i]-self.mu[i], self.std[i])

                elif (self.strategy == 'none'):
                    temp[:,i] = data[:,i]

                else:
                    raise ValueError('Incorrect normalization strategy')

            elif self.dataType[i] == 'categorical':
                #convert categorical features into binaries and append at the end of feature tensor
                temp = torch.cat((temp,F.one_hot(data[:,i].to(torch.int64))),dim=1)
                list_index_cat = np.append(list_index_cat,i)
                                   
            else:
                raise ValueError("Data type must be either continuous or categorical")

        # delete original (not one-hot encoded) categorical features
        j = 0
        for i in np.array(list_index_cat, dtype=np.int64):          
            temp = torch.cat([temp[:,0:i+j], temp[:,i+1+j:]],dim=1)
            j -= 1

        return temp

    def unnormalize(self, data):
        temp = torch.zeros(data.shape,device=data.device)
        for i in range(0, self.cols):
            if self.dataType[i] == 'continuous':
                
                if(self.strategy == 'min-max-1'):
                    temp[:,i] = torch.mul(data[:,i], self.max[i]-self.min[i]) +self.min[i]

                elif(self.strategy == 'global-min-max-1'):
                    temp[:,i] = torch.mul(data[:,i], self.globalmax-self.globalmin) +self.globalmin

                elif(self.strategy == 'min-max-2'):
                    temp[:,i] = torch.mul(0.5*data[:,i]+0.5, self.max[i]-self.min[i]) +self.min[i]

                elif(self.strategy == 'global-min-max-2'):
                    temp[:,i] = torch.mul(0.5*data[:,i]+0.5, self.globalmax-self.globalmin) +self.globalmin
            
                elif(self.strategy == 'mean-std'):
                    temp[:,i] = torch.mul(data[:,i], self.std[i]) + self.mu[i]

                elif (self.strategy == 'none'):
                    temp[:,i] = data[:,i]

                else:
                    raise ValueError('Incorrect normalization strategy')
                
            elif self.dataType[i] == 'categorical':
                temp[:,i] = data[:,i]

            else:
                raise ValueError("Data type must be either continuous or categorical")
        return temp