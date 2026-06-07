'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
import torch
import numpy as np


class Transforms(object):

    def __init__(self,
                 normalized_image_range=[0, 255],
                 random_flip_type=['none'],
                 random_remove_points=[0.70, 0.70],
                 random_noise_type=['none'],
                 random_noise_spread=-1):
        '''
        Transforms and augmentation class

        Arg(s):
            normalized_image_range : list[float]
                intensity range after normalizing images
            random_flip_type : list[str]
                none, horizontal, vertical
            random_remove_points : list[float]
                percentage of points to remove in range map
            random_noise_type : str
                type of noise to add: gaussian, uniform
            random_noise_spread : float
                if gaussian, then standard deviation; if uniform, then min-max range
        '''

        # Image normalization
        self.normalized_image_range = normalized_image_range

        # Geometric augmentations
        self.do_random_horizontal_flip = True if 'horizontal' in random_flip_type else False
        self.do_random_vertical_flip = True if 'vertical' in random_flip_type else False

        self.do_random_remove_points = True if -1 not in random_remove_points else False
        self.remove_points_range = random_remove_points

        self.do_random_noise = \
            True if (random_noise_type != 'none' and random_noise_spread > 0) else False

        self.random_noise_type = random_noise_type
        self.random_noise_spread = random_noise_spread

    def transform(self,
                  images_arr,
                  range_maps_arr=[],
                  validity_maps_arr=[],
                  random_transform_probability=0.50):
        '''
        Applies transform to images and ground truth

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            range_maps_arr : list[torch.Tensor]
                list of N x c x H x W tensors
            validity_maps_arr : list[torch.Tensor]
                list of N x c x H x W tensors
            random_transform_probability : float
                probability to perform transform
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
            list[torch.Tensor[float32]] : list of transformed N x c x H x W range maps tensors
        '''

        device = images_arr[0].device

        n_dim = images_arr[0].ndim

        if n_dim == 4:
            n_batch, _, n_height, n_width = images_arr[0].shape
        elif n_dim == 5:
            n_batch, _, _, n_height, n_width = images_arr[0].shape
        else:
            raise ValueError('Unsupported number of dimensions: {}'.format(n_dim))

        do_random_transform = \
            np.random.rand(n_batch) <= random_transform_probability

        # Normalize images to a given range
        images_arr = self.normalize_images(
            images_arr,
            normalized_image_range=self.normalized_image_range)

        if self.do_random_horizontal_flip:

            do_horizontal_flip = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            images_arr = self.horizontal_flip(
                images_arr,
                do_horizontal_flip)

            range_maps_arr = self.horizontal_flip(
                range_maps_arr,
                do_horizontal_flip)

            validity_maps_arr = self.horizontal_flip(
                validity_maps_arr,
                do_horizontal_flip)

        if self.do_random_vertical_flip:

            do_vertical_flip = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            images_arr = self.vertical_flip(
                images_arr,
                do_vertical_flip)

            range_maps_arr = self.vertical_flip(
                range_maps_arr,
                do_vertical_flip)

            validity_maps_arr = self.vertical_flip(
                validity_maps_arr,
                do_vertical_flip)

        if self.do_random_remove_points:

            do_remove_points = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            values = torch.rand(n_batch, device=device)

            remove_points_min, remove_points_max = self.remove_points_range

            densities = \
                (remove_points_max - remove_points_min) * values + remove_points_min

            range_maps_arr = self.remove_random_nonzero(
                images_arr=range_maps_arr,
                do_remove=do_remove_points,
                densities=densities)

        if self.do_random_noise:

            do_add_noise = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            range_maps_arr = self.add_noise(
                range_maps_arr,
                do_add_noise=do_add_noise,
                noise_type=self.random_noise_type,
                noise_spread=self.random_noise_spread)

        # Return the transformed inputs
        outputs = []

        if len(images_arr) > 0:
            outputs.append(images_arr)

        if len(range_maps_arr) > 0:
            outputs.append(range_maps_arr)

        if len(validity_maps_arr) > 0:
            outputs.append(validity_maps_arr)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    '''
    Photometric transforms
    '''
    def normalize_images(self, images_arr, normalized_image_range=[0, 1]):
        '''
        Normalize image to a given range

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            normalized_image_range : list[float]
                intensity range after normalizing images
        Returns:
            images_arr[torch.Tensor[float32]] : list of normalized N x C x H x W tensors
        '''

        if normalized_image_range == [0, 1]:
            images_arr = [
                images / 255.0 for images in images_arr
            ]
        elif normalized_image_range == [-1, 1]:
            images_arr = [
                2.0 * (images / 255.0) - 1.0 for images in images_arr
            ]
        elif normalized_image_range == [0, 255]:
            pass
        else:
            raise ValueError('Unsupported normalization range: {}'.format(
                normalized_image_range))

        return images_arr

    '''
    Geometric transforms
    '''
    def horizontal_flip(self, images_arr, do_horizontal_flip):
        '''
        Perform horizontal flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_horizontal_flip : bool
                N booleans to determine if horizontal flip is performed on each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_horizontal_flip[b]:
                    images[b, ...] = torch.flip(image, dims=[-1])

            images_arr[i] = images

        return images_arr

    def vertical_flip(self, images_arr, do_vertical_flip):
        '''
        Perform vertical flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_vertical_flip : bool
                N booleans to determine if vertical flip is performed on each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_vertical_flip[b]:
                    images[b, ...] = torch.flip(image, dims=[-2])

            images_arr[i] = images

        return images_arr

    def remove_random_nonzero(self, images_arr, do_remove, densities):
        '''
        Remove random nonzero for each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_remove : bool
                N booleans to determine if random remove is performed on each sample
            densities : float
                N floats to determine how much to remove from each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_remove[b]:

                    nonzero_indices = self.random_nonzero(image, density=densities[b])
                    image[nonzero_indices] = 0.0

                    images[b, ...] = image

            images_arr[i] = images

        return images_arr

    def random_nonzero(self, T, density=0.10):
        '''
        Randomly selects nonzero elements

        Arg(s):
            T : torch.Tensor[float32]
                N x C x H x W tensor
            density : float
                percentage of nonzero elements to select
        Returns:
            list[tuple[torch.Tensor[float32]]] : list of tuples of indices
        '''

        # Find all nonzero indices
        nonzero_indices = (T > 0).nonzero(as_tuple=True)

        # Randomly choose a subset of the indices
        random_subset = torch.randperm(nonzero_indices[0].shape[0], device=T.device)
        random_subset = random_subset[0:int(density * random_subset.shape[0])]

        random_nonzero_indices = [
            indices[random_subset] for indices in nonzero_indices
        ]

        return random_nonzero_indices

    def add_noise(self, images_arr, do_add_noise, noise_type, noise_spread):
        '''
        Add noise to images

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_add_noise : bool
                N booleans to determine if noise will be added
            noise_type : str
                gaussian, uniform
            noise_spread : float
                if gaussian, then standard deviation; if uniform, then min-max range
        '''

        for i, images in enumerate(images_arr):
            device = images.device

            for b, image in enumerate(images):
                if do_add_noise[b]:

                    shape = image.shape
                    validity_map = torch.where(
                        image > 0,
                        torch.ones_like(image),
                        torch.zeros_like(image))

                    if noise_type == 'gaussian':
                        image = image + noise_spread * torch.randn(*shape, device=device)
                    elif noise_type == 'uniform':
                        image = image + noise_spread * (torch.rand(*shape, device=device) - 0.5)
                    else:
                        raise ValueError('Unsupported noise type: {}'.format(noise_type))

                    images[b, ...] = image * validity_map

            images_arr[i] = images

        return images_arr
