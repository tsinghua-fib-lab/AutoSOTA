import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter

class Augmentor:
    def __init__(self, sparse, aug_params):
        self.sparse = sparse
        self.aug_params = aug_params
        self.color_jitter = None
        self.scale_prob = 0

        if self.aug_params.color_jitter:
            self.color_jitter = ColorJitter(brightness=list(aug_params.color_jitter.brightness), contrast=list(aug_params.color_jitter.contrast), saturation=list(aug_params.color_jitter.saturation), hue=aug_params.color_jitter.hue / 3.14)

        if self.aug_params.random_scale:
            self.scale_prob = self.aug_params.random_scale.scale_prob

    def dilate_transform(self, disp):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        disp = cv2.dilate(disp, kernel, iterations=1)

        return disp

    def color_transform(self, *images):
        if np.random.rand() < self.aug_params.color_jitter.asymmetric_prob:
            images = [np.array(self.color_jitter(Image.fromarray(img.astype(np.uint8))), dtype=np.float32) for img in images]
        else:
            image_stack = np.concatenate(images, axis=0).astype(np.uint8)
            image_stack = np.array(self.color_jitter(Image.fromarray(image_stack)), dtype=np.float32)
            images = np.split(image_stack, len(images), axis=0)

        return images

    def erase_transform(self, image):
        ht, wd = image.shape[:2]

        if np.random.rand() < self.aug_params.random_erase.prob:
            mean_color = np.mean(image.reshape(-1, 3), axis=0)

            for _ in range(np.random.randint(1, self.aug_params.random_erase.max_time + 1)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(self.aug_params.random_erase.bound[0], self.aug_params.random_erase.bound[1])
                dy = np.random.randint(self.aug_params.random_erase.bound[0], self.aug_params.random_erase.bound[1])
                image[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return image

    def resize_sparse_disp_map(self, disp, valid, fx, fy):
        ht, wd = disp.shape[:2]
        ht1 = int(np.round(ht * fy))
        wd1 = int(np.round(wd * fx))
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)
        coords = coords.reshape(-1, 2).astype(np.float32)

        disp = disp.reshape(-1)
        valid = valid.reshape(-1)
        coords = coords[valid >= 0.5]
        disp = disp[valid >= 0.5]
        coords = coords * [fx, fy]
        disp = disp * fx

        xx = np.round(coords[:, 0]).astype(np.int32)
        yy = np.round(coords[:, 1]).astype(np.int32)
        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        disp = disp[v]

        resized_disp = np.zeros([ht1, wd1], dtype=np.float32)
        resized_valid = np.zeros([ht1, wd1], dtype=np.float32)
        resized_disp[yy, xx] = disp
        resized_valid[yy, xx] = 1

        return resized_disp, resized_valid

    def spatial_transform(self, images):
        ht, wd = images['left'].shape[:2]
        min_scale = np.maximum((self.aug_params.crop_size[0] + 1) / ht, (self.aug_params.crop_size[1] + 1) / wd)
        scale_x = min_scale
        scale_y = min_scale

        if self.aug_params.random_scale:
            scale = 2 ** np.random.uniform(self.aug_params.random_scale.min_scale, self.aug_params.random_scale.max_scale)
            scale_x = scale
            scale_y = scale

            if (not self.sparse) and (np.random.rand() < self.aug_params.random_scale.stretch_prob):
                scale_x *= 2 ** np.random.uniform(-self.aug_params.random_scale.max_stretch, self.aug_params.random_scale.max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.aug_params.random_scale.max_stretch, self.aug_params.random_scale.max_stretch)
            
            scale_x = np.clip(scale_x, min_scale, None)
            scale_y = np.clip(scale_y, min_scale, None)

        if (np.random.rand() < self.scale_prob) or (min_scale > 1):
            images['left'] = cv2.resize(images['left'], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            images['right'] = cv2.resize(images['right'], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            if self.sparse:
                images['disp'], images['valid'] = self.resize_sparse_disp_map(images['disp'], images['valid'], scale_x, scale_y)
            else:
                images['disp'] = cv2.resize(images['disp'], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR) * scale_x
                images['valid'] = cv2.resize(images['valid'], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)

        if (not self.sparse) and self.aug_params.y_jitter:
            max_jitter = (images['left'].shape[0] - self.aug_params.crop_size[0]) // 2
            y_jitter = np.minimum(2, max_jitter)

            y0 = np.random.randint(y_jitter, images['left'].shape[0] - self.aug_params.crop_size[0] - y_jitter + 1)
            x0 = np.random.randint(0, images['left'].shape[1] - self.aug_params.crop_size[1] + 1)
            y1 = y0 + np.random.randint(-y_jitter, y_jitter + 1)

            images['right'] = images['right'][y1:y1 + self.aug_params.crop_size[0], x0:x0 + self.aug_params.crop_size[1]]
        else:
            y0 = np.random.randint(0, images['left'].shape[0] - self.aug_params.crop_size[0] + 1)
            x0 = np.random.randint(0, images['left'].shape[1] - self.aug_params.crop_size[1] + 1)

            images['right'] = images['right'][y0:y0 + self.aug_params.crop_size[0], x0:x0 + self.aug_params.crop_size[1]]

        for key in images:
            if key not in ['right']:
                images[key] = images[key][y0:y0 + self.aug_params.crop_size[0], x0:x0 + self.aug_params.crop_size[1]]

        return images

    def __call__(self, name, **images):
        # Dilate SceneFlow Driving subset's disparity to simulate real LiDAR
        if 'mm_focallength' in name:
            images['disp'] = self.dilate_transform(images['disp'])

        if self.color_jitter:
            images['left'], images['right'] = self.color_transform(images['left'], images['right'])

        if self.aug_params.random_erase:
            images['right'] = self.erase_transform(images['right'])

        images = self.spatial_transform(images)

        for key in images:
            images[key] = np.ascontiguousarray(images[key])

        return tuple(images.values())