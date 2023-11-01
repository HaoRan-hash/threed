import numpy as np
import scipy
import random
import torch


class Compose:
    def __init__(self, transforms):
        """
        transforms: List
        """
        self.transforms = transforms

    def __call__(self, pos, color, normal):
        for transform in self.transforms:
            pos, color, normal = transform(pos, color, normal)
        return pos, color, normal


class ColorContrast:
    def __init__(self, p, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor
    
    def __call__(self, pos, color, normal):
        if np.random.rand() < self.p:
            low = color.min(axis=0, keepdims=True)
            high = color.max(axis=0, keepdims=True)
            contrast_color = (color - low) * (255 / (high - low))
            
            blend_factor = np.random.rand() if not self.blend_factor else self.blend_factor
            color = blend_factor * contrast_color + (1 - blend_factor) * color
        return pos, color, normal


class PointCloudScaling:
    def __init__(self, ratio_low, ratio_high, anisotropic=True, mirror=[-1, -1, -1]):
        """
        mirror: the possibility of mirroring. set to a negative value to not mirror
        """
        self.ratio_low = ratio_low
        self.ratio_high = ratio_high
        self.anisotropic = anisotropic
        self.mirror = np.array(mirror)
        self.use_mirroring = np.sum(self.mirror > 0) != 0
    
    def __call__(self, pos, color, normal):
        scale_ratio = np.random.uniform(self.ratio_low, self.ratio_high, (3 if self.anisotropic else 1, ))
        if self.use_mirroring:
            mirror = (np.random.rand(3) > self.mirror).astype(np.float32) * 2 - 1
            scale_ratio = scale_ratio * mirror
        pos = pos * scale_ratio
        
        return pos, color, normal


class PointCloudRotation_Y:
    def __init__(self, angle, has_normal):
        """
        angle is a float in (0, 1]
        """
        self.angle = angle * np.pi
        self.has_normal = has_normal
    
    def __call__(self, pos, color, normal):
        angle = np.random.uniform(low=(-self.angle), high=self.angle, size=(1,))
        angle_cos = np.cos(angle)
        angle_sin = np.sin(angle)
        
        rotation_matrix = np.array([[angle_cos[0], 0, -angle_sin[0]],
                                    [0, 1, 0],
                                    [angle_sin[0], 0, angle_cos[0]]])
        pos = np.dot(pos, rotation_matrix)
        
        if self.has_normal:
            normal = np.dot(normal, rotation_matrix)

        return pos, color, normal


class PointCloudRotation_Z:
    def __init__(self, angle, has_normal):
        """
        angle is a float in (0, 1]
        """
        self.angle = angle * np.pi
        self.has_normal = has_normal
    
    def __call__(self, pos, color, normal):
        angle = np.random.uniform(low=(-self.angle), high=self.angle, size=(1,))
        angle_cos = np.cos(angle)
        angle_sin = np.sin(angle)
        
        rotation_matrix = np.array([[angle_cos[0], -angle_sin[0], 0],
                                    [angle_sin[0], angle_cos[0], 0],
                                    [0, 0, 1]])
        pos = np.dot(pos, rotation_matrix)
        
        if self.has_normal:
            normal = np.dot(normal, rotation_matrix)
        
        return pos, color, normal
        

class PointCloudFloorCentering:
    def __init__(self):
        pass
    
    def __call__(self, pos, color, normal):
        pos = pos - pos.mean(axis=0, keepdims=True)
        pos[:, 2] = pos[:, 2] - pos[:, 2].min()
        
        return pos, color, normal


class PointCloudCenterAndNormalize:
    def __init__(self):
        pass
    
    def __call__(self, pos, x):
        # height append
        heights = pos[:, 1] - pos[:, 1].min()
        heights = np.expand_dims(heights, axis=1)
        x = np.concatenate((x, heights), axis=1)
        
        # center and normalize
        pos = pos - pos.mean(axis=0)
        max_dis = (np.sqrt((np.square(pos)).sum(axis=1))).max()
        pos = pos / max_dis
        
        return pos, x
        

class PointCloudJitter:
    def __init__(self, sigma, clip):
        self.sigma = sigma
        self.clip = clip
    
    def __call__(self, pos, color, normal):
        noise = np.clip(np.random.randn(len(pos), 3) * self.sigma, -self.clip, self.clip)
        pos = pos + noise
        
        return pos, color, normal
    

class ColorDrop:
    def __init__(self, p):
        self.p = p
    
    def __call__(self, pos, color, normal):
        if np.random.rand() < self.p:
            color[:, :] = 0
        return pos, color, normal


class ColorNormalize:
    def __init__(self, mean=[0.5136457, 0.49523646, 0.44921124], std=[0.18308958, 0.18415008, 0.19252081]):
        self.mean = mean
        self.std = std
    
    def __call__(self, pos, color, normal):
        color = color / 255
        color = (color - self.mean) / self.std
        
        return pos, color, normal


class NormalDrop:
    def __init__(self, p):
        self.p = p
        
    def __call__(self, pos, color, normal):
        if np.random.rand() < self.p:
            normal[:, :] = 0
        return pos, color, normal


class ElasticDistortion():
    def __init__(self, distortion_params=None):
        self.distortion_params = [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                       (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=False, fill_value=0)
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, pos, color, normal):
        if random.random() < 0.95:
            for granularity, magnitude in self.distortion_params:
                pos = self.elastic_distortion(pos, granularity, magnitude)
        return pos, color, normal
