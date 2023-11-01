import numpy as np


class Compose:
    def __init__(self, transforms):
        """
        transforms: List
        """
        self.transforms = transforms

    def __call__(self, pos, x):
        for transform in self.transforms:
            pos, x = transform(pos, x)
        return pos, x


class ColorContrast:
    def __init__(self, p, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor
    
    def __call__(self, pos, x):
        if np.random.rand() < self.p:
            low = x.min(axis=0, keepdims=True)
            high = x.max(axis=0, keepdims=True)
            contrast_x = (x - low) * (255 / (high - low))
            
            blend_factor = np.random.rand() if not self.blend_factor else self.blend_factor
            x = blend_factor * contrast_x + (1 - blend_factor) * x
        return pos, x


class PointCloudScaling:
    def __init__(self, ratio_low, ratio_high, anisotropic=True):
        self.ratio_low = ratio_low
        self.ratio_high = ratio_high
        self.anisotropic = anisotropic
    
    def __call__(self, pos, x):
        scale_ratio = np.random.uniform(self.ratio_low, self.ratio_high, (3 if self.anisotropic else 1, ))
        pos = pos * scale_ratio
        
        return pos, x


class PointCloudRotation_Y:
    def __init__(self, angle, has_normal):
        """
        angle is a float in (0, 1]
        """
        self.angle = angle * np.pi
        self.has_normal = has_normal
    
    def __call__(self, pos, x):
        angle = np.random.uniform(low=(-self.angle), high=self.angle, size=(1,))
        angle_cos = np.cos(angle)
        angle_sin = np.sin(angle)
        
        rotation_matrix = np.array([[angle_cos[0], 0, -angle_sin[0]],
                                    [0, 0, 1],
                                    [angle_sin[0], 0, angle_cos[0]]])
        pos = np.dot(pos, rotation_matrix)
        
        if self.has_normal:
            x[:, 0:3] = np.dot(x[:, 0:3], rotation_matrix)

        return pos, x


class PointCloudRotation_Z:
    def __init__(self, angle):
        """
        angle is a float in (0, 1]
        """
        self.angle = angle * np.pi
    
    def __call__(self, pos, x):
        angle = np.random.uniform(low=(-self.angle), high=self.angle, size=(1,))
        angle_cos = np.cos(angle)
        angle_sin = np.sin(angle)
        
        rotation_matrix = np.array([[angle_cos[0], -angle_sin[0], 0],
                                    [angle_sin[0], angle_cos[0], 0],
                                    [0, 0, 1]])
        pos = np.dot(pos, rotation_matrix)
        
        return pos, x
        

class PointCloudFloorCentering:
    def __init__(self):
        pass
    
    def __call__(self, pos, x):
        pos = pos - pos.mean(axis=0, keepdims=True)
        pos[:, 2] = pos[:, 2] - pos[:, 2].min()
        
        return pos, x


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
    
    def __call__(self, pos, x):
        noise = np.clip(np.random.randn(len(pos), 3) * self.sigma, -self.clip, self.clip)
        pos = pos + noise
        
        return pos, x
    

class ColorDrop:
    def __init__(self, p):
        self.p = p
    
    def __call__(self, pos, x):
        if np.random.rand() < self.p:
            x[:, :] = 0
        return pos, x


class ColorNormalize:
    def __init__(self, mean=[0.5136457, 0.49523646, 0.44921124], std=[0.18308958, 0.18415008, 0.19252081]):
        self.mean = mean
        self.std = std
    
    def __call__(self, pos, x):
        x = x / 255
        x = (x - self.mean) / self.std
        
        return pos, x


class NormalDrop:
    def __init__(self, p):
        self.p = p
        
    def __call__(self, pos, x):
        if np.random.rand() < self.p:
            x[:, 0:3] = 0
        return pos, x
