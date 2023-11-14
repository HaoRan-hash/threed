import torch
import numpy as np
import open3d as o3d


class PointCloudFloorCenteringTensor:
    def __init__(self):
        pass
    
    def __call__(self, pos, x):
        pos = pos - pos.mean(dim=1, keepdim=True)
        pos[:, :, 2] = pos[:, :, 2] - pos[:, :, 2].min()
        
        return pos, x


class ColorNormalizeTensor:
    def __init__(self, mean=[0.5136457, 0.49523646, 0.44921124], std=[0.18308958, 0.18415008, 0.19252081]):
        self.mean = mean
        self.std = std
    
    def __call__(self, pos, color, normal):
        color = color / 255
        color = (color - torch.tensor(self.mean, device=color.device)) / torch.tensor(self.std, device=color.device)
        
        return pos, color, normal


class PointCloudScalingBatch:
    def __init__(self, ratio_low, ratio_high, anisotropic=True):
        self.ratio_low = ratio_low
        self.ratio_high = ratio_high
        self.anisotropic = anisotropic
    
    def __call__(self, pos, color, normal):
        scale_ratio = np.random.uniform(self.ratio_low, self.ratio_high, (pos.shape[0], 3 if self.anisotropic else 1))
        scale_ratio = torch.as_tensor(scale_ratio, device=pos.device, dtype=torch.float32).unsqueeze(dim=1)
        pos = pos * scale_ratio
        
        return pos, color, normal


def get_normal(pos):
    b, n, _ = pos.shape
    pos = pos.to(device='cpu', dtype=torch.float64).numpy()
    res = np.zeros((b, n, 3), dtype=np.float64)
    
    for i in range(b):
        temp = pos[i]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(temp)   # 必须是float64
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(32))
        res[i] = pcd.normals
    
    res = torch.as_tensor(res.astype(np.float32))
    return res
