import torch
import phf_cuda
import time
import logging


def phf(points):
    masks = [(points[..., 0]<=0) & (points[..., 1]<=0) & (points[..., 2]<=0),
             (points[..., 0]<=0) & (points[..., 1]<=0) & (points[..., 2]>0),
             (points[..., 0]<=0) & (points[..., 1]>0) & (points[..., 2]<=0),
             (points[..., 0]<=0) & (points[..., 1]>0) & (points[..., 2]>0),
             (points[..., 0]>0) & (points[..., 1]<=0) & (points[..., 2]<=0),
             (points[..., 0]>0) & (points[..., 1]<=0) & (points[..., 2]>0),
             (points[..., 0]>0) & (points[..., 1]>0) & (points[..., 2]<=0),
             (points[..., 0]>0) & (points[..., 1]>0) & (points[..., 2]>0)]
    masks = torch.stack(masks).float()
    masks = masks.permute(1, 2, 3, 0)
    return masks


def phf2(points):
    b, n, k, _ = points.shape
    device = points.device
    masks = torch.zeros((b, n, k, 8), device=device, dtype=torch.float32)
    
    phf_cuda.phf(points, masks)
    return masks


if __name__ == '__main__':
    b, n, k = 8, 24000, 32
    device = 'cuda:5'
    centroids = torch.randn((b, n, 3), device=device)
    neighbor = torch.randn((b, n, k, 3), device=device)
    
    points = neighbor - centroids.unsqueeze(dim=2)
    points = points.to(torch.float16)
    
    res = phf2(points)
    
    # 验证一致性
    # temp = (res1 - res2)
    # print(temp.sum())
    