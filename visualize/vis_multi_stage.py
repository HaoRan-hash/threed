import torch
import fps_cuda
from utils_func import index_points, index_gts
import numpy as np


def fps(points, nsamples):
        """
        points.shape = (b, n, 3)
        return indices.shape = (b, nsamples)
        """
        b, n, _ = points.shape
        device = points.device
        dis = torch.ones((b, n), device=device) * 1e10
        indices = torch.zeros((b, nsamples), device=device, dtype=torch.long)

        fps_cuda.fps(points, dis, indices)
        return indices


def gen_color(y):
    """
    y.shape = (n,) cuda
    """
    y = y.to('cpu').squeeze()
    color_map = [[255, 0, 0], [0, 0, 205],
                 [255, 0, 255], [0, 255, 0],
                 [255, 255, 0], [153, 50, 204]]
    color_map = torch.as_tensor(color_map, dtype=torch.float32)
    res = torch.zeros((len(y), 3))
    for i in range(6):
        mask = (y == i)
        res[mask] = color_map[i]
    return res


orange_color = [255, 165, 0]
blue_color = [0, 191, 255]
purple_color = [155, 48, 255]
def gen_multi_stage(pos, gt, file_path):
    """
    pos.shape = (1, n, 3) cuda
    gt.shape = (1, n) cuda
    """
    for i in range(3):
        gt_color = gen_color(gt)
        with open(f'{file_path}/{i}_blue.txt', mode='w') as f1, open(f'{file_path}/{i}_orange.txt', mode='w') as f2, open(f'{file_path}/{i}_purple.txt', mode='w') as f3, open(f'{file_path}/{i}_gt.txt', mode='w') as f4:
            for j in range(pos.shape[1]):
                f1.write(f'{pos[0, j, 0]}, {pos[0, j, 1]}, {pos[0, j, 2]}, {blue_color[0]}, {blue_color[1]}, {blue_color[2]}\n')
                f2.write(f'{pos[0, j, 0]}, {pos[0, j, 1]}, {pos[0, j, 2]}, {orange_color[0]}, {orange_color[1]}, {orange_color[2]}\n')
                f3.write(f'{pos[0, j, 0]}, {pos[0, j, 1]}, {pos[0, j, 2]}, {purple_color[0]}, {purple_color[1]}, {purple_color[2]}\n')
                f4.write(f'{pos[0, j, 0]}, {pos[0, j, 1]}, {pos[0, j, 2]}, {gt_color[j, 0]}, {gt_color[j, 1]}, {gt_color[j, 2]}\n')

        indices = fps(pos, pos.shape[1] // 4)
        pos = index_points(pos, indices)
        gt = index_gts(gt, indices)


object_to_part = {0: [0, 1, 2, 3], 1: [4, 5], 2: [6, 7], 3: [8, 9, 10, 11], 4: [12, 13, 14, 15], 5: [16, 17, 18], 
                    6: [19, 20, 21], 7: [22, 23], 8: [24, 25, 26, 27], 9: [28, 29], 10: [30, 31, 32, 33, 34, 35],
                    11: [36, 37], 12: [38, 39, 40], 13: [41, 42, 43], 14: [44, 45, 46], 15: [47, 48, 49]}
if __name__ == '__main__':
    points = np.loadtxt('/home/lindi/chenhr/threed/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02958343/1a30678509a1fdfa7fb5267a071ae23a.txt', dtype=np.float32)
    pos = points[:, 0:3]
    gt = points[:, -1]
    gt = gt - object_to_part[3][0]   # 不同物体要修改
    
    device = 'cuda:6'
    pos = torch.as_tensor(pos, device=device).unsqueeze(dim=0)
    gt = torch.as_tensor(gt, device=device).unsqueeze(dim=0)
    
    # indices = fps(pos, pos.shape[1] // 4)
    # pos = index_points(pos, indices)
    # gt = index_gts(gt, indices)
    
    gen_multi_stage(pos, gt, 'car')
        