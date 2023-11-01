import torch
import numpy as np
from data_aug import PointCloudCenterAndNormalize
from data_aug_tensor import PointCloudScalingBatch
from utils_func import knn_query_cuda2
import sys
sys.path.append('/home/lindi/chenhr/threed/pointnext/partseg')
sys.path.append('/home/lindi/chenhr/threed/pointnet++/partseg')
sys.path.append('/home/lindi/chenhr/threed/pointmeta/partseg')
from memorynet_partseg_contrast import Memorynet
from pointnet2_partseg import Pointnet2
from pointnext_partseg import Pointnext
from pointmeta_partseg import PointMeta


def gen_color(y):
    """
    y.shape = (n,) cuda
    """
    y = y.to('cpu').squeeze()
    color_map = [[0, 255, 0], [0, 0, 255],
                 [107, 142, 35], [255, 0, 255],
                 [255, 165, 0], [153, 50, 204]]
    color_map = torch.as_tensor(color_map, dtype=torch.float32)
    res = torch.zeros((len(y), 3))
    for i in range(6):
        mask = (y == i)
        res[mask] = color_map[i]
    return res


def refine_seg(y_pred, pos, k, device):
    """
    y_pred.shape = (n, )
    pos.shape = (n, 3)
    """
    parts, counts = y_pred.unique(return_counts=True)
    y_pred_clone = y_pred.clone()
    for part, count in zip(parts, counts):
        if count < 10:
            mask = (y_pred_clone == part)
            _, neigh_pred, _ = knn_query_cuda2(k + 1, pos[mask].unsqueeze(dim=0), pos.unsqueeze(dim=0), y_pred.view(1, len(y_pred), 1).to(dtype=torch.float32))
            neigh_pred = neigh_pred.permute(0, 3, 2, 1).view(count, k + 1).to(dtype=torch.long)
            
            neigh_pred_counts = torch.zeros((count, 50), dtype=torch.long, device=device)
            neigh_pred_counts.scatter_add_(dim=1, index=neigh_pred, src=torch.ones_like(neigh_pred))
            
            y_pred[mask] = neigh_pred_counts.argmax(dim=-1)
    return y_pred


object_to_part = {0: [0, 1, 2, 3], 1: [4, 5], 2: [6, 7], 3: [8, 9, 10, 11], 4: [12, 13, 14, 15], 5: [16, 17, 18], 
                    6: [19, 20, 21], 7: [22, 23], 8: [24, 25, 26, 27], 9: [28, 29], 10: [30, 31, 32, 33, 34, 35],
                    11: [36, 37], 12: [38, 39, 40], 13: [41, 42, 43], 14: [44, 45, 46], 15: [47, 48, 49]}
def predict(pos, x, y, object_label, model, device, model_path, model_name, use_voting, save_gt):
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    
    # 预测
    with torch.no_grad():
        if use_voting:
            voting_transforms = PointCloudScalingBatch(0.8, 1.2)
            
            y_pred = torch.zeros((pos.shape[0], 50, pos.shape[1]), dtype=torch.float32, device=device)
            for i in range(10):
                temp_pos, temp_x = voting_transforms(pos, x)
                temp_pred, _, _ = model(temp_pos, temp_x, object_label)
                y_pred += temp_pred
            y_pred = y_pred / 10
        else:
            y_pred, _, _ = model(pos, x, object_label)
    y_pred = y_pred.transpose(1, 2)
    
    object_label = object_label[0].item()
    y_pred = y_pred[0, :, object_to_part[object_label]].argmax(dim=-1)
    # refine seg
    y_pred = refine_seg(y_pred, pos[0], 10, device)
    
    gt_color = gen_color(y - object_to_part[object_label][0])
    pred_color = gen_color(y_pred)
    pos = pos.to('cpu').squeeze()

    with open(f'table/{model_name}.txt', mode='w') as f:   # 改一下名字
        for i in range(len(pos)):
            f.write(f'{pos[i, 0]}, {pos[i, 1]}, {pos[i, 2]}, {pred_color[i, 0]}, {pred_color[i, 1]}, {pred_color[i, 2]}\n')
    
    if save_gt:
        with open(f'table/gt.txt', mode='w') as f:
            for i in range(len(pos)):
                f.write(f'{pos[i, 0]}, {pos[i, 1]}, {pos[i, 2]}, {gt_color[i, 0]}, {gt_color[i, 1]}, {gt_color[i, 2]}\n')


if __name__ == '__main__':
    points = np.genfromtxt('/home/lindi/chenhr/threed/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/04379243/c24b7a315dbf2f3178ab7c8b395efbfe.txt', dtype=np.float32)

    # choice = np.random.choice(len(points), 2048)
    # points = points[choice]
    pos, x, y = points[:, 0:3], points[:, 3:6], points[:, -1]
    object_label = np.array([15], dtype=np.int64)   # 根据物体来填
    
    test_aug = PointCloudCenterAndNormalize()
    pos, x = test_aug(pos, x)
    
    device = 'cuda:7'
    pos = torch.as_tensor(pos, dtype=torch.float32).to(device).unsqueeze(dim=0)
    x = torch.as_tensor(x, dtype=torch.float32).to(device).unsqueeze(dim=0)
    y = torch.as_tensor(y, dtype=torch.int64).to(device).unsqueeze(dim=0)
    object_label = torch.as_tensor(object_label, dtype=torch.int64).to(device)
    
    model = Memorynet(50).to(device)
    model_path = '/home/lindi/chenhr/threed/pointnext/partseg/memorynet_partseg_contrast.pth'
    model_name = 'pointnext_ours'
    predict(pos, x, y, object_label, model, device, model_path, model_name, True, True)
    
    model = Pointnet2(50).to(device)
    model_path = '/home/lindi/chenhr/threed/pointnet++/partseg/pointnet2_partseg.pth'
    model_name = 'pointnet2'
    predict(pos, x, y, object_label, model, device, model_path, model_name, False, False)
    
    model = PointMeta(50).to(device)
    model_path = '/home/lindi/chenhr/threed/pointmeta/partseg/pointmeta_partseg.pth'
    model_name = 'pointmeta'
    predict(pos, x, y, object_label, model, device, model_path, model_name, False, False)
    
    model = Pointnext(50).to(device)
    model_path = '/home/lindi/chenhr/threed/pointnext/partseg/pointnext_partseg.pth'
    model_name = 'pointnext'
    predict(pos, x, y, object_label, model, device, model_path, model_name, False, False)
