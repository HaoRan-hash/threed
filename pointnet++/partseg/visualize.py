import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append('/mnt/Disk16T/chenhr/threed/data_engine')
sys.path.append('/mnt/Disk16T/chenhr/threed/utils')
from utils_func import knn_query_cuda2
from dataset import ShapeNet
from data_aug import *
from data_aug_tensor import *
from memorynet_partseg_contrast import Memorynet, object_to_part
from pointnet2_partseg import Pointnet2


def gen_color(y):
    """
    y.shape = (n,)
    """
    color_map = [[0, 255, 0], [0, 0, 255],
                 [107, 142, 35], [255, 0, 255],
                 [255, 165, 0], [153, 50, 204]]
    color_map = np.asarray(color_map, dtype=np.float32)
    res = np.zeros((len(y), 3))
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
    for part, count in zip(parts, counts):
        if count < 10:
            mask = (y_pred == part)
            _, neigh_pred, _ = knn_query_cuda2(k + 1, pos[mask].unsqueeze(dim=0), pos.unsqueeze(dim=0), y_pred.view(1, len(y_pred), 1).to(dtype=torch.float32))
            neigh_pred = neigh_pred.permute(0, 3, 2, 1).view(count, k + 1).to(dtype=torch.long)
            
            neigh_pred_counts = torch.zeros((count, 50), dtype=torch.long, device=device)
            neigh_pred_counts.scatter_add_(dim=1, index=neigh_pred, src=torch.ones_like(neigh_pred))
            
            y_pred[mask] = neigh_pred_counts.argmax(dim=-1)
    return y_pred


save_dir = Path('/mnt/Disk16T/chenhr/threed/pointnet++/partseg/vis_results/baseline')
gt_dir = Path('/mnt/Disk16T/chenhr/threed/pointnet++/partseg/vis_results/gt')
def voting_test(dataloader, model, device, model_path, voting_num, save_gt=False):
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    idx_to_class = dataloader.dataset.idx_to_class
    voting_transforms = Compose([PointCloudScalingBatch(0.8, 1.2)])

    with torch.no_grad():
        for i, (pos, normal, y, object_labels) in enumerate(tqdm(dataloader)):
            pos = pos.to(device)
            normal = normal.to(device)
            y = y.to(device)
            object_labels = object_labels.to(device)
            
            # voting test
            y_pred = torch.zeros((pos.shape[0], 50, pos.shape[1]), dtype=torch.float32, device=device)
            for _ in range(voting_num):
                temp_pos, _, temp_normal = voting_transforms(pos, None, normal)
                temp_pred = model(temp_pos, temp_normal, object_labels)
                # temp_pred, _, _ = model(temp_pos, temp_normal, object_labels)   # memory版
                y_pred += temp_pred
            y_pred = y_pred / voting_num
            
            y_pred = y_pred.permute(0, 2, 1)
            y_pred = F.softmax(y_pred, dim=-1)

            cur_object_label = object_labels[0].item()
            cur_y_pred = y_pred[0, :, object_to_part[cur_object_label]].argmax(dim=-1)   # 相当于做了mask
            cur_y_pred += object_to_part[cur_object_label][0]
            cur_y = y[0]
            
            # refine seg
            # cur_y_pred = refine_seg(cur_y_pred, pos[0], 10, device)   # memory版使用
            
            # visualize
            shape_name = idx_to_class[cur_object_label]
            cur_dir = save_dir / shape_name
            if not cur_dir.exists():
                cur_dir.mkdir(mode=0o755)
            save_file_name = cur_dir / (shape_name + f'_{i}.txt')
            cur_y_pred = cur_y_pred - object_to_part[cur_object_label][0]
            cur_y_pred = cur_y_pred.cpu().numpy()
            cur_y_pred_color = gen_color(cur_y_pred)
            pos = pos.squeeze(dim=0).cpu().numpy()
            save_array = np.concatenate((pos, cur_y_pred_color), axis=1)
            np.savetxt(save_file_name, save_array, fmt='%.4f')   # 分隔符是空格
            if save_gt:
                cur_dir = gt_dir / shape_name
                if not cur_dir.exists():
                    cur_dir.mkdir(mode=0o755)
                gt_file_name = cur_dir / (shape_name + f'_{i}.txt')
                cur_y = cur_y - object_to_part[cur_object_label][0]
                cur_y = cur_y.cpu().numpy()
                cur_y_color = gen_color(cur_y)
                save_array = np.concatenate((pos, cur_y_color), axis=1)
                np.savetxt(gt_file_name, save_array, fmt='%.4f')   # 分隔符是空格
            

if __name__ == '__main__':
    device = 'cuda:6'

    model = Pointnet2(50).to(device)
    # model = Memorynet(50).to(device)   # memory版

    model_path = 'partseg/checkpoints/pointnet2_partseg.pth'
    # model_path = 'partseg/checkpoints/memorynet_partseg_contrast.pth'   # memory版
    
    # voting test
    test_aug = Compose([PointCloudCenterAndNormalize()])
    test_dataset = ShapeNet('/mnt/Disk16T/chenhr/threed_data/data/shapenetcore_partanno_segmentation_benchmark_v0_normal', 
                            split='test', npoints=2048, transforms=test_aug)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    voting_test(test_dataloader, model, device, model_path, 1, save_gt=False)   # memory时save_gt，voting_num设为10
