import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import S3dis
from data_aug import Compose
from data_aug_tensor import *
import sys
sys.path.append('/home/lindi/chenhr/threed/pointnet++/seg')
sys.path.append('/home/lindi/chenhr/threed/pointnext/seg')
sys.path.append('/home/lindi/chenhr/threed/pointmeta/seg')
from pointnet2_seg_amp import Pointnet2
from pointnext_seg_amp import PointNeXt
from pointmeta_seg import PointMeta
from memorynet_seg_4_contrast import Memorynet


def gen_color(y):
    """
    y.shape = (n,) cuda
    """
    y = y.to('cpu').squeeze()
    color_map = [[0, 255, 0], [0, 0, 255], [136, 206, 250],
                 [255, 255, 0], [255, 0, 255], [219, 90, 107],
                 [107, 142, 35], [255, 165, 0], [153, 50, 204],
                 [139, 26, 26], [0, 100, 0], [156, 156, 156], [0, 0, 0]]
    color_map = torch.as_tensor(color_map, dtype=torch.float32)
    res = torch.zeros((len(y), 3))
    for i in range(13):
        mask = (y == i)
        res[mask] = color_map[i]
    return res


def predict(dataloader, test_transform, model, device, model_path, model_name, use_multi_scales, save_gt):
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    scales = [0.9, 1.0, 1.1]
    
    # 获取数据
    temp = 47
    dataloader = iter(dataloader)
    for i in range(temp):
        pos, x, y, sort_idx, counts = next(dataloader)
    
    # 数据预处理
    pos = pos.to(device)
    x = x.to(device)
    y = y.to(device)
    
    sort_idx = sort_idx.squeeze().numpy()
    counts = counts.squeeze().numpy()
    
    # 预测
    all_pred = torch.zeros((1, 13, pos.shape[1]), dtype=torch.float32, device=device)
    all_idx = torch.zeros((1, pos.shape[1]), dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(counts.max()):
            idx_select = np.cumsum(np.insert(counts, 0, 0)[0:-1]) + i % counts
            idx_select = sort_idx[idx_select]
            np.random.shuffle(idx_select)
            all_idx[0, idx_select] += 1
            
            cur_pos = pos[:, idx_select, :]
            cur_x = x[:, idx_select, :]
            cur_pos = cur_pos - cur_pos.min(dim=1, keepdim=True)[0]
            
            if use_multi_scales:
                multi_scale_pos = []
                multi_scale_x = []
                for scale in scales:
                    # 做变换
                    temp1, temp2 = test_transform(cur_pos * scale, cur_x)
                    multi_scale_pos.append(temp1)
                    multi_scale_x.append(temp2)
                multi_scale_pos = torch.cat(multi_scale_pos, dim=0)
                multi_scale_x = torch.cat(multi_scale_x, dim=0)
                
                cur_pred, _, _ = model(multi_scale_pos, multi_scale_x)
                cur_pred = cur_pred.mean(dim=0)
            else:
                cur_pos, cur_x = test_transform(cur_pos, cur_x)
                cur_pred, _, _ = model(cur_pos, cur_x)
            all_pred[:, :, idx_select] += cur_pred
        
        all_pred = all_pred / all_idx.unsqueeze(dim=1)
    
    pred_res = all_pred.argmax(dim=1)
    gt_color = gen_color(y)
    pred_color = gen_color(pred_res)
    pos = pos.to('cpu').squeeze()

    with open(f'{temp}/{model_name}.txt', mode='w') as f:
        for i in range(len(pos)):
            f.write(f'{pos[i, 0]}, {pos[i, 1]}, {pos[i, 2]}, {pred_color[i, 0]}, {pred_color[i, 1]}, {pred_color[i, 2]}\n')
    
    if save_gt:
        with open(f'{temp}/gt.txt', mode='w') as f:
            for i in range(len(pos)):
                f.write(f'{pos[i, 0]}, {pos[i, 1]}, {pos[i, 2]}, {gt_color[i, 0]}, {gt_color[i, 1]}, {gt_color[i, 2]}\n')


if __name__ == '__main__':
    device = 'cuda:7'
    test_aug = Compose([PointCloudFloorCenteringTensor(),
                        ColorNormalizeTensor()])
    test_dataset = S3dis('/home/lindi/chenhr/threed/data/processed_s3dis', split='test', loop=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    
    model = Memorynet(13, 4, 32, [4, 8, 4, 4]).to(device)
    model_path = '/home/lindi/chenhr/threed/pointmeta/seg/checkpoints/memorynet_seg_4_contrast_area5_4291.pth'
    model_name = 'pointmeta_ours'
    predict(test_dataloader, test_aug, model, device, model_path, model_name, True, True)
    
    model = Pointnet2(13).to(device)
    model_path = '/home/lindi/chenhr/threed/pointnet++/seg/checkpoints/pointnet2_seg_amp.pth'
    model_name = 'pointnet2'
    predict(test_dataloader, test_aug, model, device, model_path, model_name, False, False)
    
    model = PointNeXt(13, 4, 32, [3, 6, 3, 3]).to(device)
    model_path = '/home/lindi/chenhr/threed/pointnext/seg/pointnext_seg_amp.pth'
    model_name = 'pointnext'
    predict(test_dataloader, test_aug, model, device, model_path, model_name, False, False)
    
    model = PointMeta(13, 4, 32, [4, 8, 4, 4]).to(device)
    model_path = '/home/lindi/chenhr/threed/pointmeta/seg/checkpoints/pointmeta_seg_0001lr.pth'
    model_name = 'pointmeta'
    predict(test_dataloader, test_aug, model, device, model_path, model_name, False, False)
    