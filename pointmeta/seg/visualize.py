import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append('/mnt/Disk16T/chenhr/threed/data_engine')
from dataset import S3dis
from data_aug import *
from data_aug_tensor import *
from memorynet_seg_4_contrast import Memorynet


def gen_color(y):
    """
    y.shape = (n,)
    """
    color_map = [[0, 255, 0], [0, 0, 255], [136, 206, 250],
                 [255, 255, 0], [255, 0, 255], [219, 90, 107],
                 [107, 142, 35], [255, 165, 0], [153, 50, 204],
                 [139, 26, 26], [0, 100, 0], [156, 156, 156], [0, 0, 0]]
    color_map = np.asarray(color_map, dtype=np.float32)
    res = np.zeros((len(y), 3))
    for i in range(13):
        mask = (y == i)
        res[mask] = color_map[i]
    return res


save_dir = Path('/mnt/Disk16T/chenhr/threed/pointmeta/seg/vis_results/pointmeta_mem')
gt_dir = Path('/mnt/Disk16T/chenhr/threed/pointmeta/seg/vis_results/gt')
def test_entire_room(dataloader, test_transform, model, device, model_path, save_gt=False):
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    
    pbar = tqdm(dataloader)
    with torch.no_grad():
        for pos, color, y, sort_idx, counts, name in pbar:
            pos = pos.to(device)
            color = color.to(device)
            y = y.to(device)
            
            sort_idx = sort_idx.squeeze().numpy()
            counts = counts.squeeze().numpy()
            
            all_pred = torch.zeros((1, 13, pos.shape[1]), dtype=torch.float32, device=device)
            all_idx = torch.zeros((1, pos.shape[1]), dtype=torch.float32, device=device)
            for i in range(counts.max()):
                idx_select = np.cumsum(np.insert(counts, 0, 0)[0:-1]) + i % counts
                idx_select = sort_idx[idx_select]
                np.random.shuffle(idx_select)
                all_idx[0, idx_select] += 1
                
                cur_pos = pos[:, idx_select, :]
                cur_color = color[:, idx_select, :]
                cur_pos = cur_pos - cur_pos.min(dim=1, keepdim=True)[0]
                
                # 做变换
                cur_pos, cur_color, _ = test_transform(cur_pos, cur_color, None)
                cur_pred, _, _ = model(cur_pos, cur_color)
                all_pred[:, :, idx_select] += cur_pred
            
            all_pred = all_pred / all_idx.unsqueeze(dim=1)
            
            # visualize
            name = name[0].split('/')[-1][:-4]
            save_file_name = save_dir / (name + '.txt')
            all_pred = all_pred.argmax(dim=1).squeeze(dim=0).cpu().numpy()
            all_pred_color = gen_color(all_pred)
            pos = pos[0].cpu().numpy()
            save_array = np.concatenate((pos, all_pred_color), axis=1)
            np.savetxt(save_file_name, save_array, fmt='%.4f')
            if save_gt:
                gt_file_name = gt_dir / (name + '.txt')
                y_color = gen_color(y[0].cpu().numpy())
                save_array = np.concatenate((pos, y_color), axis=1)
                np.savetxt(gt_file_name, save_array, fmt='%.4f')


def multi_scale_test(dataloader, test_transform, model, device, model_path, save_gt=False):
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    scales = [0.9, 1.0, 1.1]
    
    pbar = tqdm(dataloader)
    with torch.no_grad():
        for pos, color, y, sort_idx, counts, name in pbar:
            pos = pos.to(device)
            color = color.to(device)
            y = y.to(device)
            
            sort_idx = sort_idx.squeeze().numpy()
            counts = counts.squeeze().numpy()
            
            all_pred = torch.zeros((1, 13, pos.shape[1]), dtype=torch.float32, device=device)
            all_idx = torch.zeros((1, pos.shape[1]), dtype=torch.float32, device=device)
            for i in range(counts.max()):
                idx_select = np.cumsum(np.insert(counts, 0, 0)[0:-1]) + i % counts
                idx_select = sort_idx[idx_select]
                np.random.shuffle(idx_select)
                all_idx[0, idx_select] += 1
                
                cur_pos = pos[:, idx_select, :]
                cur_color = color[:, idx_select, :]
                cur_pos = cur_pos - cur_pos.min(dim=1, keepdim=True)[0]
                
                multi_scale_pos = []
                multi_scale_color = []
                for scale in scales:
                    # 做变换
                    temp1, temp2, _ = test_transform(cur_pos * scale, cur_color, None)
                    multi_scale_pos.append(temp1)
                    multi_scale_color.append(temp2)
                multi_scale_pos = torch.cat(multi_scale_pos, dim=0)
                multi_scale_color = torch.cat(multi_scale_color, dim=0)
                
                cur_pred, _, _ = model(multi_scale_pos, multi_scale_color)
                cur_pred = cur_pred.mean(dim=0)
                all_pred[:, :, idx_select] += cur_pred
            
            all_pred = all_pred / all_idx.unsqueeze(dim=1)
            
            # visualize
            name = name[0].split('/')[-1][:-4]
            save_file_name = save_dir / (name + '.txt')
            all_pred = all_pred.argmax(dim=1).squeeze(dim=0).cpu().numpy()
            all_pred_color = gen_color(all_pred)
            pos = pos[0].cpu().numpy()
            save_array = np.concatenate((pos, all_pred_color), axis=1)
            np.savetxt(save_file_name, save_array, fmt='%.4f')
            if save_gt:
                gt_file_name = gt_dir / (name + '.txt')
                y_color = gen_color(y[0].cpu().numpy())
                save_array = np.concatenate((pos, y_color), axis=1)
                np.savetxt(gt_file_name, save_array, fmt='%.4f')


if __name__ == '__main__':
    device = 'cuda:7'

    memorynet = Memorynet(13, 4, 32, [4, 8, 4, 4]).to(device)
    
    model_path = 'pointmeta/seg/checkpoints/memorynet_seg_4_contrast_area5.pth'
    
    # test entire room
    test_aug = Compose([PointCloudFloorCenteringTensor(),
                        ColorNormalizeTensor()])
    test_dataset = S3dis('/mnt/Disk16T/chenhr/threed_data/data/processed_s3dis', split='test', loop=1, test_area=5)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    test_entire_room(test_dataloader, test_aug, memorynet, device, model_path, save_gt=True)

    #multi scale test
    multi_scale_test(test_dataloader, test_aug, memorynet, device, model_path)
