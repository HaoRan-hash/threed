import os
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import sys
sys.path.append('/mnt/Disk16T/chenhr/threed/data_engine')
from dataset import Scannet, scan_valtest_collate_fn
from data_aug import *
from data_aug_tensor import *
from model_memory import PointMeta_Memory


def gen_color(y):
    """
    y.shape = (n,)
    """
    color_map = [[174, 199, 232], [152, 223, 137], [31, 119, 180],
                 [255, 188, 120], [188, 189, 35], [140, 86, 74],
                 [255, 152, 151], [213, 39, 40], [196, 176, 213],
                 [149, 103, 188], [197, 156, 148], [23, 190, 208], 
                 [247, 182, 210], [219, 219, 141], [255, 127, 14],
                 [158, 218, 230], [43, 160, 45], [112, 128, 144],
                 [227, 119, 194], [82, 83, 163]]
    color_map = np.asarray(color_map, dtype=np.float32)
    res = np.zeros((len(y), 3))
    for i in range(20):
        mask = (y == i)
        res[mask] = color_map[i]
    return res


save_dir = Path('/mnt/Disk16T/chenhr/threed/pointmeta/scannet/vis_results/memory')
gt_dir = Path('/mnt/Disk16T/chenhr/threed/pointmeta/scannet/vis_results/gt')
def test_entire_room(dataloader, test_transform, model, device, model_path, rank, save_gt=False):
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    
    if rank == 0:
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader
    with torch.no_grad():
        for pos, color, normal, y, sort_idx, counts, name in pbar:
            pos = pos.to(device)
            color = color.to(device)
            normal = normal.to(device)
            y = y.to(device)
            
            # sort_idx = sort_idx.squeeze().numpy()
            # counts = counts.squeeze().numpy()
            
            all_pred = torch.zeros((1, 20, pos.shape[1]), dtype=torch.float32, device=device)
            all_idx = torch.zeros((1, pos.shape[1]), dtype=torch.float32, device=device)
            for i in range(counts.max()):
                idx_select = np.cumsum(np.insert(counts, 0, 0)[0:-1]) + i % counts
                idx_select = sort_idx[idx_select]
                np.random.shuffle(idx_select)
                all_idx[0, idx_select] += 1
                
                cur_pos = pos[:, idx_select, :]
                cur_color = color[:, idx_select, :]
                cur_normal = normal[:, idx_select, :]
                cur_pos = cur_pos - cur_pos.min(dim=1)[0]
                
                # 做变换
                cur_pos, cur_color, cur_normal = test_transform(cur_pos, cur_color, cur_normal)
                with torch.cuda.amp.autocast():
                    cur_pred, _, _ = model(cur_pos, cur_color, cur_normal)
                cur_pred = cur_pred.to(dtype=torch.float32)
                all_pred[:, :, idx_select] += cur_pred
            
            all_pred = all_pred / all_idx.unsqueeze(dim=1)
            dist.barrier()
            
            # visualize
            save_file_name = save_dir / (str(name.stem) + '.txt')
            all_pred = all_pred.argmax(dim=1)
            mask = (y != 20)
            all_pred, y = all_pred[mask], y[mask]
            all_pred = all_pred.squeeze(dim=0).cpu().numpy()
            all_pred_color = gen_color(all_pred)
            pos = pos[mask].squeeze(dim=0).cpu().numpy()
            save_array = np.concatenate((pos, all_pred_color), axis=1)
            np.savetxt(save_file_name, save_array, fmt='%.4f')   # 分隔符是空格
            if save_gt:
                gt_file_name = gt_dir / (str(name.stem) + '.txt')
                y = y.squeeze(dim=0).cpu().numpy()
                y_color = gen_color(y)
                save_array = np.concatenate((pos, y_color), axis=1)
                np.savetxt(gt_file_name, save_array, fmt='%.4f')   # 分隔符是空格


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--use_ddp', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.use_ddp:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
    else:
        rank = 0

    device = f'cuda:{rank}'
    torch.cuda.set_device(device)

    pointmeta = PointMeta_Memory(20, 4, 32, [4, 8, 4, 4], args.use_ddp).to(device)
    if args.use_ddp:
        pointmeta = nn.SyncBatchNorm.convert_sync_batchnorm(pointmeta)
        pointmeta = nn.parallel.DistributedDataParallel(pointmeta, device_ids=[rank], output_device=rank)

    model_path = 'scannet/checkpoints/memorynet_scannet_norm.pth'
    
    # test entire room
    test_aug = Compose([ColorNormalizeTensor(mean=[0, 0, 0], std=[1, 1, 1])])
    test_dataset = Scannet('/mnt/Disk16T/chenhr/threed_data/data/scannetv2_norm', split='val_test', loop=1)
    if args.use_ddp:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=8, sampler=test_sampler, collate_fn=scan_valtest_collate_fn)
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False, collate_fn=scan_valtest_collate_fn)
    test_entire_room(test_dataloader, test_aug, pointmeta, device, model_path, rank, save_gt=True)
