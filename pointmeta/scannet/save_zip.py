import torch
from torch import nn
import torch.distributed as dist
from argparse import ArgumentParser
import os
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
import sys
sys.path.append('/mnt/Disk16T/chenhr/threed/data_engine')
from dataset import Scannet_Test
from data_aug import *
from data_aug_tensor import *
from model import PointMeta
from model_memory import PointMeta_Memory
import math
from tqdm import tqdm
from pathlib import Path


save_dir = Path('')
label_mapping={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14, 13: 16, 14: 24, 15: 28, 16: 33, 17: 34, 18: 36, 19: 39}
def voting_test(dataloader, test_transform, model, device, model_path, rank):
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    scales = [0.95, 1.0, 1.05]
    rotations = [0, 0.5, 1, 1.5]
    
    if rank == 0:
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader
    with torch.no_grad():
        for pos, color, normal, sort_idx, counts, name in pbar:
            pos = pos.to(device)
            color = color.to(device)
            normal = normal.to(device)
            
            sort_idx = sort_idx.squeeze().numpy()
            counts = counts.squeeze().numpy()
            
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
                
                cum_temp = 0
                for scale in scales:
                    for rot in rotations:
                        angle = math.pi * rot
                        cos, sin = math.cos(angle), math.sin(angle)
                        rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]], device=device)
                        temp_pos, temp_color, temp_normal = test_transform(torch.matmul(cur_pos, rotmat) * scale, 
                                                                           cur_color, torch.matmul(cur_normal, rotmat))
                        temp_pos = temp_pos - temp_pos.min(dim=1)[0]
                        with torch.cuda.amp.autocast():
                            cur_pred = model(temp_pos, temp_color, temp_normal)
                            # memory版
                            # cur_pred, _, _ = model(temp_pos, temp_color, temp_normal)
                        cur_pred = cur_pred.to(dtype=torch.float32)
                        cum_temp += cur_pred
                cum_temp = cum_temp / (len(scales) * len(rotations))
                all_pred[:, :, idx_select] += cum_temp
            
            all_pred = all_pred / all_idx.unsqueeze(dim=1)
            dist.barrier()
            
            save_file_name = save_dir / (str(name.stem) + '.txt')
            all_pred = all_pred.argmax(dim=1).squeeze().cpu().numpy()
            all_pred = np.vectorize(label_mapping.get)(all_pred)
            np.savetxt(save_file_name, all_pred, fmt="%d")
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--use_ddp', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.use_ddp:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
    else:
        rank = 0
    seed = np.random.randint(1, 10000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = f'cuda:{rank}'
    torch.cuda.set_device(device)
    
    test_aug = Compose([ColorNormalizeTensor(mean=[0, 0, 0], std=[1, 1, 1])])
    test_dataset = Scannet_Test('/mnt/Disk16T/chenhr/threed_data/data/scannetv2_test', loop=1)
    if args.use_ddp:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=8, sampler=test_sampler)
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)
    
    pointmeta = PointMeta(20, 4, 32, [4, 8, 4, 4]).to(device)
    # memory版
    # pointmeta = PointMeta_Memory(20, 4, 32, [4, 8, 4, 4], args.use_ddp).to(device)
    if args.use_ddp:
        pointmeta = nn.SyncBatchNorm.convert_sync_batchnorm(pointmeta)
        pointmeta = nn.parallel.DistributedDataParallel(pointmeta, device_ids=[rank], output_device=rank)
    model_path = ''
    
    voting_test(test_dataloader, test_aug, pointmeta, device, model_path, rank)
