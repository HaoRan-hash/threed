import os
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from ignite.metrics import ConfusionMatrix, IoU, mIoU
from tqdm import tqdm
from argparse import ArgumentParser
import json
import sys
sys.path.append('/mnt/Disk16T/chenhr/threed/pointnext')
from dataset import Scannet
from data_aug import *
from data_aug_tensor import *
from model_memory import PointNeXt_Memory
from lovasz_loss import lovasz_softmax
import math


def train_loop(dataloader, model, loss_fn, optimizer, device, 
               cur_epoch, total_epoch, show_gap, interval, rank):
    model.train()
    if rank == 0 and cur_epoch % show_gap == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {cur_epoch}/{total_epoch}', unit='batch')
    else:
        pbar = dataloader
    
    scaler = torch.cuda.amp.GradScaler()
    cm = ConfusionMatrix(20, device=device)
    
    for i, (pos, color, normal, y) in enumerate(pbar):
        pos = pos.to(device)
        color = color.to(device)
        normal = normal.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            y_pred, coarse_seg_loss, contrast_loss = model(pos, color, normal, y, cur_epoch/total_epoch)
            loss = loss_fn(y_pred, y) + coarse_seg_loss + contrast_loss + 3 * lovasz_softmax(y_pred.softmax(dim=1), y, ignore=20)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        # optimizer.step()

        cm.reset()
        cm.update((y_pred, y))
        matrix = cm.compute()
        acc = matrix.diag().sum() / matrix.sum()
        acc = acc.item()

        if rank == 0 and cur_epoch % show_gap == 0 and i % interval == 0:
            pbar.set_postfix_str(f'loss={loss:.4f}, acc={acc:.4f}')


best_miou = 0
best_epoch = 0
def val_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, device, cur_epoch, path, show_gap, log_dir, rank):
    model.eval()
    steps = len(dataloader)
    idx_to_class = dataloader.dataset.idx_to_class
    loss = 0
    
    cm = ConfusionMatrix(20, device=device)
    iou_fn = IoU(cm)
    miou_fn = mIoU(cm)
    with torch.no_grad():
        for pos, color, normal, y in dataloader:
            pos = pos.to(device)
            color = color.to(device)
            normal = normal.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast():
                y_pred, _, _ = model(pos, color, normal)
                loss += loss_fn(y_pred, y)

            cm.update((y_pred, y))
            
    loss = loss / steps
    # 计算oa和macc
    matrix = cm.compute()
    oa = matrix.diag().sum() / matrix.sum()
    oa = oa.item()
    macc = torch.mean(matrix.diag() / matrix.sum(dim=1)).item()
    
    iou = iou_fn.compute()
    miou = miou_fn.compute().item()

    global best_miou, best_epoch
    if miou >= best_miou:
        best_miou = miou
        best_epoch = cur_epoch
        if rank == 0:
            torch.save({'epoch': cur_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': lr_scheduler.state_dict(),
                            'miou': best_miou}, path)

    if rank == 0 and cur_epoch % show_gap == 0:
        with open(log_dir, mode='a') as f:
            f.write(f'Epoch {cur_epoch}\n\n')
            for i in range(20):
                f.write(f'{idx_to_class[i]}:   {iou[i]:.4f}\n')
            f.write(f'val_loss={loss:.4f}, val_miou={miou:.4f}, val_oa={oa:.4f}, val_macc={macc:.4f}\n')
            f.write('-------------------------------------------------------\n')
        print(f'val_loss={loss:.4f}, val_miou={miou:.4f}, val_oa={oa:.4f}, val_macc={macc:.4f}')


def test_entire_room(dataloader, test_transform, model, loss_fn, device, model_path, log_dir, rank):
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    steps = len(dataloader)
    idx_to_class = dataloader.dataset.idx_to_class
    loss = 0
    
    if rank == 0:
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader
    cm = ConfusionMatrix(20, device=device)
    iou_fn = IoU(cm)
    miou_fn = mIoU(cm)
    with torch.no_grad():
        for pos, color, normal, y, sort_idx, counts in pbar:
            pos = pos.to(device)
            color = color.to(device)
            normal = normal.to(device)
            y = y.to(device)
            
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
                
                # 做变换
                cur_pos, cur_color, cur_normal = test_transform(cur_pos, cur_color, cur_normal)
                with torch.cuda.amp.autocast():
                    cur_pred, _, _ = model(cur_pos, cur_color, cur_normal)
                cur_pred = cur_pred.to(dtype=torch.float32)
                all_pred[:, :, idx_select] += cur_pred
            
            all_pred = all_pred / all_idx.unsqueeze(dim=1)
            
            loss += loss_fn(all_pred, y)
            cm.update((all_pred, y))
        
    loss = loss / steps
    # 计算oa和macc
    matrix = cm.compute()
    oa = matrix.diag().sum() / matrix.sum()
    oa = oa.item()
    macc = torch.mean(matrix.diag() / matrix.sum(dim=1)).item()
    
    iou = iou_fn.compute()
    miou = miou_fn.compute().item()
    
    if rank == 0:
        with open(log_dir, mode='a') as f:
            for i in range(20):
                f.write(f'{idx_to_class[i]}:   {iou[i]:.4f}\n')
            f.write(f'test_loss={loss:.4f}, test_miou={miou:.4f}, test_oa={oa:.4f}, test_macc={macc:.4f}\n')
            f.write('-------------------------------------------------------\n')
        print(f'test_loss={loss:.4f}, test_miou={miou:.4f}, test_oa={oa:.4f}, test_macc={macc:.4f}')


def voting_test(dataloader, test_transform, model, loss_fn, device, model_path, log_dir, rank):
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    steps = len(dataloader)
    idx_to_class = dataloader.dataset.idx_to_class
    loss = 0
    scales = [0.95, 1.0, 1.05]
    rotations = [0, 0.5, 1, 1.5]
    
    if rank == 0:
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader
    cm = ConfusionMatrix(20, device=device)
    iou_fn = IoU(cm)
    miou_fn = mIoU(cm)
    with torch.no_grad():
        for pos, color, normal, y, sort_idx, counts in pbar:
            pos = pos.to(device)
            color = color.to(device)
            normal = normal.to(device)
            y = y.to(device)
            
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
                            cur_pred, _, _ = model(temp_pos, temp_color, temp_normal)
                        cur_pred = cur_pred.to(dtype=torch.float32)
                        cum_temp += cur_pred
                cum_temp = cum_temp / len(scales)
                all_pred[:, :, idx_select] += cum_temp
            
            all_pred = all_pred / all_idx.unsqueeze(dim=1)
            
            loss += loss_fn(all_pred, y)
            cm.update((all_pred, y))
            dist.barrier()
        
    loss = loss / steps
    # 计算oa和macc
    matrix = cm.compute()
    oa = matrix.diag().sum() / matrix.sum()
    oa = oa.item()
    macc = torch.mean(matrix.diag() / matrix.sum(dim=1)).item()
    
    iou = iou_fn.compute()
    miou = miou_fn.compute().item()
    
    if rank == 0:
        with open(log_dir, mode='a') as f:
            for i in range(20):
                f.write(f'{idx_to_class[i]}:   {iou[i]:.4f}\n')
            f.write(f'test_loss={loss:.4f}, test_miou={miou:.4f}, test_oa={oa:.4f}, test_macc={macc:.4f}\n')
            f.write('-------------------------------------------------------\n')
        print(f'test_loss={loss:.4f}, test_miou={miou:.4f}, test_oa={oa:.4f}, test_macc={macc:.4f}')


def get_parameter_groups(model, weight_decay, log_dir, rank):
    parameter_group_names = {'decay': {'weight_decay': weight_decay,
                                       'params': []},
                             'no_decay': {'weight_decay': 0,
                                          'params': []}}
    parameter_group_vars = {'decay': {'weight_decay': weight_decay,
                                       'params': []},
                             'no_decay': {'weight_decay': 0,
                                          'params': []}}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  
        
        if len(param.shape) == 1 or name.endswith('.bias'):
            group_name = 'no_decay'
        else:
            group_name = 'decay'
        
        parameter_group_names[group_name]['params'].append(name)
        parameter_group_vars[group_name]['params'].append(param)
    
    if rank == 0:
        with open(log_dir, mode='a') as f:
            f.write(f'Param groups = {json.dumps(parameter_group_names, indent=2)}\n')
    return list(parameter_group_vars.values())


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
    
    log_dir = 'scannet/logs/memorynet_scannet_norm.log'
    # logging.basicConfig(filename=log_dir, format='%(message)s', level=logging.INFO)
    if rank == 0:
        with open(log_dir, mode='a') as f:
            f.write(f'random seed {seed}\n')

    train_aug = Compose([PointCloudRotation_Z(1.0, True),
                         PointCloudScaling(0.8, 1.2),
                         ElasticDistortion(),
                         NormalDrop(0.2),
                         ColorContrast(p=0.2),
                         ColorDrop(p=0.2),
                         ColorNormalize(mean=[0, 0, 0], std=[1, 1, 1])])
    train_dataset = Scannet('/mnt/Disk16T/chenhr/threed_data/data/scannetv2_norm', split='train', loop=6, npoints=64000, transforms=train_aug)
    val_aug = Compose([ColorNormalize(mean=[0, 0, 0], std=[1, 1, 1])])
    val_dataset = Scannet('/mnt/Disk16T/chenhr/threed_data/data/scannetv2_norm', split='val', loop=1, npoints=None, transforms=val_aug)
    
    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=8, sampler=train_sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=8, sampler=val_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=8, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=False)

    device = f'cuda:{rank}'
    torch.cuda.set_device(device)

    pointnext = PointNeXt_Memory(20, 4, 32, [3, 6, 3, 3], args.use_ddp).to(device)
    if args.use_ddp:
        pointnext = nn.SyncBatchNorm.convert_sync_batchnorm(pointnext)
        pointnext = nn.parallel.DistributedDataParallel(pointnext, device_ids=[rank], output_device=rank)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, ignore_index=20)
    
    # 配置不同的weight decay
    parameter_group = get_parameter_groups(pointnext, weight_decay=1e-4, log_dir=log_dir, rank=rank)
    optimizer = torch.optim.AdamW(parameter_group, lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70, 90], 0.1)

    epochs = 100
    show_gap = 1
    save_path = 'scannet/checkpoints/memorynet_scannet_norm.pth'
    for i in range(epochs):
        if args.use_ddp:
            train_sampler.set_epoch(i)
        train_loop(train_dataloader, pointnext, loss_fn, optimizer, device, i, epochs, show_gap, 1, rank)
        val_loop(val_dataloader, pointnext, loss_fn, optimizer, lr_scheduler, device, i, save_path, show_gap, log_dir, rank)
        lr_scheduler.step()
    
    # test entire room
    test_aug = Compose([ColorNormalizeTensor(mean=[0, 0, 0], std=[1, 1, 1])])
    test_dataset = Scannet('/mnt/Disk16T/chenhr/threed_data/data/scannetv2_norm', split='val_test', loop=1)
    if args.use_ddp:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=8, sampler=test_sampler)
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)
    test_entire_room(test_dataloader, test_aug, pointnext, loss_fn, device, save_path, log_dir, rank)
    voting_test(test_dataloader, test_aug, pointnext, loss_fn, device, save_path, log_dir, rank)
    