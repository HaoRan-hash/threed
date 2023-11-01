import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from ignite.metrics import Accuracy, ConfusionMatrix, IoU, mIoU
from tqdm import tqdm
import json
import sys
sys.path.append('/home/lindi/chenhr/threed/pointnet++')
from utils_func import ball_query_cuda2, knn_query_cuda2, index_points, index_gts, SemanticAwareAttention, PEGenerator
from dataset import S3dis
from data_aug import *
from data_aug_tensor import *
import fps_cuda
import math


class PointSetAbstractionLayer(nn.Module):
    def __init__(self, stride, radius, k, in_channels, mlp_units):
        super(PointSetAbstractionLayer, self).__init__()
        self.stride = stride
        self.radius = radius
        self.k = k
        
        mlp = [nn.Conv2d(in_channels, mlp_units[0], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_units[0]),
                    nn.ReLU(inplace=True)]
        for i in range(len(mlp_units) - 1):
            mlp += [nn.Conv2d(mlp_units[i], mlp_units[i + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_units[i + 1]),
                    nn.ReLU(inplace=True)]

        self.mlp = nn.Sequential(*mlp)
    
    @torch.no_grad()
    def fps(self, points):
        """
        points.shape = (b, n, 3)
        return indices.shape = (b, self.nsamples)
        """
        b, n, _ = points.shape
        device = points.device
        dis = torch.ones((b, n), device=device) * 1e10
        indices = torch.zeros((b, self.nsamples), device=device, dtype=torch.long)

        fps_cuda.fps(points, dis, indices)
        return indices
    
    def group(self, points, features, centroids):
        """
        points.shape = (b, n, 3)
        features.shape = (b, n, c)
        centroids.shape = (b, self.nsamples, 3)
        return res.shape = (b, self.nsamples, k, 3+c)
        """
        group_points, group_point_features, _ = ball_query_cuda2(self.radius, self.k, centroids, points, features)
        
        group_points = group_points - centroids.transpose(1, 2).unsqueeze(dim=2)   # 要的是相对坐标

        res = torch.cat((group_points / self.radius, group_point_features), dim=1)
        return res

    def forward(self, points, features, gts=None):
        """
        points.shape = (b, n, 3)   坐标信息
        features.shape = (b, n, c)   特征信息
        gts.shape = (b, n)
        return centroids.shape = (b, self.nsamples, 3)
        return group_features.shape = (b, self.nsamples, c')
        return gts.shape = (b, self.nsamples)
        """
        self.nsamples = points.shape[1] // self.stride

        fps_indices = self.fps(points)

        centroids = index_points(points, fps_indices)
        if gts is not None:
            gts = index_gts(gts, fps_indices)

        group_features = self.group(points, features, centroids)

        group_features, _ = self.mlp(group_features).max(dim=2)
        group_features = group_features.transpose(1, 2)

        return centroids, group_features, gts


class PointFeaturePropagationLayer(nn.Module):
    def __init__(self, in_channels, mlp_units):
        super(PointFeaturePropagationLayer, self).__init__()
        mlp = [nn.Conv1d(in_channels, mlp_units[0], kernel_size=1, bias=False),
                    nn.BatchNorm1d(mlp_units[0]),
                    nn.ReLU(inplace=True)]
        for i in range(len(mlp_units) - 1):
            mlp += [nn.Conv1d(mlp_units[i], mlp_units[i + 1], kernel_size=1, bias=False),
                    nn.BatchNorm1d(mlp_units[i + 1]),
                    nn.ReLU(inplace=True)]

        self.mlp = nn.Sequential(*mlp)
    
    def forward(self, points_1, features_1, points_2, features_2):
        """
        points_1来自于更前面层的输出, 点更多
        points_1.shape = (b, n1, 3)
        features_1.shape = (b, n1, c1)
        points_2.shape = (b, n2, 3)
        features_2.shape = (b, n2, c2)
        return final_features.shape = (b, n1, c')
        """
        _, n1, _ = points_1.shape
        _, n2, _ = points_2.shape

        if n2 == 1:
            receive_features = features_2.expand(-1, n1, -1)
        else:
            _, correspond_features, distance = knn_query_cuda2(3, points_1, points_2, features_2)

            dist_recip = 1/ (distance + 1e-8)
            norm_term = dist_recip.sum(dim=-1, keepdim=True)
            weights = dist_recip / norm_term

            receive_features = torch.sum(weights.transpose(1, 2).unsqueeze(dim=1) * correspond_features, dim=2)
        
        final_features = torch.cat((receive_features, features_1.transpose(1, 2)), dim=1)

        final_features = self.mlp(final_features).transpose(1, 2)

        return final_features


class Memory(nn.Module):
    def __init__(self, num_class, length, store_channels, query_channels, base_miu=0.1, end_miu=0.001):
        super(Memory, self).__init__()
        self.num_class = num_class
        self.length = length
        self.store_channels = store_channels
        self.query_channles = query_channels
        self.base_miu = base_miu
        self.end_miu = end_miu
        self.poly_power = 0.9
        self.tao = 1.0
        self.register_buffer('memory', torch.zeros((num_class, length, store_channels)))
        self.attention = SemanticAwareAttention(query_channels, store_channels, store_channels)
        self.cur_occupy = [0] * num_class
        
        self.dropout = nn.Dropout(0.1)
        self.attn_mlp = nn.Sequential(nn.Conv1d(query_channels, query_channels*4, 1),
                                 nn.BatchNorm1d(query_channels * 4),
                                 nn.ReLU(True),
                                 nn.Dropout(0.1),
                                 nn.Conv1d(query_channels*4, query_channels, 1))

        self.proj_mlp = nn.Sequential(nn.Conv1d(query_channels, store_channels, 1, bias=False),
                                      nn.BatchNorm1d(store_channels),
                                      nn.ReLU(True),
                                      nn.Conv1d(store_channels, store_channels, 1, bias=False),
                                      nn.BatchNorm1d(store_channels),
                                      nn.ReLU(True),
                                      nn.Conv1d(store_channels, store_channels, 1))
    
    def is_full(self):
        res = True
        for i in range(self.num_class):
            res = (res and (self.cur_occupy[i] == self.length))
        return res
    
    @torch.no_grad()
    def update(self, features, gts, coarse_pred, epoch_ratio):
        """
        features.shape = (b, n, store_channels)
        gts.shape = (b, n)
        coarse_pred.shape = (b, num_class, n)
        """
        coarse_pred = coarse_pred.detach().transpose(1, 2).softmax(dim=-1)
        _, pred_labels = coarse_pred.max(dim=-1)
        
        mask1 = (pred_labels == gts)
        cur_miu = math.pow(1 - epoch_ratio, self.poly_power) * (self.base_miu - self.end_miu) + self.end_miu
        
        for i in range(self.num_class):
            mask2 = (gts == i)
            mask = (mask1 & mask2)
            cur_features = features[mask]
            n = len(cur_features)
            
            # debug
            # with open('seg/pointnext_contrast.log', mode='a') as f:
            #     f.write(f'class {i}, {n} samples\n')
            
            if n != 0 :   # 如果存在该类的feature
                # 模仿dataset的选取策略
                if n >= self.length:
                    choice = torch.arange(0, self.length, 1, dtype=torch.long)
                else:
                    temp = torch.arange(0, n, 1, dtype=torch.long)
                    pad_choice = torch.randint(0, n, (self.length-n, ), dtype=torch.long)
                    choice = torch.cat((temp, pad_choice))
                
                if self.cur_occupy[i] != self.length:   # 该类的memory未满
                    self.memory[i] = cur_features[choice]
                    self.cur_occupy[i] += self.length
                else:
                    self.memory[i] = cur_features[choice] * cur_miu + self.memory[i] * (1 - cur_miu)
    
    def forward(self, features, coarse_pred, gts=None):
        """
        features.shape = (b, n, query_channels)
        coarse_pred.shape = (b, num_class, n)
        return res.shape = (b, n, query_channels)
        """
        b = features.shape[0]
        contrast_loss = 0
        if self.training and (not self.is_full()):
            print('not full')
            return features, contrast_loss
        
        memory_features = self.memory.mean(dim=1).unsqueeze(dim=0).expand(b, -1, -1)
        memory_features = F.normalize(memory_features, dim=-1)
        if gts is not None:
            proj_f = F.normalize(self.proj_mlp(features.transpose(1, 2)), dim=1)   # (b, store_channels, n)
            contrast_map = torch.matmul(memory_features, proj_f) / self.tao
            contrast_loss = F.cross_entropy(contrast_map, gts)
        
        reve_features = self.attention(F.normalize(features, dim=-1), memory_features, memory_features)

        res = features + self.dropout(reve_features)
        res = res + self.attn_mlp(res.transpose(1, 2)).transpose(1, 2)
        
        return res, contrast_loss


class Memorynet(nn.Module):
    def __init__(self, num_class):
        super(Memorynet, self).__init__()
        self.sa1 = PointSetAbstractionLayer(4, 0.1, 32, 4+3, [32, 32, 64])
        self.sa2 = PointSetAbstractionLayer(4, 0.2, 32, 64+3, [64, 64, 128])
        self.sa3 = PointSetAbstractionLayer(4, 0.4, 32, 128+3, [128, 128, 256])
        self.sa4 = PointSetAbstractionLayer(4, 0.8, 32, 256+3, [256, 256, 512])

        self.fp1 = PointFeaturePropagationLayer(512+256, [256, 256])
        self.pe_gen1 = PEGenerator(256)
        self.coarse_mlp1 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(256, num_class, kernel_size=1))
        self.memory1 = Memory(num_class, 128, 256, 256)
        
        self.fp2 = PointFeaturePropagationLayer(256+128, [256, 256])
        self.pe_gen2 = PEGenerator(256)
        self.coarse_mlp2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(256, num_class, kernel_size=1))
        self.memory2 = Memory(num_class, 128, 256, 256)
        
        self.fp3 = PointFeaturePropagationLayer(256+64, [256, 128])
        self.fp4 = PointFeaturePropagationLayer(128+4, [128, 128])

        self.out_mlp = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.5),
                                     nn.Conv1d(128, num_class, kernel_size=1))

    def forward(self, pos, x, y=None, epoch_ratio=None):
        """
        pos.shape = (b, n, 3)
        x.shape = (b, n, 3)
        """
        x = torch.cat((pos[:, :, -1:], x), dim=-1)

        pos1, x1, y1 = self.sa1(pos, x, y)
        pos2, x2, y2 = self.sa2(pos1, x1, y1)
        pos3, x3, y3 = self.sa3(pos2, x2, y2)
        pos4, x4, y4 = self.sa4(pos3, x3, y3)

        x3 = self.fp1(pos3, x3, pos4, x4)
        # 给x3加上pe
        x3 = self.pe_gen1(pos3, x3, 0.8, 32)
        coarse_pred1 = self.coarse_mlp1(x3.transpose(1, 2))
        coarse_seg_loss1 = 0
        if self.training:
            coarse_seg_loss1 = F.cross_entropy(coarse_pred1, y3, label_smoothing=0.2)
            self.memory1.update(x3, y3, coarse_pred1, epoch_ratio)
        x3, _ = self.memory1(x3, coarse_pred1)
        
        x2 = self.fp2(pos2, x2, pos3, x3)
        # 给x2加上pe
        x2 = self.pe_gen2(pos2, x2, 0.4, 32)
        coarse_pred2 = self.coarse_mlp2(x2.transpose(1, 2))
        coarse_seg_loss2 = 0
        if self.training:
            coarse_seg_loss2 = F.cross_entropy(coarse_pred2, y2, label_smoothing=0.2)
            self.memory2.update(x2, y2, coarse_pred2, epoch_ratio)
        x2, contrast_loss = self.memory2(x2, coarse_pred2, y2)
        
        x1 = self.fp3(pos1, x1, pos2, x2)
        x = self.fp4(pos, x, pos1, x1)

        y_pred = self.out_mlp(x.transpose(1, 2))
        coarse_seg_loss = coarse_seg_loss1 * 0.2 + coarse_seg_loss2 * 0.2

        return y_pred, coarse_seg_loss, contrast_loss * 0.01


def train_loop(dataloader, model, loss_fn, metric_fn, optimizer, device, 
               cur_epoch, total_epoch, show_gap, interval):
    model.train()
    if cur_epoch % show_gap == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {cur_epoch}/{total_epoch}', unit='batch')
    else:
        pbar = dataloader
    
    scaler = torch.cuda.amp.GradScaler()
    
    for i, (pos, x, y) in enumerate(pbar):
        pos = pos.to(device)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            y_pred, coarse_seg_loss, contrast_loss = model(pos, x, y, cur_epoch/total_epoch)
            loss = loss_fn(y_pred, y) + coarse_seg_loss + contrast_loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()

        metric_fn.reset()
        metric_fn.update((y_pred, y))
        acc = metric_fn.compute()

        if cur_epoch % show_gap == 0 and i % interval == 0:
            pbar.set_postfix_str(f'loss={loss:.4f}, acc={acc:.4f}')


best_miou = 0
best_epoch = 0
def val_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, device, cur_epoch, path, show_gap, log_dir):
    model.eval()
    steps = len(dataloader)
    idx_to_class = dataloader.dataset.idx_to_class
    loss = 0
    
    oa = Accuracy(device=device)
    cm = ConfusionMatrix(13, device=device)
    iou_fn = IoU(cm)
    miou_fn = mIoU(cm)
    with torch.no_grad():
        for pos, x, y in dataloader:
            pos = pos.to(device)
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast():
                y_pred, _, _ = model(pos, x)
                loss += loss_fn(y_pred, y)

            oa.update((y_pred, y))
            cm.update((y_pred, y))
            
    loss = loss / steps
    oa = oa.compute()
    # 计算macc
    matrix = cm.compute()
    macc = torch.mean(matrix.diag() / matrix.sum(dim=1)).item()
    
    iou = iou_fn.compute()
    miou = miou_fn.compute().item()

    global best_miou, best_epoch
    if miou >= best_miou:
        best_miou = miou
        best_epoch = cur_epoch
        torch.save({'epoch': cur_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                        'miou': best_miou}, path)

    if cur_epoch % show_gap == 0:
        with open(log_dir, mode='a') as f:
            f.write(f'Epoch {cur_epoch}\n\n')
            for i in range(13):
                f.write(f'{idx_to_class[i]}:   {iou[i]:.4f}\n')
            f.write(f'val_loss={loss:.4f}, val_miou={miou:.4f}, val_oa={oa:.4f}, val_macc={macc:.4f}\n')
            f.write('-------------------------------------------------------\n')
        print(f'val_loss={loss:.4f}, val_miou={miou:.4f}, val_oa={oa:.4f}, val_macc={macc:.4f}')


def test_entire_room(dataloader, test_transform, model, loss_fn, device, model_path, log_dir):
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    steps = len(dataloader)
    idx_to_class = dataloader.dataset.idx_to_class
    loss = 0
    
    pbar = tqdm(dataloader)
    oa = Accuracy(device=device)
    cm = ConfusionMatrix(13, device=device)
    iou_fn = IoU(cm)
    miou_fn = mIoU(cm)
    with torch.no_grad():
        for pos, x, y, sort_idx, counts in pbar:
            pos = pos.to(device)
            x = x.to(device)
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
                cur_x = x[:, idx_select, :]
                cur_pos = cur_pos - cur_pos.min(dim=1, keepdim=True)[0]
                
                # 做变换
                cur_pos, cur_x = test_transform(cur_pos, cur_x)
                cur_pred, _, _ = model(cur_pos, cur_x)
                all_pred[:, :, idx_select] += cur_pred
            
            all_pred = all_pred / all_idx.unsqueeze(dim=1)
            
            loss += loss_fn(all_pred, y)
            oa.update((all_pred, y))
            cm.update((all_pred, y))
        
    loss = loss / steps
    oa = oa.compute()
    # 计算macc
    matrix = cm.compute()
    macc = torch.mean(matrix.diag() / matrix.sum(dim=1)).item()
    
    iou = iou_fn.compute()
    miou = miou_fn.compute().item()
    
    with open(log_dir, mode='a') as f:
        for i in range(13):
            f.write(f'{idx_to_class[i]}:   {iou[i]:.4f}\n')
        f.write(f'test_loss={loss:.4f}, test_miou={miou:.4f}, test_oa={oa:.4f}, test_macc={macc:.4f}\n')
        f.write('-------------------------------------------------------\n')
    print(f'test_loss={loss:.4f}, test_miou={miou:.4f}, test_oa={oa:.4f}, test_macc={macc:.4f}')


def multi_scale_test(dataloader, test_transform, model, loss_fn, device, model_path, log_dir):
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    steps = len(dataloader)
    idx_to_class = dataloader.dataset.idx_to_class
    loss = 0
    scales = [0.9, 1.0, 1.1]
    
    pbar = tqdm(dataloader)
    oa = Accuracy(device=device)
    cm = ConfusionMatrix(13, device=device)
    iou_fn = IoU(cm)
    miou_fn = mIoU(cm)
    with torch.no_grad():
        for pos, x, y, sort_idx, counts in pbar:
            pos = pos.to(device)
            x = x.to(device)
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
                cur_x = x[:, idx_select, :]
                cur_pos = cur_pos - cur_pos.min(dim=1, keepdim=True)[0]
                
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
                all_pred[:, :, idx_select] += cur_pred
            
            all_pred = all_pred / all_idx.unsqueeze(dim=1)
            
            loss += loss_fn(all_pred, y)
            oa.update((all_pred, y))
            cm.update((all_pred, y))
        
    loss = loss / steps
    oa = oa.compute()
    # 计算macc
    matrix = cm.compute()
    macc = torch.mean(matrix.diag() / matrix.sum(dim=1)).item()
    
    iou = iou_fn.compute()
    miou = miou_fn.compute().item()
    
    with open(log_dir, mode='a') as f:
        for i in range(13):
            f.write(f'{idx_to_class[i]}:   {iou[i]:.4f}\n')
        f.write(f'test_loss={loss:.4f}, test_miou={miou:.4f}, test_oa={oa:.4f}, test_macc={macc:.4f}\n')
        f.write('-------------------------------------------------------\n')
    print(f'test_loss={loss:.4f}, test_miou={miou:.4f}, test_oa={oa:.4f}, test_macc={macc:.4f}')
    return matrix


def get_parameter_groups(model, weight_decay, log_dir):
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
    
    with open(log_dir, mode='a') as f:
        f.write(f'Param groups = {json.dumps(parameter_group_names, indent=2)}\n')
    return list(parameter_group_vars.values())


if __name__ == '__main__':
    seed = 7208
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    log_dir = 'seg/logs/memorynet_seg_x3x2_contrast_vanilla_attention.log'
    # logging.basicConfig(filename=log_dir, format='%(message)s', level=logging.INFO)
    with open(log_dir, mode='a') as f:
        f.write(f'random seed {seed}\n')

    train_aug = Compose([ColorContrast(p=0.2),
                         PointCloudScaling(0.9, 1.1),
                         PointCloudFloorCentering(),
                         PointCloudJitter(0.005, 0.02),
                         ColorDrop(p=0.2),
                         ColorNormalize()])
    train_dataset = S3dis('/home/lindi/chenhr/threed/data/processed_s3dis', split='train', loop=30, npoints=24000, transforms=train_aug, test_area=5)
    val_aug = Compose([PointCloudFloorCentering(),
                       ColorNormalize()])
    val_dataset = S3dis('/home/lindi/chenhr/threed/data/processed_s3dis', split='val', loop=1, npoints=None, transforms=val_aug, test_area=5)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    device = 'cuda:5'

    pointnet2 = Memorynet(13).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
    metric_acc = Accuracy(device=device)
    
    # 配置不同的weight decay
    parameter_group = get_parameter_groups(pointnet2, weight_decay=1e-4, log_dir=log_dir)
    optimizer = torch.optim.AdamW(parameter_group, lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-5)

    epochs = 100
    show_gap = 1
    save_path = 'seg/checkpoints/memorynet_seg_x3x2_contrast_vanilla_attention.pth'
    for i in range(epochs):
        train_loop(train_dataloader, pointnet2, loss_fn, metric_acc, optimizer, device, i, epochs, show_gap, 1)
        val_loop(val_dataloader, pointnet2, loss_fn, optimizer, lr_scheduler, device, i, save_path, show_gap, log_dir)
        lr_scheduler.step()
    
    # test entire room
    test_aug = Compose([PointCloudFloorCenteringTensor(),
                        ColorNormalizeTensor()])
    test_dataset = S3dis('/home/lindi/chenhr/threed/data/processed_s3dis', split='test', loop=1, test_area=5)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    # test_entire_room(test_dataloader, test_aug, pointnet2, loss_fn, device, save_path, log_dir)

    #multi scale test
    multi_scale_test(test_dataloader, test_aug, pointnet2, loss_fn, device, save_path, log_dir)
    