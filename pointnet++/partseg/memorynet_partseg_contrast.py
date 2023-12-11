import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from ignite.metrics import Accuracy
from tqdm import tqdm
import sys
sys.path.append('/mnt/Disk16T/chenhr/threed/data_engine')
sys.path.append('/mnt/Disk16T/chenhr/threed/utils')
from utils_func import ball_query_cuda2, knn_query_cuda2, index_points, index_gts, PolyFocalLoss, SemanticAwareAttention_Mask, PEGenerator
from dataset import ShapeNet
from data_aug import *
from data_aug_tensor import *
import fps_cuda
import json
import math


class PointSetAbstractionLayer(nn.Module):
    def __init__(self, stride, radius, k, in_channels, mlp_units, is_group_all=False):
        super(PointSetAbstractionLayer, self).__init__()
        self.stride = stride
        self.radius = radius
        self.k = k
        self.is_group_all = is_group_all
        
        mlp = [nn.Conv2d(in_channels, mlp_units[0], kernel_size=1),
                    nn.BatchNorm2d(mlp_units[0]),
                    nn.ReLU(inplace=True)]
        for i in range(len(mlp_units) - 1):
            mlp += [nn.Conv2d(mlp_units[i], mlp_units[i + 1], kernel_size=1),
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
        return res.shape = (b, 3+c, k, self.nsamples)
        """
        group_points, group_features, _ = ball_query_cuda2(self.radius, self.k, centroids, points, features)
        
        group_points = group_points - centroids.transpose(1, 2).unsqueeze(dim=2)   # 要的是相对坐标

        res = torch.cat((group_points / self.radius, group_features), dim=1)
        return res

    def group_all(self, points, features):
        """
        points.shape = (b, n, 3)
        features.shape = (b, n, c)
        return res.shape = (b, 3+c, n, 1)
        """
        b, n, _ = points.shape
        device = points.device
        choice = torch.randint(0, n, (b, 1), device=device)
        
        centroids = index_points(points, choice)   # centroids.shape = (b, 1, 3)
        group_points = points.unsqueeze(dim=1)
        group_features = features.unsqueeze(dim=1)
        
        group_points = group_points - centroids.unsqueeze(dim=2)
        res = torch.cat((group_points, group_features), dim=-1).permute(0, 3, 2, 1)
        return centroids, res
        
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
        
        if self.is_group_all:
            centroids, group_features = self.group_all(points, features)
        else:
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
        mlp = [nn.Conv1d(in_channels, mlp_units[0], kernel_size=1),
                    nn.BatchNorm1d(mlp_units[0]),
                    nn.ReLU(inplace=True)]
        for i in range(len(mlp_units) - 1):
            mlp += [nn.Conv1d(mlp_units[i], mlp_units[i + 1], kernel_size=1),
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
            receive_features = features_2.expand(-1, n1, -1).transpose(1, 2)
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
        self.attention = SemanticAwareAttention_Mask(query_channels, store_channels, store_channels)
        self.cur_occupy = [0] * num_class
        
        self.dropout = nn.Dropout(0.1)
        self.attn_mlp = nn.Sequential(nn.Conv1d(query_channels, query_channels*4, 1, bias=False),
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
        self.contrast_loss_fn = PolyFocalLoss(2.0, 0.25, 1.0)
    
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
    
    def forward(self, features, coarse_pred, mask, gts=None):
        """
        features.shape = (b, n, query_channels)
        coarse_pred.shape = (b, num_class, n)
        mask.shape = (b, 1, num_class)
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
            # contrast_loss = F.cross_entropy(contrast_map, gts)
            contrast_loss = self.contrast_loss_fn(contrast_map, gts)
        
        reve_features = self.attention(F.normalize(features, dim=-1), memory_features, memory_features, mask, coarse_pred)

        res = features + self.dropout(reve_features)
        res = res + self.attn_mlp(res.transpose(1, 2)).transpose(1, 2)
        
        return res, contrast_loss


class Memorynet(nn.Module):
    def __init__(self, num_class):
        super(Memorynet, self).__init__()
        self.in_linear = nn.Linear(7, 32)
        self.sa1 = PointSetAbstractionLayer(4, 0.2, 32, 32+3, [64, 64, 128])
        self.sa2 = PointSetAbstractionLayer(4, 0.4, 64, 128+3, [128, 128, 256])
        self.sa3 = PointSetAbstractionLayer(1, None, None, 256+3, [256, 512, 1024], True)

        self.fp1 = PointFeaturePropagationLayer(1024+256, [256, 256])
        self.pe_gen1 = PEGenerator(256)
        self.coarse_mlp1 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(256, num_class, kernel_size=1))
        self.coarse_loss_fn1 = PolyFocalLoss(2.0, 0.25, 1.0)
        self.memory1 = Memory(num_class, 32, 256, 256)
        
        self.fp2 = PointFeaturePropagationLayer(256+128, [256, 128])
        self.pe_gen2 = PEGenerator(128)
        self.coarse_mlp2 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(128, num_class, kernel_size=1))
        self.coarse_loss_fn2 = PolyFocalLoss(2.0, 0.25, 1.0)
        self.memory2 = Memory(num_class, 32, 128, 128)
        
        self.fp3 = PointFeaturePropagationLayer(128+32+16, [128, 128, 128])

        self.mlp = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.5),
                                    nn.Conv1d(128, num_class, kernel_size=1))

    def forward(self, pos, normal, object_labels, y=None, epoch_ratio=None):
        """
        pos.shape = (b, n, 3)
        normal.shape = (b, n, 4)
        object_labels.shape = (b,)
        y.shape = (b, n)
        """
        mask = object_to_part_onehot[object_labels]
        mask = (1 - mask).to(dtype=torch.bool)
        mask_float = torch.zeros_like(mask, dtype=torch.float32)
        mask_float.masked_fill_(mask, float('-inf'))
        mask_float = mask_float.unsqueeze(dim=1)
        
        n = pos.shape[1]
        x = torch.cat((pos, normal), dim=-1)
        x = self.in_linear(x)

        pos1, x1, y1 = self.sa1(pos, x, y)
        pos2, x2, y2 = self.sa2(pos1, x1, y1)
        pos3, x3, y3 = self.sa3(pos2, x2, y2)

        x2 = self.fp1(pos2, x2, pos3, x3)
        # 给x2加上pe
        x2 = self.pe_gen1(pos2, x2, 0.8, 32)
        coarse_pred1 = self.coarse_mlp1(x2.transpose(1, 2))
        coarse_seg_loss1 = 0
        if self.training:
            coarse_seg_loss1 = self.coarse_loss_fn1(coarse_pred1, y2)
            self.memory1.update(x2, y2, coarse_pred1, epoch_ratio)
        x2, _ = self.memory1(x2, coarse_pred1, mask_float)
        
        x1 = self.fp2(pos1, x1, pos2, x2)
        # 给x1加上pe
        x1 = self.pe_gen2(pos1, x1, 0.4, 32)
        coarse_pred2 = self.coarse_mlp2(x1.transpose(1, 2))
        coarse_seg_loss2 = 0
        if self.training:
            coarse_seg_loss2 = self.coarse_loss_fn2(coarse_pred2, y1)
            self.memory2.update(x1, y1, coarse_pred2, epoch_ratio)
        x1, contrast_loss = self.memory2(x1, coarse_pred2, mask_float, y1)
        
        object_labels = F.one_hot(object_labels, 16).to(dtype=torch.float32)
        object_labels = object_labels.unsqueeze(dim=1).expand(-1, n, -1)
        x = self.fp3(pos, torch.cat((x, object_labels), dim=-1), pos1, x1)

        y_pred = self.mlp(x.transpose(1, 2))

        coarse_seg_loss = coarse_seg_loss1 * 0.2 + coarse_seg_loss2 * 0.2
        return y_pred, coarse_seg_loss, contrast_loss * 0.01


def train_loop(dataloader, model, loss_fn, metric_fn, optimizer, device, 
               cur_epoch, total_epoch, show_gap, interval):
    model.train()
    if cur_epoch % show_gap == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {cur_epoch}/{total_epoch}', unit='batch')
    else:
        pbar = dataloader
    
    for i, (pos, x, y, object_labels) in enumerate(pbar):
        pos = pos.to(device)
        x = x.to(device)
        y = y.to(device)
        object_labels = object_labels.to(device)
        optimizer.zero_grad()
        
        y_pred, coarse_seg_loss, contrast_loss = model(pos, x, object_labels, y, cur_epoch/total_epoch)
        loss = loss_fn(y_pred, y) + coarse_seg_loss + contrast_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        metric_fn.reset()
        metric_fn.update((y_pred, y))
        acc = metric_fn.compute()

        if cur_epoch % show_gap == 0 and i % interval == 0:
            pbar.set_postfix_str(f'loss={loss:.4f}, acc={acc:.4f}')


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


best_miou = 0
best_epoch = 0
object_to_part = {0: [0, 1, 2, 3], 1: [4, 5], 2: [6, 7], 3: [8, 9, 10, 11], 4: [12, 13, 14, 15], 5: [16, 17, 18], 
                    6: [19, 20, 21], 7: [22, 23], 8: [24, 25, 26, 27], 9: [28, 29], 10: [30, 31, 32, 33, 34, 35],
                    11: [36, 37], 12: [38, 39, 40], 13: [41, 42, 43], 14: [44, 45, 46], 15: [47, 48, 49]}
def val_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, device, cur_epoch, path, show_gap, log_dir):
    model.eval()
    steps = len(dataloader)
    idx_to_class = dataloader.dataset.idx_to_class
    loss = 0
    object_mious = [[] for _ in range(16)]
    
    with torch.no_grad():
        for pos, x, y, object_labels in dataloader:
            pos = pos.to(device)
            x = x.to(device)
            y = y.to(device)
            object_labels = object_labels.to(device)
            
            y_pred, _, _ = model(pos, x, object_labels)
            loss += loss_fn(y_pred, y)
            
            y_pred = y_pred.permute(0, 2, 1)
            y_pred = F.softmax(y_pred, dim=-1)
            for i in range(len(y_pred)):
                cur_object_label = object_labels[i].item()
                cur_y_pred = y_pred[i, :, object_to_part[cur_object_label]].argmax(dim=-1)   # 相当于做了mask
                cur_y_pred += object_to_part[cur_object_label][0]
                cur_y = y[i]
                
                temp = []
                for part_class in object_to_part[cur_object_label]:
                    if (torch.sum(cur_y == part_class) == 0 and torch.sum(cur_y_pred == part_class) == 0):
                        temp.append(1)
                    else:
                        intersection = torch.sum((cur_y == part_class) & (cur_y_pred == part_class)).item()
                        union = torch.sum((cur_y == part_class) | (cur_y_pred == part_class)).item()
                        temp.append(intersection / union)
                object_mious[cur_object_label].append(np.mean(temp))
        
    loss = loss / steps
    cat_mious = [np.mean(object_mious[i]) for i in range(16)]
    all_mious = [y for x in object_mious for y in x]
    ins_miou = np.mean(all_mious)
    cat_miou = np.mean(cat_mious)

    global best_miou, best_epoch
    if ins_miou >= best_miou:
        best_miou = ins_miou
        best_epoch = cur_epoch
        torch.save({'epoch': cur_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                        'miou': best_miou}, path)

    if cur_epoch % show_gap == 0:
        with open(log_dir, mode='a') as f:
            f.write(f'Epoch {cur_epoch}\n\n')
            for i in range(16):
                f.write(f'{idx_to_class[i]}:   {cat_mious[i]:.4f}\n')
            f.write(f'val_loss={loss:.4f}, ins_miou={ins_miou:.4f}, cat_miou={cat_miou:.4f}\n')
            f.write('-------------------------------------------------------\n')
        print(f'val_loss={loss:.4f}, ins_miou={ins_miou:.4f}, cat_miou={cat_miou:.4f}')


def voting_test(dataloader, model, loss_fn, device, model_path, log_dir, voting_num):
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    steps = len(dataloader)
    idx_to_class = dataloader.dataset.idx_to_class
    loss = 0
    object_mious = [[] for _ in range(16)]
    voting_transforms = Compose([PointCloudScalingBatch(0.8, 1.2)])

    with torch.no_grad():
        for pos, x, y, object_labels in tqdm(dataloader):
            pos = pos.to(device)
            x = x.to(device)
            y = y.to(device)
            object_labels = object_labels.to(device)
            
            # voting test
            y_pred = torch.zeros((pos.shape[0], 50, pos.shape[1]), dtype=torch.float32, device=device)
            for i in range(voting_num):
                temp_pos, temp_x = voting_transforms(pos, x)
                temp_pred, _, _ = model(temp_pos, temp_x, object_labels)
                y_pred += temp_pred
            y_pred = y_pred / voting_num
            
            loss += loss_fn(y_pred, y)
            
            y_pred = y_pred.permute(0, 2, 1)
            y_pred = F.softmax(y_pred, dim=-1)
            for i in range(len(y_pred)):
                cur_object_label = object_labels[i].item()
                cur_y_pred = y_pred[i, :, object_to_part[cur_object_label]].argmax(dim=-1)   # 相当于做了mask
                cur_y_pred += object_to_part[cur_object_label][0]
                cur_y = y[i]
                
                # refine seg
                cur_y_pred = refine_seg(cur_y_pred, pos[i], 10, device)
                
                temp = []
                for part_class in object_to_part[cur_object_label]:
                    if (torch.sum(cur_y == part_class) == 0 and torch.sum(cur_y_pred == part_class) == 0):
                        temp.append(1)
                    else:
                        intersection = torch.sum((cur_y == part_class) & (cur_y_pred == part_class)).item()
                        union = torch.sum((cur_y == part_class) | (cur_y_pred == part_class)).item()
                        temp.append(intersection / union)
                object_mious[cur_object_label].append(np.mean(temp))
                
    loss = loss / steps
    cat_mious = [np.mean(object_mious[i]) for i in range(16)]
    all_mious = [y for x in object_mious for y in x]
    ins_miou = np.mean(all_mious)
    cat_miou = np.mean(cat_mious)
    
    with open(log_dir, mode='a') as f:
        for i in range(16):
            f.write(f'{idx_to_class[i]}:   {cat_mious[i]:.4f}\n')
        f.write(f'test_loss={loss:.4f}, ins_miou={ins_miou:.4f}, cat_miou={cat_miou:.4f}\n')
        f.write('-------------------------------------------------------\n')
    print(f'test_loss={loss:.4f}, ins_miou={ins_miou:.4f}, cat_miou={cat_miou:.4f}')


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


object_to_part_onehot = torch.zeros((16, 50), dtype=torch.uint8, device='cuda:6')
def transform_object_label():
    for i in range(16):
        for part in object_to_part[i]:
            object_to_part_onehot[i, part] = 1
transform_object_label()


if __name__ == '__main__':
    seed = 6692
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    log_dir = 'partseg/memorynet_partseg_contrast.log'
    # logging.basicConfig(filename=log_dir, format='%(message)s', level=logging.INFO)
    with open(log_dir, mode='a') as f:
        f.write(f'random seed {seed}\n')

    train_aug = Compose([PointCloudScaling(0.8, 1.2),
                         PointCloudCenterAndNormalize(),
                         PointCloudJitter(0.001, 0.005),
                         NormalDrop(0.2)])
    train_dataset = ShapeNet('/home/lindi/chenhr/threed/data/shapenetcore_partanno_segmentation_benchmark_v0_normal', 
                             split='train', npoints=2048, transforms=train_aug)
    val_aug = Compose([PointCloudCenterAndNormalize()])
    val_dataset = ShapeNet('/home/lindi/chenhr/threed/data/shapenetcore_partanno_segmentation_benchmark_v0_normal', 
                            split='test', npoints=2048, transforms=val_aug)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    device = 'cuda:4'

    memorynet = Memorynet(50).to(device)
    loss_fn = PolyFocalLoss(2.0, 0.25, 1.0)
    metric_acc = Accuracy(device=device)
    
    # 配置不同的weight decay
    parameter_group = get_parameter_groups(memorynet, weight_decay=1e-4, log_dir=log_dir)
    optimizer = torch.optim.AdamW(parameter_group, lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [140, 180], 0.1)

    epochs = 200
    show_gap = 1
    save_path = 'partseg/memorynet_partseg_contrast.pth'
    for i in range(epochs):
        train_loop(train_dataloader, memorynet, loss_fn, metric_acc, optimizer, device, i, epochs, show_gap, 1)
        val_loop(val_dataloader, memorynet, loss_fn, optimizer, lr_scheduler, device, i, save_path, show_gap, log_dir)
        lr_scheduler.step()
    
    # voting test
    test_dataset = ShapeNet('/home/lindi/chenhr/threed/data/shapenetcore_partanno_segmentation_benchmark_v0_normal', 
                            split='test', npoints=2048, transforms=val_aug)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)
    voting_test(test_dataloader, memorynet, loss_fn, device, save_path, log_dir, 10)
