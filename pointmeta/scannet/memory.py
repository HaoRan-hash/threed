import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import os
import sys
sys.path.append('/mnt/Disk16T/chenhr/threed/utils')
from utils_func import SemanticAwareAttention


class Memory(nn.Module):
    def __init__(self, num_class, length, store_channels, query_channels, base_miu=0.1, end_miu=0.001, use_ddp=False):
        super(Memory, self).__init__()
        self.num_class = num_class
        self.length = length
        self.store_channels = store_channels
        self.query_channles = query_channels
        self.base_miu = base_miu
        self.end_miu = end_miu
        self.poly_power = 0.9
        self.tao = 0.1
        self.use_ddp = use_ddp
        self.register_buffer('memory', torch.zeros((num_class, length, store_channels)))
        self.attention = SemanticAwareAttention(query_channels, store_channels, store_channels)
        self.register_buffer('cur_occupy', torch.zeros((num_class, )))
        
        self.dropout = nn.Dropout(0.1)
        self.attn_mlp = nn.Sequential(nn.Conv1d(query_channels, query_channels*4, 1, bias=False),
                                 nn.BatchNorm1d(query_channels * 4),
                                 nn.ReLU(True),
                                 nn.Dropout(0.1),
                                 nn.Conv1d(query_channels*4, query_channels, 1))
        
        # self.proj_mlp = nn.Sequential(nn.Conv1d(query_channels, store_channels, 1, bias=False),
        #                               nn.BatchNorm1d(store_channels),
        #                               nn.ReLU(True),
        #                               nn.Conv1d(store_channels, store_channels, 1, bias=False),
        #                               nn.BatchNorm1d(store_channels),
        #                               nn.ReLU(True),
        #                               nn.Conv1d(store_channels, store_channels, 1))
    
    def get_not_full_num(self):
        not_full_num = 0
        for i in range(self.num_class):
            if self.cur_occupy[i] != self.length:
                not_full_num += 1
        return not_full_num
    
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
        # 做多卡同步
        if self.use_ddp:
            dist.barrier()
            dist.all_reduce(self.memory, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.cur_occupy, op=dist.ReduceOp.MAX)
            dist.barrier()
            # print(f'device num: {int(os.environ["WORLD_SIZE"])}')   # debug
            self.memory = self.memory / int(os.environ['WORLD_SIZE'])
    
    def forward(self, features, coarse_pred, gts):
        """
        features.shape = (b, n, query_channels)
        coarse_pred.shape = (b, num_class, n)
        return res.shape = (b, n, query_channels)
        """
        b = features.shape[0]
        aug_coef, loss_coef = 1, 1
        not_full_num = self.get_not_full_num()
        if self.training and (not_full_num != 0):
            aug_coef = 0
            loss_coef = 0
        # print(f'aug_coef={aug_coef}, loss_coef={loss_coef}, not full num={not_full_num}')   # debug
        
        memory_features = self.memory.mean(dim=1).unsqueeze(dim=0).expand(b, -1, -1)
        memory_features = F.normalize(memory_features, dim=-1)
        
        # 计算contrast loss
        contrast_loss = 0
        # proj_f = F.normalize(self.proj_mlp(features.transpose(1, 2)), dim=1)   # (b, store_channels, n)
        # contrast_map = torch.matmul(memory_features, proj_f) / self.tao
        # contrast_loss = loss_coef * F.cross_entropy(contrast_map, gts, ignore_index=20)
        
        reve_features = self.attention(F.normalize(features, dim=-1), memory_features, memory_features, coarse_pred)
        res = features + aug_coef * self.dropout(reve_features)
        res = res + aug_coef * self.attn_mlp(res.transpose(1, 2)).transpose(1, 2)
        
        return res, contrast_loss