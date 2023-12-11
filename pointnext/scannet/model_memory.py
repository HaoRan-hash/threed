import torch
from torch import nn
import torch.nn.functional as F
import fps_cuda
import sys
sys.path.append('/mnt/Disk16T/chenhr/threed/utils')
from utils_func import ball_query_cuda2, knn_query_cuda2, index_points, index_gts, PEGenerator
from memory import Memory


class PointSetAbstractionLayer(nn.Module):
    def __init__(self, stride, radius, k, in_channels, out_channels):
        super(PointSetAbstractionLayer, self).__init__()
        self.stride = stride
        self.radius = radius
        self.k = k
        
        self.mlp = nn.Sequential(nn.Conv2d(in_channels + 3, out_channels, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True))
    
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


class InvResMLP(nn.Module):
    def __init__(self, radius, k, in_channels):
        super(InvResMLP, self).__init__()
        self.radius = radius
        self.k = k
        
        self.lba1 = nn.Sequential(nn.Conv2d(in_channels + 3, in_channels, 1, bias=False),
                                  nn.BatchNorm2d(in_channels),
                                  nn.ReLU(True))
        self.lba2 = nn.Sequential(nn.Conv1d(in_channels, in_channels * 4, 1, bias=False),
                                  nn.BatchNorm1d(in_channels * 4),
                                  nn.ReLU(True),
                                  nn.Conv1d(in_channels * 4, in_channels, 1, bias=False),
                                  nn.BatchNorm1d(in_channels))
    
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
    
    def forward(self, inputs):
        """
        pos.shape = (b, n, 3)
        x.shape = (b, n, c)
        return shape: (b, n, 3), (b, n, c)
        """
        pos, x = inputs
        group_features = self.group(pos, x, pos)
        group_features, _ = self.lba1(group_features).max(dim=2)
        
        group_features = self.lba2(group_features).transpose(1, 2)
        return pos, F.relu(x + group_features, True)


class StageBlocks(nn.Module):
    def __init__(self, stride, radius, k, in_channels, out_channels, InvRes_num):
        super(StageBlocks, self).__init__()
        self.sa = PointSetAbstractionLayer(stride, radius, k, in_channels, out_channels)
        self.InvRes_num = InvRes_num
        InvResMLPs = []
        for i in range(InvRes_num):
            InvResMLPs.append(InvResMLP(radius*2, k, out_channels))
        self.InvResMLPs = nn.Sequential(*InvResMLPs)
    
    def forward(self, pos, x, y=None):
        pos, x, y = self.sa(pos, x, y)
        pos, x = self.InvResMLPs((pos, x))
        
        return pos, x, y


class PointNeXt_Memory(nn.Module):
    def __init__(self, num_class, stride, k, InvRes_num_list, use_ddp):
        super().__init__()
        self.in_linear = nn.Linear(7, 64)
        self.stage1 = StageBlocks(stride, 0.05, k, 64, 128, InvRes_num_list[0])
        self.stage2 = StageBlocks(stride, 0.1, k, 128, 256, InvRes_num_list[1])
        self.stage3 = StageBlocks(stride, 0.2, k, 256, 512, InvRes_num_list[2])
        self.stage4 = StageBlocks(stride, 0.4, k, 512, 1024, InvRes_num_list[3])
        
        self.fp1 = PointFeaturePropagationLayer(1024+512, [512, 512])
        self.pe_gen1 = PEGenerator(512)
        self.coarse_mlp1 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(512, num_class, kernel_size=1))
        self.memory1 = Memory(num_class, 128, 512, 512, use_ddp=use_ddp)
        
        self.fp2 = PointFeaturePropagationLayer(512+256, [256, 256])
        self.pe_gen2 = PEGenerator(256)
        self.coarse_mlp2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(256, num_class, kernel_size=1))
        self.memory2 = Memory(num_class, 128, 256, 256, use_ddp=use_ddp)

        self.fp3 = PointFeaturePropagationLayer(256+128, [128, 128])
        self.fp4 = PointFeaturePropagationLayer(128+64, [64, 64])
        
        self.out_mlp = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.5),
                                     nn.Conv1d(128, num_class, kernel_size=1))
    
    def forward(self, pos, color, normal, y=None, epoch_ratio=None):
        """
        pos.shape = (b, n, 3)
        color.shape = (b, n, 3)
        normal.shape = (b, n, 3)
        """
        n = pos.shape[1]
        height = pos[:, :, -1:]
        x = torch.cat((color, height, normal), dim=-1)
        x = self.in_linear(x)
        
        pos1, x1, y1 = self.stage1(pos, x, y)
        pos2, x2, y2 = self.stage2(pos1, x1, y1)
        pos3, x3, y3 = self.stage3(pos2, x2, y2)
        pos4, x4, y4 = self.stage4(pos3, x3, y3)
        
        x3 = self.fp1(pos3, x3, pos4, x4)
        # 给x3加上pe
        x3 = self.pe_gen1(pos3, x3, 0.4, 32)
        coarse_pred1 = self.coarse_mlp1(x3.transpose(1, 2))
        coarse_seg_loss1 = 0
        if self.training:
            coarse_seg_loss1 = F.cross_entropy(coarse_pred1, y3, label_smoothing=0.2, ignore_index=20)
            self.memory1.update(x3, y3, coarse_pred1, epoch_ratio)
        x3, _ = self.memory1(x3, coarse_pred1, None)
        
        x2 = self.fp2(pos2, x2, pos3, x3)
        # 给x2加上pe
        x2 = self.pe_gen2(pos2, x2, 0.2, 32)
        coarse_pred2 = self.coarse_mlp2(x2.transpose(1, 2))
        coarse_seg_loss2 = 0
        if self.training:
            coarse_seg_loss2 = F.cross_entropy(coarse_pred2, y2, label_smoothing=0.2, ignore_index=20)
            self.memory2.update(x2, y2, coarse_pred2, epoch_ratio)
        x2, contrast_loss = self.memory2(x2, coarse_pred2, y2)
        
        x1 = self.fp3(pos1, x1, pos2, x2)
        x = self.fp4(pos, x, pos1, x1)
        
        x_max = x.max(dim=1, keepdim=True)[0].expand(-1, n, -1)
        x = torch.cat((x, x_max), dim=-1)
        
        y_pred = self.out_mlp(x.transpose(1, 2))
        
        coarse_seg_loss = coarse_seg_loss1 * 0.1 + coarse_seg_loss2 * 0.1
        return y_pred, coarse_seg_loss, contrast_loss * 0
