import torch
from torch import nn
import torch.nn.functional as F
import fps_cuda
import sys
sys.path.append('/home/lindi/chenhr/threed/pointnet++')
from utils_func import ball_query_cuda2, knn_query_cuda2, index_points, index_gts, PEGenerator
from memory import Memory


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


class Pointnet2_Memory(nn.Module):
    def __init__(self, num_class, use_ddp):
        super().__init__()
        self.sa1 = PointSetAbstractionLayer(4, 0.05, 32, 7+3, [32, 32, 64])
        self.sa2 = PointSetAbstractionLayer(4, 0.1, 32, 64+3, [64, 64, 128])
        self.sa3 = PointSetAbstractionLayer(4, 0.2, 32, 128+3, [128, 128, 256])
        self.sa4 = PointSetAbstractionLayer(4, 0.4, 32, 256+3, [256, 256, 512])

        self.fp1 = PointFeaturePropagationLayer(512+256, [256, 256])
        self.pe_gen1 = PEGenerator(256)
        self.coarse_mlp1 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(256, num_class, kernel_size=1))
        self.memory1 = Memory(num_class, 128, 256, 256, use_ddp=use_ddp)
        
        self.fp2 = PointFeaturePropagationLayer(256+128, [256, 256])
        self.pe_gen2 = PEGenerator(256)
        self.coarse_mlp2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(256, num_class, kernel_size=1))
        self.memory2 = Memory(num_class, 128, 256, 256, use_ddp=use_ddp)
        
        self.fp3 = PointFeaturePropagationLayer(256+64, [256, 128])
        self.fp4 = PointFeaturePropagationLayer(128+7, [128, 128])

        self.out_mlp = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.5),
                                     nn.Conv1d(256, num_class, kernel_size=1))

    def forward(self, pos, color, normal, y=None, epoch_ratio=None):
        """
        pos.shape = (b, n, 3)
        color.shape = (b, n, 3)
        normal.shape = (b, n, 3)
        """
        n = pos.shape[1]
        height = pos[:, :, -1:]
        x = torch.cat((color, height, normal), dim=-1)

        pos1, x1, y1 = self.sa1(pos, x, y)
        pos2, x2, y2 = self.sa2(pos1, x1, y1)
        pos3, x3, y3 = self.sa3(pos2, x2, y2)
        pos4, x4, y4 = self.sa4(pos3, x3, y3)

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
