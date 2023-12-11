import torch
from torch import nn
import torch.nn.functional as F
import fps_cuda
import sys
sys.path.append('/mnt/Disk16T/chenhr/threed/utils')
from utils_func import ball_query_cuda2, knn_query_cuda2, index_points, index_gts


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


class Pointnet2(nn.Module):
    def __init__(self, num_class):
        super(Pointnet2, self).__init__()
        self.sa1 = PointSetAbstractionLayer(4, 0.05, 32, 7+3, [32, 32, 64])
        self.sa2 = PointSetAbstractionLayer(4, 0.1, 32, 64+3, [64, 64, 128])
        self.sa3 = PointSetAbstractionLayer(4, 0.2, 32, 128+3, [128, 128, 256])
        self.sa4 = PointSetAbstractionLayer(4, 0.4, 32, 256+3, [256, 256, 512])

        self.fp1 = PointFeaturePropagationLayer(512+256, [256, 256])
        self.fp2 = PointFeaturePropagationLayer(256+128, [256, 256])
        self.fp3 = PointFeaturePropagationLayer(256+64, [256, 128])
        self.fp4 = PointFeaturePropagationLayer(128+7, [128, 128])

        self.out_mlp = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.5),
                                     nn.Conv1d(128, num_class, kernel_size=1))

    def forward(self, pos, color):
        """
        pos.shape = (b, n, 3)
        color.shape = (b, n, 3)
        """
        height = pos[:, :, -1:] - pos[:, :, -1].min()
        x = torch.cat((pos, color, height), dim=-1)

        pos1, x1, _ = self.sa1(pos, x)
        pos2, x2, _ = self.sa2(pos1, x1)
        pos3, x3, _ = self.sa3(pos2, x2)
        pos4, x4, _ = self.sa4(pos3, x3)

        x3 = self.fp1(pos3, x3, pos4, x4)
        x2 = self.fp2(pos2, x2, pos3, x3)
        x1 = self.fp3(pos1, x1, pos2, x2)
        x = self.fp4(pos, x, pos1, x1)

        y_pred = self.out_mlp(x.transpose(1, 2))

        return y_pred
