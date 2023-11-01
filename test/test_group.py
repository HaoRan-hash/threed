import torch
import pointnet2_batch_cuda as pointnet2_cuda
from torch.autograd import Function
from utils_func import index_points
import time


class GroupingOperation(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor):
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample, device=features.device)

        pointnet2_cuda.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


if __name__ == '__main__':
    b, n, c = 8, 24000, 32
    nsample, k = 6000, 32
    
    device = 'cuda:5'
    points = torch.randn((b, n, c), device=device, dtype=torch.float32, requires_grad=True)
    idx = torch.randint(low=0, high=24000, size=(b, nsample, k), device=device, dtype=torch.long)
    
    # 我的group
    # res1 = index_points(points, idx)
    # res1.backward(torch.ones_like(res1))
    # print(torch.cuda.max_memory_reserved(device=device))
    
    # 别人的group
    res2 = GroupingOperation.apply(points.transpose(1, 2).contiguous(), idx.to(dtype=torch.int32)).permute(0, 2, 3, 1)
    res2.backward(torch.ones_like(res2))
    print(torch.cuda.max_memory_reserved(device=device))
    
    # # 验证
    # mask = (res1 == res2)
    # print(mask.sum())
