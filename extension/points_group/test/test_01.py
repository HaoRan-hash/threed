import torch
from torch.autograd import Function
import points_group
import logging


class NeighborGroup(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, points, indices):
        """
        points.shape = (b, c, n)
        indices.shape = (b, nsample, k)
        return shape = (b, c, k, nsample)
        """
        assert points.is_contiguous()
        assert indices.is_contiguous()
        
        b, c, n = points.shape
        _, nsample, k = indices.shape        
        group_output = torch.zeros((b, c, k, nsample), device=points.device, dtype=torch.float32)
        
        points_group.neighbor_group(points, indices, group_output)
        ctx.save_for_backward(indices)
        ctx.n = n
        
        return group_output
    
    @staticmethod
    def backward(ctx, output_grad):
        """
        output_grad.shape = (b, c, k, nsample)
        return shape = (b, c, n)
        """
        indices, = ctx.saved_tensors
        n = ctx.n
        
        b, c, _, _ = output_grad.shape
        points_grad = torch.zeros((b, c, n), device=output_grad.device, dtype=torch.float32)
        
        points_group.neighbor_group_grad(output_grad, indices, points_grad)
        return points_grad, None


def index_points(points, indices):
    """
    points.shape = (b, n, c)
    indices.shape = (b, nsamples) or (b, nsamples, k)
    return res.shape = (b, nsamples, c) or (b, nsamples, k, c)
    """
    device = points.device
    b = points.shape[0]

    view_shape = list(indices.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    expand_shape = list(indices.shape)
    expand_shape[0] = -1
    batch_indices = torch.arange(b, device=device).view(view_shape).expand(expand_shape)
    res = points[batch_indices, indices, :]

    return res


if __name__ == '__main__':
    logging.basicConfig(filename='test_01.log', format='%(message)s', level=logging.INFO)
    b, c, n, nsample, k = 8, 64, 24000, 6000, 32
    device = 'cuda:5'
    points = torch.randn((b, c, n), device=device, requires_grad=True)
    indices = torch.randint(0, n, (b, nsample, k), device=device, dtype=torch.long)
    
    res1 = NeighborGroup.apply(points, indices)
    res1.sum().backward()
    logging.info(points.grad)
    logging.info('-------------------------------------------')
    
    points.grad = None
    res2 = index_points(points.transpose(1, 2), indices).permute(0, 3, 2, 1)
    res2.sum().backward()
    logging.info(points.grad)
    
    # mask = (res1 != res2)
    # print(mask.sum())
        