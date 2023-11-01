import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
import points_query
from torch.autograd import Function
import points_group


@torch.no_grad()
def get_square_distance(query_pos, all_pos):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    return shape = (b, sample, n)
    """
    res = ((query_pos.unsqueeze(dim=2) - all_pos.unsqueeze(dim=1)) ** 2).sum(dim=-1)
    return res


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


@torch.no_grad()
def index_gts(gts, indices):
    """
    gts.shape = (b, n)
    indices.shape = (b, self.nsamples)
    return res.shape = (b, self.nsamples)
    """
    device = gts.device
    b = gts.shape[0]

    batch_indices = torch.arange(b, device=device).view(b, 1).expand(-1, indices.shape[1])
    res = gts[batch_indices, indices]

    return res


def knn_query(k, query_pos, all_pos, all_x):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    all_x.shape = (b, n, c)
    return shape = (b, sample, k, 3), (b, sample, k, c), (b, sample, k)
    """
    square_dis = get_square_distance(query_pos, all_pos)
    k_dis, k_indices = square_dis.topk(k, largest=False)   # k_indices.shape = (b, sample, k)
    
    return index_points(all_pos, k_indices), index_points(all_x, k_indices), torch.sqrt(k_dis)


def ball_query(radius, k, query_pos, all_pos, all_x):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    all_x.shape = (b, n, c)
    return shape = (b, sample, k, 3), (b, sample, k, c), (b, sample, k)
    """
    square_dis = get_square_distance(query_pos, all_pos)
    k_dis, k_indices = square_dis.topk(k, largest=False)   # k_indices.shape = (b, sample, k)
    
    # ball query比knn麻烦一点
    mask = (k_dis > radius**2)
    temp = k_indices[:, :, 0:1].expand(-1, -1, k)
    temp2 = k_dis[:, :, 0:1].expand(-1, -1, k)
    
    k_indices[mask] = temp[mask]
    k_dis[mask] = temp2[mask]
    
    return index_points(all_pos, k_indices), index_points(all_x, k_indices), torch.sqrt(k_dis)


def knn_query_cuda(k, query_pos, all_pos, all_x):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    all_x.shape = (b, n, c)
    return shape = (b, sample, k, 3), (b, sample, k, c), (b, sample, k)
    """
    with torch.no_grad():
        b, m, _ = query_pos.shape
        device = query_pos.device
        k_indices = torch.zeros((b, m, k), dtype=torch.long, device=device)
        k_dis = torch.zeros((b, m, k), dtype=torch.float32, device=device)
        
        points_query.knn_query(k, all_pos, query_pos, k_indices, k_dis)
    return index_points(all_pos, k_indices), index_points(all_x, k_indices), torch.sqrt(k_dis)


def ball_query_cuda(radius, k, query_pos, all_pos, all_x):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    all_x.shape = (b, n, c)
    return shape = (b, sample, k, 3), (b, sample, k, c), (b, sample, k)
    """
    with torch.no_grad():
        b, m, _ = query_pos.shape
        device = query_pos.device
        k_indices = torch.zeros((b, m, k), dtype=torch.long, device=device)
        k_dis = torch.zeros((b, m, k), dtype=torch.float32, device=device)
        
        points_query.ball_query(k, radius, all_pos, query_pos, k_indices, k_dis)
    return index_points(all_pos, k_indices), index_points(all_x, k_indices), torch.sqrt(k_dis)


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
    @torch.cuda.amp.custom_bwd
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


def knn_query_cuda2(k, query_pos, all_pos, all_x):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    all_x.shape = (b, n, c)
    return shape = (b, 3, k, sample), (b, c, k, sample), (b, sample, k)
    """
    with torch.no_grad():
        b, m, _ = query_pos.shape
        device = query_pos.device
        k_indices = torch.zeros((b, m, k), dtype=torch.long, device=device)
        k_dis = torch.zeros((b, m, k), dtype=torch.float32, device=device)
        
        points_query.knn_query(k, all_pos, query_pos, k_indices, k_dis)
    all_pos = all_pos.transpose(1, 2).contiguous()
    all_x = all_x.transpose(1, 2).contiguous()
    
    group_points = NeighborGroup.apply(all_pos, k_indices)
    group_features = NeighborGroup.apply(all_x, k_indices)
    return group_points, group_features, torch.sqrt(k_dis)


def ball_query_cuda2(radius, k, query_pos, all_pos, all_x):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    all_x.shape = (b, n, c)
    return shape = (b, 3, k, sample), (b, c, k, sample), (b, sample, k)
    """
    with torch.no_grad():
        b, m, _ = query_pos.shape
        device = query_pos.device
        k_indices = torch.zeros((b, m, k), dtype=torch.long, device=device)
        k_dis = torch.zeros((b, m, k), dtype=torch.float32, device=device)
        
        points_query.ball_query(k, radius, all_pos, query_pos, k_indices, k_dis)
    all_pos = all_pos.transpose(1, 2).contiguous()
    all_x = all_x.transpose(1, 2).contiguous()
    
    group_points = NeighborGroup.apply(all_pos, k_indices)
    group_features = NeighborGroup.apply(all_x, k_indices)
    return group_points, group_features, torch.sqrt(k_dis)


class MultiheadAttentionRelPE(nn.Module):
    def __init__(self, embed_dim, kdim, vdim, num_heads):
        super(MultiheadAttentionRelPE, self).__init__()
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(kdim, embed_dim)
        self.v_proj = nn.Linear(vdim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, q, k, v, pe1, pe2):
        """
        q.shape = (n, b, embed_dim)
        k.shape = (l, b, kdim)
        v.shape = (l, b, vdim)
        pe1.shape = (b, n, 3)
        pe2.shape = (l, 3)
        """
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        q = rearrange(q, 'n b (num_head head_dim) -> (b num_head) n head_dim', num_head = self.num_heads)
        k = rearrange(k, 'l b (num_head head_dim) -> (b num_head) l head_dim', num_head = self.num_heads)
        v = rearrange(v, 'l b (num_head head_dim) -> (b num_head) l head_dim', num_head = self.num_heads)
        
        pe_scores = torch.matmul(pe1, pe2.T)
        pe_scores = pe_scores.unsqueeze(dim=1).expand(-1, self.num_heads, -1, -1).flatten(0, 1)   # pe_scores.shape = (b*num_head, n, l)
        attn_map = (torch.matmul(q, k.transpose(1, 2)) + pe_scores) / math.sqrt(self.head_dim)
        attn_map = torch.softmax(attn_map, dim=-1)
        output = torch.matmul(attn_map, v)
        
        output = rearrange(output, '(b num_head) n head_dim -> n b (num_head head_dim)', num_head = self.num_heads)
        output = self.out_proj(output)
        
        return output


class SemanticAwareAttention(nn.Module):
    def __init__(self, embed_dim, kdim, vdim):
        super(SemanticAwareAttention, self).__init__()
        self.embed_dim = embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(kdim, embed_dim)
        self.v_proj = nn.Linear(vdim, embed_dim)
    
    def forward(self, q, k, v, coarse_pred=None, need_weights=False):
        """
        q.shape = (b, n, embed_dim)
        k.shape = (b, num_class, kdim)
        v.shape = (b, num_class, vdim)
        coarse_pred.shape = (b, num_class, n)
        """
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        attn_map = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.embed_dim)
        attn_map = attn_map.softmax(dim=-1)
        
        if coarse_pred is not None:
            coarse_pred = coarse_pred.detach()
            coarse_pred = coarse_pred.transpose(1, 2).softmax(dim=-1)
            attn_map = torch.softmax(attn_map * coarse_pred, dim=-1)
            # attn_map = attn_map * coarse_pred
            # attn_map = attn_map / attn_map.sum(dim=-1, keepdim=True)
        
        output = torch.bmm(attn_map, v)
        
        if need_weights:
            return output, attn_map
        else:
            return output


class PEGenerator(nn.Module):
    def __init__(self, out_channel):
        super(PEGenerator, self).__init__()
        self.pe_mlp = nn.Sequential(nn.Conv2d(3, 3, 1, bias=False),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU(True),
                                    nn.Conv2d(3, out_channel, 1))
    
    def forward(self, pos, x, radius, k):
        """
        pos.shape = (b, n, 3)
        x.shape = (b, n, c)
        return shape = (b, n, c)
        """
        pos = pos.detach()
        group_pos, _, _ = ball_query_cuda2(radius, k, pos, pos, pos)   # (b, 3, k, n)
        group_pos = group_pos - pos.transpose(1, 2).unsqueeze(dim=2)
        pe, _ = self.pe_mlp(group_pos).max(dim=2)
        x = x + pe.transpose(1, 2)
        
        return x


class PolyFocalLoss(nn.Module):
    def __init__(self, gamma, alpha, epsilon):
        super(PolyFocalLoss, self). __init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(self, y_pred, y):
        """
        y_pred.shape = (b, c, n)
        y.shape = (b, n)
        """
        y_pred = y_pred.transpose(1, 2) 
        p = torch.sigmoid(y_pred)
        
        y = F.one_hot(y, y_pred.shape[2])   # y.shape = (b, n, c)
        y = y.to(dtype=y_pred.dtype)
        
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction='none')
        
        pt = y * p + (1 - y) * (1 - p)
        focal_loss = bce_loss * ((1 - pt) ** self.gamma)
        
        alpha_t = y * self.alpha + (1 - y) * (1 - self.alpha)
        focal_loss = alpha_t * focal_loss
        
        poly_focal_loss = focal_loss + self.epsilon * torch.pow(1 - pt, self.gamma + 1)
        
        return poly_focal_loss.mean()


class SemanticAwareAttention_Mask(nn.Module):
    def __init__(self, embed_dim, kdim, vdim):
        super(SemanticAwareAttention_Mask, self).__init__()
        self.embed_dim = embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(kdim, embed_dim)
        self.v_proj = nn.Linear(vdim, embed_dim)
    
    def forward(self, q, k, v, mask, coarse_pred=None, need_weights=False):
        """
        q.shape = (b, n, embed_dim)
        k.shape = (b, num_class, kdim)
        v.shape = (b, num_class, vdim)
        mask.shape = (b, 1, num_class)  float32
        coarse_pred.shape = (b, num_class, n)
        """
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        attn_map = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.embed_dim)
        attn_map = attn_map + mask
        attn_map = attn_map.softmax(dim=-1)

        if coarse_pred is not None:
            coarse_pred = coarse_pred.detach()
            coarse_pred = coarse_pred.transpose(1, 2).softmax(dim=-1)
            attn_map = torch.softmax(attn_map * coarse_pred + mask, dim=-1)
        
        output = torch.bmm(attn_map, v)
        
        if need_weights:
            return output, attn_map
        else:
            return output
