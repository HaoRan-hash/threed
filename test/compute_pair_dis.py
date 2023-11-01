import torch

def get_square_distance(query_pos, all_pos):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    return shape = (b, sample, n)
    """
    res = ((query_pos.unsqueeze(dim=2) - all_pos.unsqueeze(dim=1)) ** 2).sum(dim=-1)
    return res

b, n, sample = 1, 1, 1
torch.manual_seed(1)
query = torch.randn((b, sample, 3), device='cuda:1')
all_pos = torch.randn((b, n, 3), device='cuda:1')

# res = get_square_distance(query, all_pos)
res = torch.cdist(query, all_pos)
print(res)
