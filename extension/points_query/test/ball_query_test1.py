import torch
import points_query
import logging
import pointnet2_batch_cuda as pointnet2_cuda


def ball_query(radius, k, query_pos, all_pos):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    all_x.shape = (b, n, c)
    return shape = (b, sample, k, 3), (b, sample, k, c), (b, sample, k)
    """
    assert query_pos.is_contiguous()
    assert all_pos.is_contiguous()
    
    b, n, _ = all_pos.shape
    nsample = query_pos.shape[1]
    # idx = torch.cuda.IntTensor(b, nsample, k, device=query_pos.device)
    idx = torch.ones((b, nsample, k), device=query_pos.device, dtype=torch.int32)
    pointnet2_cuda.ball_query_wrapper(b, n, nsample, radius, k, query_pos, all_pos, idx)
    
    return idx


def my_ball_query(radius, k, query_pos, all_pos):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    return shape = (b, sample, k), (b, sample, k)
    """
    b, m, _ = query_pos.shape
    k_indices = torch.zeros((b, m, k), dtype=torch.int64, device='cuda:5')
    k_dis = torch.zeros((b, m, k), dtype=torch.float, device='cuda:5')
    
    points_query.ball_query(k, radius, all_pos, query_pos, k_indices, k_dis)
    return k_indices


if __name__ == '__main__':
    logging.basicConfig(filename='ball_query_test1.log', format='%(message)s', level=logging.INFO)
    
    b, m, n = 8, 600, 2400
    k = 16
    radius = 0.5
    query_pos = torch.randn((b, m, 3), device='cuda:5')
    all_pos = torch.randn((b, n, 3), device='cuda:5')
    
    k_indices_1 = ball_query(radius, k, query_pos, all_pos).to(dtype=torch.long)   # 不知道为什么一直是0
    logging.info(k_indices_1)
    # logging.info(k_dis)
    logging.info('------------------------------------')
    
    k_indices_2 = my_ball_query(radius, k, query_pos, all_pos)
    logging.info(k_indices_2)
    # logging.info(k_dis)
    
    # 看两个knn得到结果是否相同
    mask = k_indices_1 - k_indices_2
    print(mask.sum())
    