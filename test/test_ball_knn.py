import torch
import points_query
import logging


if __name__ == '__main__':
    logging.basicConfig(filename='test_01.log', format='%(message)s', level=logging.INFO)
    b, n, m, k = 8, 128, 64, 16
    radius = 1
    device = 'cuda:5'
    
    query_pos = torch.randn((b, m, 3), dtype=torch.float32, device=device)
    all_pos = torch.randn((b, n, 3), dtype=torch.float32, device=device)
    
    k_indices_1 = torch.zeros((b, m, k), dtype=torch.long, device=device)
    k_dis_1 = torch.zeros((b, m, k), dtype=torch.float32, device=device)
    
    # ball query
    points_query.ball_query(k, radius, all_pos, query_pos, k_indices_1, k_dis_1)
    logging.info(f'{k_indices_1}')
    logging.info(f'-----------------------------------------------')
    
    k_indices_2 = torch.zeros((b, m, k), dtype=torch.long, device=device)
    k_dis_2 = torch.zeros((b, m, k), dtype=torch.float32, device=device)
    
    # knn query
    points_query.knn_query(k, all_pos, query_pos, k_indices_2, k_dis_2)
    logging.info(f'{k_indices_2}')
    