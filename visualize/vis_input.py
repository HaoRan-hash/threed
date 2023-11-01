import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import S3dis
from data_aug import Compose
from data_aug_tensor import *


def save_input(dataloader, area):
    # 获取数据
    temp = 1
    dataloader = iter(dataloader)
    for i in range(temp):
        pos, x, y, sort_idx, counts = next(dataloader)
    
    pos = pos.squeeze().numpy()
    x = x.squeeze().numpy()
    
    np.savetxt(f'area{area}_input{temp}.txt', np.concatenate((pos, x), axis=1), delimiter=',')


if __name__ == '__main__':
    test_dataset = S3dis('/home/lindi/chenhr/threed/data/processed_s3dis', split='test', loop=1, test_area=3)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    save_input(test_dataloader, 3)
    