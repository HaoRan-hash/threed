import numpy as np
import os
from torch.utils.data import Dataset
import json
import pickle
from pathlib import Path
from data_aug import *


class ShapeNet(Dataset):
    def __init__(self, root, split, npoints=2048, transforms=None):
        super(ShapeNet, self).__init__()
        self.npoints = npoints
        self.split = split
        self.transforms = transforms
        self.idx_to_class = {}
        dir_to_idx = {}

        with open(os.path.join(root, 'synsetoffset2category.txt'), 'r') as f:
            for i, line in enumerate(f):
                line = line.strip().split()
                self.idx_to_class[i] = line[0]
                dir_to_idx[line[1]] = i
        
        if self.split == 'train':
            self.files = []
            self.object_labels = []
            with open(os.path.join(root, 'train_test_split', f'shuffled_train_file_list.json'), 'r') as f:
                temp = json.load(f)   # type(temp) = list

            with open(os.path.join(root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
                temp += json.load(f)

            for x in temp:
                x = x.split('/')
                self.files.append(os.path.join(root, x[1], x[2]+'.txt'))
                self.object_labels.append(dir_to_idx[x[1]])
        else:
            with open(os.path.join(root, 'processed', 'test_2048_fps.pkl'), 'rb') as f:
                self.data, self.object_labels = pickle.load(f)
            
    def __len__(self):
        return len(self.object_labels)

    def __getitem__(self, index):
        if self.split == 'train':
            file = self.files[index]
            object_label = np.array(self.object_labels[index])
            points = np.genfromtxt(file, dtype=np.float32)
        else:
            points, object_label = self.data[index], self.object_labels[index][0]

        if self.split == 'train':
            choice = np.random.choice(len(points), self.npoints)
            points = points[choice]
        else:
            points = points[0:self.npoints]   # 为了可复现性
        
        pos, x, y = points[:, 0:3], points[:, 3:6], points[:, -1]
        if self.transforms:
            pos, x = self.transforms(pos, x)
        
        pos, x, y, object_label = pos.astype(np.float32), x.astype(np.float32), y.astype(np.int64), object_label.astype(np.int64)

        return pos, x, y, object_label


class S3dis(Dataset):
    def __init__(self, root, split, loop, npoints=24000, voxel_size=0.04, test_area=5, transforms=None):
        super(S3dis, self).__init__()
        self.root = root
        self.split = split
        self.loop = loop
        self.npoints = npoints
        self.voxel_size = voxel_size
        self.transforms = transforms
        self.idx_to_class = {0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam', 4: 'column', 
                5: 'window', 6: 'door', 7: 'table', 8: 'chair', 9: 'sofa', 10: 'bookcase', 11: 'board', 12: 'clutter'}
        
        room_list = os.listdir(root)
        if split == 'train':
            self.room_list = list(filter(lambda x : f'Area_{test_area}' not in x, room_list))
        else:
            self.room_list = list(filter(lambda x : f'Area_{test_area}' in x, room_list))
    
    def __len__(self):
        return len(self.room_list) * self.loop

    def fnv_hash_vec(self, arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * \
            np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

    def voxel_grid_sampling(self, pos):
        """
        pos.shape = (n, 3)
        """
        voxel_indices = np.floor(pos / self.voxel_size)
        
        voxel_hash = self.fnv_hash_vec(voxel_indices)
        sort_idx = voxel_hash.argsort()
        hash_sort = voxel_hash[sort_idx]
        
        _, counts = np.unique(hash_sort, return_counts=True)
        if self.split == 'test':   # test时需要的东西和train，val时不同
            return sort_idx, counts
        
        idx_select = np.cumsum(np.insert(counts, 0, 0)[0:-1]) + np.random.randint(0, counts.max(), counts.size) % counts
        return sort_idx[idx_select]
    
    def __getitem__(self, index):
        room = os.path.join(self.root, self.room_list[index % len(self.room_list)])
        points = np.load(room)
        
        # 大家都这样做
        points[:, 0:3] = points[:, 0:3] - np.min(points[:, 0:3], axis=0)
        
        if self.split == 'test':
            sort_idx, counts = self.voxel_grid_sampling(points[:, 0:3])
            pos, x, y = points[:, 0:3], points[:, 3:-1], points[:, -1]
            pos, x, y = pos.astype(np.float32), x.astype(np.float32), y.astype(np.int64)
            return pos, x, y, sort_idx, counts
        
        # train, val的流程
        sample_indices = self.voxel_grid_sampling(points[:, 0:3])
        pos, x, y = points[sample_indices, 0:3], points[sample_indices, 3:-1], points[sample_indices, -1]
        
        # 是否指定了npoints
        if self.npoints:
            n = len(sample_indices)
            if n > self.npoints:
                init_idx = np.random.randint(n)
                crop_indices = np.argsort(np.sum(np.square(pos - pos[init_idx]), 1))[:self.npoints]
            elif n < self.npoints:
                temp = np.arange(n)
                pad_choice = np.random.choice(n, self.npoints - n)
                crop_indices = np.hstack([temp, temp[pad_choice]])
            else:
                crop_indices = np.arange(n)
            
            # 打乱
            np.random.shuffle(crop_indices)
        
            pos, x, y = pos[crop_indices], x[crop_indices], y[crop_indices]
        
        pos = pos - pos.min(0)
        if self.transforms:
            pos, x = self.transforms(pos, x)
        
        pos, x, y = pos.astype(np.float32), x.astype(np.float32), y.astype(np.int64)
        return pos, x, y


class Scannet(Dataset):
    def __init__(self, root, split, loop, npoints=64000, voxel_size=0.02, transforms=None):
        super(Scannet, self).__init__()
        self.root = Path(root)
        with open('/mnt/Disk16T/chenhr/threed_data/data/scannetv2_train.txt', 'r') as file:
            scan_train = [line.strip() for line in file.readlines()]
        with open('/mnt/Disk16T/chenhr/threed_data/data/scannetv2_val.txt', 'r') as file:
            scan_val = [line.strip() for line in file.readlines()]
        self.split = split
        if split == 'train':
            self.room_list = [self.root / f"{p}.pt" for p in scan_train]
        elif split == 'val' or split == 'val_test':
            self.room_list = [self.root / f"{p}.pt" for p in scan_val]
        elif split == 'trainval':
            scan_tarinval = scan_train + scan_val
            self.room_list = [self.root / f"{p}.pt" for p in scan_tarinval]
        else:
            raise ValueError(f'Not support {split} type')

        self.loop = loop
        self.npoints = npoints
        self.voxel_size = voxel_size
        self.transforms = transforms
        self.idx_to_class = {0: 'wall', 1: 'floor', 2: 'cabinet', 3: 'bed', 4: 'chair', 
                5: 'sofa', 6: 'table', 7: 'door', 8: 'window', 9: 'bookshelf', 10: 'picture', 11: 'counter', 12: 'desk',
                13: 'curtain', 14: 'refrigerator', 15: 'shower curtain', 16: 'toilet', 17: 'sink', 18: 'bathtub', 19: 'otherfurniture'}
    
    def __len__(self):
        return len(self.room_list) * self.loop
    
    def fnv_hash_vec(self, arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * \
            np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

    def voxel_grid_sampling(self, pos):
        """
        pos.shape = (n, 3)
        """
        voxel_indices = np.floor(pos / self.voxel_size)
        
        voxel_hash = self.fnv_hash_vec(voxel_indices)
        sort_idx = voxel_hash.argsort()
        hash_sort = voxel_hash[sort_idx]
        
        _, counts = np.unique(hash_sort, return_counts=True)
        if self.split == 'val_test':   # test时需要的东西和train，val时不同
            return sort_idx, counts
        
        idx_select = np.cumsum(np.insert(counts, 0, 0)[0:-1]) + np.random.randint(0, counts.max(), counts.size) % counts
        return sort_idx[idx_select]
    
    def __getitem__(self, index):
        room = self.room_list[index % len(self.room_list)]
        pos, color, normal, y = torch.load(room)   # color的范围是[0, 255]
        pos, color, normal, y = pos.numpy(), color.numpy(), normal.numpy(), y.numpy()
        
        if self.transforms:
            pos, color, normal = self.transforms(pos, color, normal)
        pos = pos - pos.min(0)
        
        if self.split == 'val_test':
            sort_idx, counts = self.voxel_grid_sampling(pos)
            pos, color, normal, y = pos.astype(np.float32), color.astype(np.float32), normal.astype(np.float32), y.astype(np.int64)
            return pos, color, normal, y, sort_idx, counts

        # train, val的流程
        sample_indices = self.voxel_grid_sampling(pos)
        pos, color, normal, y = pos[sample_indices], color[sample_indices], normal[sample_indices], y[sample_indices]
        
        # 是否指定了npoints
        if self.npoints:
            n = len(sample_indices)
            if n > self.npoints:
                init_idx = np.random.randint(n)
                crop_indices = np.argsort(np.sum(np.square(pos - pos[init_idx]), 1))[:self.npoints]
            elif n < self.npoints:
                temp = np.arange(n)
                pad_choice = np.random.choice(n, self.npoints - n)
                crop_indices = np.hstack([temp, temp[pad_choice]])
            else:
                crop_indices = np.arange(n)
            
            # 打乱
            np.random.shuffle(crop_indices)
        
            pos, color, normal, y = pos[crop_indices], color[crop_indices], normal[crop_indices], y[crop_indices]
        pos = pos - pos.min(0)
        
        pos, color, normal, y = pos.astype(np.float32), color.astype(np.float32), normal.astype(np.float32), y.astype(np.int64)
        return pos, color, normal, y
