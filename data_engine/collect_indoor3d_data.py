import os
import numpy as np


name2class = {'ceiling': 0, 'floor': 1, 'wall': 2, 'beam': 3,
              'column': 4, 'window': 5, 'door': 6, 'table': 7,
              'chair': 8, 'sofa': 9, 'bookcase': 10, 'board': 11, 'clutter': 12}
data_path = 'data/Stanford3dDataset_v1.2_Aligned_Version'
anno_path = 'data/Stanford3dDataset_v1.2_Aligned_Version/anno_paths.txt'
output_path = 'data/processed_s3dis'


def processed_s3dis():
    anno_folders = np.loadtxt(anno_path, dtype=np.str)
    
    for folder in anno_folders:
        folder = str(folder)
        temp = folder.split('/')
        
        output_file = output_path + '/' + temp[0] + '_' + temp[1] + '.npy'
        cur_folder = data_path + '/' + folder
        
        anno_files = os.listdir(cur_folder)
        points = []
        for file in anno_files:
            if file == 'Icon' or file == '.DS_Store':
                continue
            
            cur_file = cur_folder + '/' + file
            try:
                cur_points = np.loadtxt(cur_file, dtype=np.float32)
            except ValueError as e:
                print(cur_file)
                print(e)
                print(cur_points.shape)
            
            cls_name = file.split('_')[0]
            if cls_name not in name2class:
                cls_name = 'clutter'
            cls = name2class[cls_name] * np.ones((cur_points.shape[0], 1), dtype=np.float32)
            cur_points = np.concatenate((cur_points, cls), axis=-1)
            
            points.append(cur_points)
        points = np.concatenate(points, axis=0)
        np.save(output_file, points)
            

if __name__ == '__main__':
    processed_s3dis()
