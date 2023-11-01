import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/lindi/chenhr/threed/pointnext')
from dataset import S3dis
from data_aug import *
from data_aug_tensor import *
from memorynet_seg_v1_x3x2_poly_pe_contrast import Memorynet, multi_scale_test


def six_fold_test(model, device):
    test_aug = Compose([PointCloudFloorCenteringTensor(),
                        ColorNormalizeTensor()])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
    overall_cm = torch.zeros((13, 13), device=device, dtype=torch.long)
    log_dir = 'seg/six_fold_test.log'
    
    for i in range(1, 7):
        test_dataset = S3dis('/home/lindi/chenhr/threed/data/processed_s3dis', split='test', loop=1, test_area=i)
        idx_to_class = test_dataset.idx_to_class
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
        model_path = f'seg/all_area_ckpt/area_{i}.pth'
        
        cm = multi_scale_test(test_dataloader, test_aug, model, loss_fn, device, model_path, log_dir)
        overall_cm += cm
    
    oa = (overall_cm.diag().sum() / overall_cm.sum()).item()
    macc = (overall_cm.diag() / overall_cm.sum(dim=1)).mean().item()
    iou = overall_cm.diag() / (overall_cm.sum(dim=0) + overall_cm.sum(dim=1) - overall_cm.diag())
    miou = iou.mean().item()
    
    with open(log_dir, mode='a') as f:
        for i in range(13):
            f.write(f'{idx_to_class[i]}:   {iou[i]:.4f}\n')
        f.write(f'miou={miou:.4f}, oa={oa:.4f}, macc={macc:.4f}\n')
        f.write('-------------------------------------------------------\n')
    print(f'miou={miou:.4f}, oa={oa:.4f}, macc={macc:.4f}')


if __name__ == '__main__':
    device = 'cuda:7'
    memorynet = Memorynet(13, 4, 32, [3, 6, 3, 3]).to(device)
    six_fold_test(memorynet, device)
