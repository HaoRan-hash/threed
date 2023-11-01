import numpy as np
import torch
import torchvision

if __name__ == '__main__':
    img_1 = torchvision.io.read_image('./door00_cut.png')[0:3]
    img_2 = torchvision.io.read_image('./door01_cut.png')[0:3]
    img_3 = torchvision.io.read_image('./door02_cut.png')[0:3]
    img_4 = torchvision.io.read_image('./door03_cut.png')[0:3]
    img_5 = torchvision.io.read_image('./door04_cut.png')[0:3]
    
    mix_ratio_1 = [0.1, 0.1, 0.2, 0.2, 0.4]
    mix_ratio_2 = [0.1, 0.2, 0.4, 0.2, 0.1]
    mix_ratio_3 = [0.4, 0.1, 0.1, 0.2, 0.2]
    
    mix_img_1 = img_1 * mix_ratio_1[0] + img_2 * mix_ratio_1[1] + img_3 * mix_ratio_1[2] + img_4 * mix_ratio_1[3] + img_5 * mix_ratio_1[4]
    mix_img_2 = img_1 * mix_ratio_2[0] + img_2 * mix_ratio_2[1] + img_3 * mix_ratio_2[2] + img_4 * mix_ratio_2[3] + img_5 * mix_ratio_2[4]
    mix_img_3 = img_1 * mix_ratio_3[0] + img_2 * mix_ratio_3[1] + img_3 * mix_ratio_3[2] + img_4 * mix_ratio_3[3] + img_5 * mix_ratio_3[4]
    
    torchvision.io.write_png(mix_img_1.to(dtype=torch.uint8), './mix_img_1.png')
    torchvision.io.write_png(mix_img_2.to(dtype=torch.uint8), './mix_img_2.png')
    torchvision.io.write_png(mix_img_3.to(dtype=torch.uint8), './mix_img_3.png')
    