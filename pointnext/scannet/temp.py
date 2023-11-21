import torch
import numpy as np


a = torch.load('/mnt/Disk16T/chenhr/threed_data/data/ScanNet/test/scene0707_00_inst_nostuff.pth', map_location='cpu')
print(a)