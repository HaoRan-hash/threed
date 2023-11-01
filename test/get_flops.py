import torch
from utils_func import *
from deepspeed.profiling.flops_profiler import get_model_profile
import sys
sys.path.append('/home/lindi/chenhr/threed/pointnet++/seg')
from memorynet_seg_x3x2_contrast import Memorynet
import time


if __name__ == '__main__':
    seed = 4464
    torch.manual_seed(seed)
    device = 'cuda:5'
    model = Memorynet(13).to(device)
    model.eval()
    
    b, n = 1, 24000
    pos = torch.randn((b, n, 3), device=device, dtype=torch.float32)
    x = torch.randn((b, n, 3), device=device, dtype=torch.float32)
    # y = torch.randint(0, 13, (b, n), device=device, dtype=torch.long)
    
    # flops, _, params = get_model_profile(model=model, args=[pos, x], print_profile=False, detailed=False, warm_up=10,
    #                                      as_string=False, output_file=None)
    # print(f'{params / 1e6}\t{flops / 1e9}')
    
    warmup_iter = 10
    for _ in range(warmup_iter):
        model(pos, x)
        # model(pos, x, y, 0)
    
    iter = 200
    st = time.time()
    for _ in range(iter):
        model(pos, x)
        # model(pos, x, y, 0)
        torch.cuda.synchronize(device=device)
    et = time.time()
    print(f'Inference time(s): {(et-st) / iter}')
    print(torch.cuda.max_memory_reserved(device=device))
