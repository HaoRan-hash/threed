import torch

if __name__ == '__main__':
    device = 'cuda:7'
    model_path = '/home/lindi/chenhr/threed/pointnet++/seg/checkpoints/memorynet_seg_x3x2_contrast_length256.pth'
    ckpt = torch.load(model_path, map_location=device)

    save_model_path = '/home/lindi/chenhr/threed/pointnet++/seg/checkpoints/memorynet_seg_x3x2_contrast_length16_2.pth'
    memory1_memory = ckpt['model_state_dict']['memory1.memory']
    memory2_memory = ckpt['model_state_dict']['memory2.memory']
    
    # memory1_memory = memory1_memory.unflatten(dim=1, sizes=(16, 16)).mean(dim=2)
    # memory2_memory = memory2_memory.unflatten(dim=1, sizes=(16, 16)).mean(dim=2)
    dim1, dim2, dim3 = memory1_memory.shape
    memory1_memory_expand = torch.zeros((dim1, dim2 * 4, dim3), dtype=torch.float32, device=device)
    memory2_memory_expand = torch.zeros((dim1, dim2 * 4, dim3), dtype=torch.float32, device=device)
    memory1_memory_expand[:, 0:dim2, :], memory2_memory_expand[:, 0:dim2, :] = memory1_memory * 0.98, memory2_memory * 0.98
    memory1_memory_expand[:, dim2:dim2*2, :], memory2_memory_expand[:, dim2:dim2*2, :] = memory1_memory * 0.99, memory2_memory * 0.99
    memory1_memory_expand[:, dim2*2:dim2*3, :], memory2_memory_expand[:, dim2*2:dim2*3, :] = memory1_memory * 1.01, memory2_memory * 1.01
    memory1_memory_expand[:, dim2*3:dim2*4, :], memory2_memory_expand[:, dim2*3:dim2*4, :] = memory1_memory * 1.02, memory2_memory * 1.02
    
    ckpt['model_state_dict']['memory1.memory'] = memory1_memory_expand
    ckpt['model_state_dict']['memory2.memory'] = memory2_memory_expand
    
    torch.save(ckpt, save_model_path)
    