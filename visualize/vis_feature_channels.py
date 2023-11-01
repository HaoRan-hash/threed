import numpy as np
import torch
import torchvision
import cv2

if __name__ == '__main__':
    img_1 = torchvision.io.read_image('./door00_cut.png')[0:3]
    img_2 = torchvision.io.read_image('./door01_cut.png')[0:3]
    img_3 = torchvision.io.read_image('./door02_cut.png')[0:3]
    
    img_1 = img_1.to(dtype=torch.float32)
    img_2 = img_2.to(dtype=torch.float32)
    img_3 = img_3.to(dtype=torch.float32)
    
    resnet50 = torchvision.models.resnet50(pretrained=True)
    img_1_f = torchvision.transforms.functional.resize(resnet50(img_1.unsqueeze(dim=0)).squeeze(dim=0), (200, 200))
    img_2_f = torchvision.transforms.functional.resize(resnet50(img_2.unsqueeze(dim=0)).squeeze(dim=0), (200, 200))
    img_3_f = torchvision.transforms.functional.resize(resnet50(img_3.unsqueeze(dim=0)).squeeze(dim=0), (200, 200))
    
    img_1_f = img_1_f.max(dim=0)[0].detach().numpy()
    img_2_f = img_2_f.max(dim=0)[0].detach().numpy()
    img_3_f = img_3_f.max(dim=0)[0].detach().numpy()
    
    img_1_f = np.uint8(img_1_f / img_1_f.max() * 255)
    img_2_f = np.uint8(img_2_f / img_2_f.max() * 255)
    img_3_f = np.uint8(img_3_f / img_3_f.max() * 255)
    
    img_1_f = cv2.applyColorMap(img_1_f, cv2.COLORMAP_JET)
    img_2_f = cv2.applyColorMap(img_2_f, cv2.COLORMAP_JET)
    img_3_f = cv2.applyColorMap(img_3_f, cv2.COLORMAP_JET)
    
    img_1_f = cv2.addWeighted(img_1.permute(1, 2, 0).to(dtype=torch.uint8).numpy(), 0.5, img_1_f, 0.5, 0)
    img_2_f = cv2.addWeighted(img_2.permute(1, 2, 0).to(dtype=torch.uint8).numpy(), 0.5, img_2_f, 0.5, 0)
    img_3_f = cv2.addWeighted(img_3.permute(1, 2, 0).to(dtype=torch.uint8).numpy(), 0.5, img_3_f, 0.5, 0)
    
    cv2.imwrite('./img_1_f.png', img_1_f)
    cv2.imwrite('./img_2_f.png', img_2_f)
    cv2.imwrite('./img_3_f.png', img_3_f)
    