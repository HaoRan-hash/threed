CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 scannet/pointnet2_scannet.py
