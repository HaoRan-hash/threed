# CUDA_VISIBLE_DEVICES=3,7 python -m torch.distributed.launch --use_env --nproc_per_node=2 scannet/memorynet_scannet.py --use_ddp
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --use_env --nproc_per_node=4 scannet/memorynet_scannet_trainval.py --use_ddp
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --use_env --nproc_per_node=2 scannet/visualize_nonorm.py --use_ddp