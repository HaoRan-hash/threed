CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --use_env --nproc_per_node=4 scannet/memorynet_scannet.py --use_ddp
