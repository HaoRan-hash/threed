CUDA_VISIBLE_DEVICES=3,7 python -m torch.distributed.launch --use_env --nproc_per_node=2 scannet/pointmeta_scannet.py --use_ddp
