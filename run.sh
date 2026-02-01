# Training
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch  --nproc_per_node=2  --master_port 35675 \
--use_env main.py --batch_size 8 --output_dir outputs/XXX \
#--loda /XXX/checkpoint0007.pth

# Evaluation
#export CUDA_VISIBLE_DEVICES=0,1
#python -m torch.distributed.launch  --nproc_per_node=2  --master_port 35675 \
#--use_env main.py --batch_size 16 --output_dir outputs/retriv_2 \
#--resume /XXX/BEST_checkpoint.pth --eval


