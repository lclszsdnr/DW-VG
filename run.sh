export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch  --nproc_per_node=2  --master_port 35675 \
--use_env main.py --batch_size 8 --output_dir outputs/gpt_g2p0.25 \
#--resume /ssh/lcl/mdetr_cr/outputs/gpt_g3/checkpoint0007.pth

#export CUDA_VISIBLE_DEVICES=0,1
#python -m torch.distributed.launch  --nproc_per_node=2  --master_port 35675 \
#--use_env main.py --batch_size 16 --output_dir outputs/retriv_2 \
#--resume /ssh/lcl/mdetr_cr/outputs/s1_640/BEST_checkpoint.pth --eval
###

