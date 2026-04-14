export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# DEVICES=1
AUXLOSS_COEFFICIENT=0.5
LAYERS_RATIO=0.5
HEADS_RATIO=0.3
REWEIGHT_COEFFICIENT=0.15

torchrun --nproc_per_node=4  --master-port 29504 probe_backbone.py \
  --model_name  /path/to/Qwen2.5-7B \
  --lora /path/to/checkpoint \
  --ckpt_dirs \,\ \
  --proj_dim 512 \
  --epochs 100000 \
  --n_train 3000 \
  --n_test 500\
  --auxloss_coefficient $AUXLOSS_COEFFICIENT \
  --layers_ratio $LAYERS_RATIO \
  --heads_ratio $HEADS_RATIO \
  --reweight_coefficient $REWEIGHT_COEFFICIENT \
  --best_ckpts examle.ckpt \
  --probe_state_path /path/to/probe.pt \
  --stage train
