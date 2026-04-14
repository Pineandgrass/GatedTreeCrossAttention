export CUDA_VISIBLE_DEVICES=0,1,2,3,4
device=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


REWEIGHT_COEFFICIENT=0.15

# torchrun --nproc_per_node=$device  --master-port 29502 probe_ckpte.py \
#   --model_name  /path/to/Qwen2.5-7B \
#   --ckpt_dirs /path/to/ckpt_switch_stage/stage1,/path/to/ckpt_switch_stage/stage2,/path/to/ckpt_switch_stage/stage3 \
#   --proj_dim 512 \
#   --epochs 100000 \
#   --n_train 3000 \
#   --n_test 500\
#   --auxloss_coefficient $AUXLOSS_COEFFICIENT \
#   --layers_ratio $LAYERS_RATIO \
#   --heads_ratio $HEADS_RATIO \
#   --reweight_coefficient $REWEIGHT_COEFFICIENT \
#   --best_ckpts /path/to/example.pt \
#   --probe_state_path /path/to/probe/probe.pt \
#   --stage train
torchrun --nproc_per_node=$device  --master-port 29502 probe_ckpt.py \
  --model_name  /path/to/qwen_hrnn/Qwen2.5-7B \
  --ckpt_dirs /path/to/ckpt_switch_stage/stage1,/path/to/ckpt_switch_stage/stage2,/path/to/ckpt_switch_stage/stage3 \
  --proj_dim 512 \
  --epochs 100000 \
  --n_train 3000 \
  --n_test 500\
  --best_ckpts /path/to/example.pt \
  --probe_state_path /path/to/probe/probe.pt \
  --stage test