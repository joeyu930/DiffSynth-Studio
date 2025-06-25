python train3.py \
  --task train \
  --train_architecture lora \
  --dataset_path ~/dataset/s3/train \
  --output_path ./models \
  --dit_path "models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 960 \
  --max_epochs 10 \
  --learning_rate 1e-4 \
  --accumulate_grad_batches 64  \
  --lora_rank 64 \
  --lora_alpha 64 \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --use_gradient_checkpointing  --num_frames 41 --height 512 --width 288


