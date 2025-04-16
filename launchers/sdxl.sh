export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="yuvalkirstain/pickapic_v2"


accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=3 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=16 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --learning_rate=1e-6 \
  --cache_dir="/export/share/datasets/vision_language/pick_a_pic_v2/" \
  --checkpointing_steps 200 \
  --beta_dpo 5000 \
   --sdxl  \
  --output_dir="sudo-sdxl"
  
