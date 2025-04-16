export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="yuvalkirstain/pickapic_v2"


CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=32 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=32 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-6 \
  --cache_dir="/export/share/datasets/vision_language/pick_a_pic_v2/" \
  --checkpointing_steps 500 \
  --beta_dpo 5000 \
  --output_dir="sudo-sd15"

