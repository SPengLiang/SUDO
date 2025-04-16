# Intro

This is the training code for *SUDO: Enhancing Text-to-Image Diffusion Models with
Self-Supervised Direct Preference Optimization*. It is based on [Diffusion-DPO](https://arxiv.org/abs/2311.12908), thanks for the great work!

# Model Checkpoints

We provide the pre-trained SD 1.5 model at.

# Setup

`pip install -r requirements.txt`

# Structure

- `launchers/` is examples of running SD1.5 or SDXL training
- `utils/` has the scoring models for evaluation or AI feedback (PickScore, HPS, Aesthetics, CLIP)
- `quick_samples.ipynb` is visualizations from a pretrained model vs baseline
- `requirements.txt` Basic pip requirements
- `train.py` Main script, this is pretty bulky at >1000 lines, training loop starts at ~L1000 at this commit (`ctrl-F` "for epoch").
- `upload_model_to_hub.py` Uploads a model checkpoint to HF (simple utility, current values are placeholder)

# Running the training

Example SD1.5 launch

```bash
# from launchers/sd15.sh
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="yuvalkirstain/pickapic_v2"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="fp16"  train.py \
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
```

## Important Args

### General

- `--pretrained_model_name_or_path` what model to train/initalize from
- `--output_dir` where to save/log to
- `--seed` training seed (not set by default)
- `--sdxl` run SDXL training
- `--sft` run SFT instead of SUDO

### Optimizers/learning rates

- `--max_train_steps` How many train steps to take

- `--gradient_accumulation_steps`

- `--train_batch_size` see above notes in script for actual BS

- `--checkpointing_steps` how often to save model

- `--gradient_checkpointing` turned on automatically for SDXL

- `--learning_rate`
- `--lr_scheduler` Type of LR warmup/decay. Default is linear warmup to constant
- `--lr_warmup_steps` number of scheduler warmup steps
- `--use_adafactor` Adafactor over Adam (lower memory, default for SDXL)

### Data

- `--dataset_name` if you want to switch from Pick-a-Pic
- `--cache_dir` where dataset is cached locally **(users will want to change this to fit their file system)**
- `--resolution` defaults to 512 for non-SDXL, 1024 for SDXL.
- `--random_crop` and `--no_hflip` changes data aug
- `--dataloader_num_workers` number of total dataloader workers

# Ethical Considerations

This release is for research purposes only in support of an academic paper. Our models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP.
