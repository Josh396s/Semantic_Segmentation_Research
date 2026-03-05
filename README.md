# LoRA-SAM for Medical Image Segmentation

This repository provides a framework for fine-tuning Meta AI’s **Segment Anything Model (SAM)** using **Low-Rank Adaptation (LoRA)**. The implementation focuses on specializing SAM for medical imaging tasks, specifically lung infection segmentation in COVID-19 radiographs, while keeping the computational footprint minimal.

## Project Features
- **Target Model**: `facebook/sam-vit-base`.
- **Parameter Efficiency**: Implements LoRA in the Vision Encoder (Attention layers) and Mask Decoder (Self-Attention blocks).
- **Dataset**: Optimized for the `QaTa-COV19-v2` dataset.
- **Distributed Training**: Utilizes PyTorch **Distributed Data Parallel (DDP)** for multi-GPU acceleration.
- **Loss Function**: Employs **Dice-Cross Entropy (DiceCE)** loss from the MONAI framework.

## Repository Structure
- `LoRA_SAM/scripts/`
  - `LoRA_Config.py`: Logic for injecting LoRA layers into SAM's transformer blocks.
  - `SAM_Dataset.py`: Custom PyTorch Dataset that generates bounding box prompts from ground truth masks.
  - `training.py`: Main entry point for distributed training and checkpointing.
  - `inference.py`: Evaluation logic and IoU calculations.

## Usage
### Training
To launch distributed training:
```bash
python LoRA_SAM/scripts/training.py --epochs 10 --batch_size 4 --num_workers 4 --lora_rank 3 --data_root ./path_to_data

## Dataset Used
https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset/data
