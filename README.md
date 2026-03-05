# Semantic Segmentation Research: SAM & LoRA Fine-Tuning

This repository explores the frontiers of semantic and panoptic segmentation by leveraging foundation models like Meta AI’s **Segment Anything Model (SAM)** and **EfficientSAM**. The primary focus of this research is implementing **Low-Rank Adaptation (LoRA)** to fine-tune these models on niche datasets, achieving high-precision results without the need for full parameter updates.

## Core Research Areas

### 1. Segment Anything Model (SAM)
Exploration of the original SAM architecture for zero-shot and prompted segmentation. This section includes benchmarks and basic implementation strategies for standard vision tasks.

### 2. EfficientSAM Optimization
Implementation of EfficientSAM to address the computational demands of real-time segmentation. By utilizing a masked image pre-training approach, this research demonstrates how to maintain high mIoU (Mean Intersection over Union) while significantly reducing latency.

### 3. LoRA Fine-Tuning Pipeline
The hallmark of this repository is the modular LoRA implementation. Instead of fine-tuning the entire transformer backbone, this project injects low-rank matrices into the attention layers, allowing for:
- **Reduced VRAM Usage**: Fine-tuning large models on consumer-grade hardware.
- **Fast Adaptation**: Rapidly specializing SAM for medical imagery, satellite data, or specific industrial use cases.

## Repository Structure

The project is organized into modular research tracks and production-ready scripts:

- **`LoRA_SAM/`**: The primary research hub containing the fine-tuning logic.
  - `scripts/`: Production-style Python scripts for structured workflows.
    - `SAM_Dataset.py`: Custom dataset wrappers for handling segmentation masks.
    - `LoRA_Config.py`: Parameterized configuration for rank (r) and alpha settings.
    - `training.py`: Modular training loop with checkpointing.
    - `inference.py`: Optimized inference engine for prompted segmentation.
- **`EfficientSAM/`**: Notebooks focused on the lightweight adaptation of the EfficientSAM architecture.
- **`SAM/`**: Baseline explorations and zero-shot prompting research.

## Technical Specifications

- **Frameworks**: PyTorch, Hugging Face Transformers
- **Architectures**: Vision Transformers (ViT), Mask2Former, SAM
- **Optimization**: LoRA (Low-Rank Adaptation), AdamW, Learning Rate Schedulers
- **Data Processing**: Custom COCO/Pascal VOC style data loaders

## Getting Started

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/Josh396s/semantic_segmentation_research.git](https://github.com/Josh396s/semantic_segmentation_research.git)
   ```
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
### Usage
To begin fine-tuning SAM with your own dataset using the LoRA pipeline:
```bash
python LoRA_SAM/scripts/training.py --config LoRA_SAM/scripts/LoRA_Config.py --data_path ./path_to_data
```

### Results & Visualizations
The research demonstrates that LoRA-fine-tuned SAM models outperform zero-shot SAM on domain-specific tasks while requiring less than 1% of the original model's trainable parameters.
