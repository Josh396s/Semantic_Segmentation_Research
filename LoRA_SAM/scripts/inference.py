import torch
import numpy as np
import cv2
import os
import glob
import argparse
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor
from SAM_Dataset import SAMDataset, get_paths
from LoRA_Config import lora_config

def calculateIoU(ground_mask, pred_mask):
    """
    Calculate the Intersection over Union (IoU) between two binary masks.
    Args:
        ground_mask (numpy array): The ground truth binary mask.
        pred_mask (numpy array): The predicted binary mask.
    Returns:
        float: The IoU score between the two masks.
    """
    intersection = np.logical_and(ground_mask, pred_mask).sum()
    union = np.logical_or(ground_mask, pred_mask).sum()
    if union == 0:
        return 0
    return intersection / union

#Function that runs the IOU over testing examples and returns the value
def run_evaluation(model, dataloader, device):
    """
    Evaluate the model on the given dataset and calculate the average IoU.
    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataset (SAMDataset): The dataset to evaluate on.
        device (torch.device): The device to run the evaluation on.
        processor (SamProcessor): The processor for preparing inputs for the model.
    Returns:
        float: The average IoU score over the dataset.
    """
    model.eval()
    model.to(device)
    ious = []
    
    print(f"Running evaluation on {len(dataloader.dataset)} images...")
    with torch.no_grad():
        for batch in dataloader:
            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_boxes=batch["input_boxes"].to(device),
                multimask_output=False
            )
            
            # Post-process predictions
            pred_masks = torch.sigmoid(outputs.pred_masks.squeeze(1))
            pred_masks = (pred_masks.cpu().numpy() > 0.5).astype(np.uint8)
            
            # Ground truth from batch
            gt_masks = batch["ground_truth_mask"].numpy()
            
            # Calculate IoU for each image in the batch
            for i in range(pred_masks.shape[0]):
                ious.append(calculateIoU(gt_masks[i], pred_masks[i].squeeze()))
                
    return np.mean(ious)

def visualize_prediction(model, dataset, device):
    """
    Visualize a random prediction from the model on the dataset.
    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataset (SAMDataset): The dataset to evaluate on.
        device (torch.device): The device to run the evaluation on.
        processor (SamProcessor): The processor for preparing inputs for the model.
    """
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]
    
    # Prepare input
    inputs = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if k != "ground_truth_mask"}
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    
    # Process prediction
    pred_mask = (torch.sigmoid(outputs.pred_masks).cpu().numpy() > 0.5).astype(np.uint8).squeeze()
    gt_mask = sample["ground_truth_mask"].numpy()
    
    # Plotting
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(gt_mask, cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title("LoRA-SAM Prediction")
    for ax in axes: ax.axis('off')
    plt.show()

def main(args):
    """
    Main function to run the evaluation and optional visualization.
    Args:
        args: Command-line arguments containing model path, data root, batch size, LoRA rank, and visualization flag.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Get Paths and Initialize Dataset
    test_img_list, test_mask_list = get_paths(
        os.path.join(args.data_root, "Test Set/Images/*.png"),
        os.path.join(args.data_root, "Test Set/Ground-truths/*.png")
    )
    
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    test_dataset = SAMDataset(test_img_list, test_mask_list, processor)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Load Model and LoRA Weights
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model = lora_config(model, args.lora_rank)
    
    # Load your trained checkpoint
    print(f"Loading checkpoint from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    # Evaluate
    avg_iou = run_evaluation(model, test_dataloader, device)
    print(f"Finished! Average IoU on Test Set: {avg_iou:.4f}")
    
    # Optional Visualization
    if args.visualize:
        visualize_prediction(model, test_dataset, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to checkpoint.pt")
    parser.add_argument('--data_root', type=str, required=True, help="Root of QaTa-COV19 dataset")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=3)
    parser.add_argument('--visualize', action='store_true', help="Plot a random result after eval")
    args = parser.parse_args()
    
    main(args)