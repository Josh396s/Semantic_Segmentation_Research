import numpy as np 
import cv2
import torch
from torch.utils.data import Dataset

#Function that gets bounding boxes from masks
def get_bounding_box(ground_truth_map):
  """
  This function takes a binary mask as input and returns a bounding box that encompasses the object in the mask.
  Args:
    ground_truth_map: 2D numpy array representing the binary mask of the object to be segmented
  """
  y_indices, x_indices = np.where(ground_truth_map > 0)
  
  # Handle empty masks
  if len(x_indices) == 0: # Handle empty masks
    return [0, 0, 256, 256]
  
  # Get bounding box from mask
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)

  # Add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))

  return [x_min, y_min, x_max, y_max]

#Function that creates a SAMDataset to be used for training
class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, img_paths, mask_paths, processor):
    self.img_paths = img_paths
    self.mask_paths = mask_paths
    self.processor = processor

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    # Load single image/mask from disk
    image = cv2.imread(self.img_paths[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
    
    # Resize to SAM's expected 256x256
    image_resized = cv2.resize(image, (256, 256))
    mask_resized = (cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    
    # Get bounding box prompt from mask
    prompt = [get_bounding_box(mask_resized)]

    # Prepare image and prompt for the model
    inputs = self.processor(image_resized, input_boxes=[prompt], return_tensors="pt")
    
    # Remove batch dimension which the processor adds by default
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    
    # Add ground truth
    inputs["ground_truth_mask"] = torch.from_numpy(mask_resized)

    return inputs