from skimage.transform import resize
from datasets import Dataset
from PIL import Image
import cv2
import glob

# Get paths for image and mask data
def get_paths(img_dir, mask_dir):
    img_paths = sorted(glob.glob(img_dir))
    mask_paths = sorted(glob.glob(mask_dir))
    return img_paths, mask_paths

#Function that gets bounding boxes from masks
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]
  return bbox

#Function that creates a SAMDataset to be used for training
class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes an image path, mask path, and processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, img_paths, mask_paths, processor):
    self.img_paths = img_paths
    self.mask_paths = mask_paths
    self.processor = processor

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    
    # Change color to RGB and resize images 
    image = cv2.imread(self.img_paths[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))

    # Resize images 
    mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
    mask = (cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    
    # Get bounding box prompt
    prompt = [get_bounding_box(mask)]

    # Prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[prompt], return_tensors="pt")

    # Remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}
    
    # Add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs
