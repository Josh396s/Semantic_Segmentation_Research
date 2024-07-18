from skimage.transform import resize
from datasets import Dataset
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import numpy as np 
import cv2

#Function that reads in images/masks and returns them as an numpy array
def read_images(train_img_path, test_img_path, train_mask_path, test_mask_path):
  # Load all images in the current folder that end with .png
  train_img = io.imread_collection(train_img_path)
  test_img = io.imread_collection(test_img_path)
  train_mask = io.imread_collection(train_mask_path)
  test_mask = io.imread_collection(test_mask_path)
  return(train_img, test_img, train_mask, test_mask)

#Function that resizes the image to (256, 256) for SAM input
def resize_images(images, mask):
  output = []
  if mask:
      for mask in images:
          # Perform resizing with nearest neighbor interpolation to maintain binary values
          resized_mask = (resize(mask, (256, 256), order=0, anti_aliasing=False) > 0.5).astype(np.uint8)
          output.append(resized_mask)
  else:
      for image in images:
          image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
          resized_image = resize(image, (256, 256), anti_aliasing=False)
          output.append(resized_image)
  return output

#Function that creates a dataset of the images/masks
def make_dataset(images, masks, subset=False, subset_size=0.2):
  if subset:
    sub = int(len(images)*subset_size)
    num_imgs = images[:sub]
    num_labels = masks[:sub]
    img = [Image.fromarray((img * 255).astype(np.uint8)) for img in num_imgs]
    label = [Image.fromarray(mask) for mask in num_labels]
  else:
    img = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
    label = [Image.fromarray(mask) for mask in masks]

  # Convert the NumPy arrays to Pillow images and store them in a dictionary
  training_dataset_dict = {
    "image": img,
    "mask": label,
  }
  training_dataset = Dataset.from_dict(training_dataset_dict)
  return training_dataset

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
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    image = self.dataset[idx]["image"]
    ground_truth_mask = [np.array(i) for i in self.dataset[idx]["mask"]]

    # get bounding box prompt
    prompt = [[get_bounding_box(i)] for i in ground_truth_mask]

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=prompt, return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}
    
    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs