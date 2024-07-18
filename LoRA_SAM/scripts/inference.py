from statistics import mean
from SAM_Dataset import *
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

#Function that loads the model into the device
def load_model(model, device):
    device = device if torch.cuda.is_available() else "cpu"
    model.to(device)

#Function that calculates the IOU of given examples
def calculateIoU(ground_mask, pred_mask):
        # Calculate the TP, FP, FN
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(ground_mask)):
            for j in range(len(ground_mask[0])):
                if ground_mask[i][j] == 1 and pred_mask[i][j] == 1:
                    TP += 1
                elif ground_mask[i][j] == 0 and pred_mask[i][j] == 1:
                    FP += 1
                elif ground_mask[i][j] == 1 and pred_mask[i][j] == 0:
                    FN += 1
        # Calculate IoU
        iou = TP / (TP + FP + FN)
        return iou

#Function that runs the IOU over testing examples and returns the value
def calculate_IOU(model, dataset, device, processor):
    test_ious = []
    model.to(device)
    for idx, sample in enumerate(dataset):
        # Get Image and ground truth mask
        image = sample["image"]
        ground_truth_mask = np.array(sample["mask"])

        # get box prompt based on ground truth segmentation map
        prompt = get_bounding_box(ground_truth_mask)

        # prepare image + box prompt for the model
        inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
        #inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # forward pass
        with torch.no_grad():
          outputs = model(**inputs, multimask_output=False)
        # apply sigmoid
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        iou = calculateIoU(ground_truth_mask, medsam_seg)
        test_ious.append(iou)
    return mean(test_ious)

#Function that places label on top of image
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


#Function that plots the image with the label
def plot_image(model, training_dataset, processor, device):
    # let's take a random training example
    idx = random.randint(0, len(training_dataset)-1)

    # load image
    test_image = training_dataset[idx]["image"]
    test_image = np.array(test_image.convert("RGB"))

    # get box prompt based on ground truth segmentation map
    ground_truth_mask = np.array(training_dataset[idx]["mask"])
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image + box prompt for the model
    inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")

    # Move the input tensor to the GPU if it's not already there
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    # apply sigmoid
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    # convert soft mask to hard mask
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)


    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    # Plot the first image on the left
    axes[0].imshow(test_image, cmap='gray')  # Assuming the first image is grayscale
    show_mask(np.array(ground_truth_mask), axes[0])
    axes[0].set_title("Ground Truth Mask")

    # Plot the second image on the right
    axes[1].imshow(test_image, cmap='gray')  # Assuming the second image is grayscale
    show_mask(np.array(medsam_seg), axes[1])
    axes[1].set_title("Predicted Mask")

    # Hide axis ticks and labels
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Display the images side by side
    plt.show()