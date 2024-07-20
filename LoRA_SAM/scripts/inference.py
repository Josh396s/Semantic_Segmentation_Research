from transformers import SamModel, SamProcessor
from torch.utils.data import DataLoader
from statistics import mean
from SAM_Dataset import *
from LoRA_Config import *
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import torch

#Function that loads the model into the device
def load_model(model, device):
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
    model.eval()
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




def main(subset_size, model_path):
    test_img_path = "/home/cahsi/Josh/Semantic_Segmentation_Research/LoRA_SAM/qatacov19-dataset/QaTa-COV19/QaTa-COV19-v2/Test Set/Images/*.png"
    test_mask_path = "/home/cahsi/Josh/Semantic_Segmentation_Research/LoRA_SAM/qatacov19-dataset/QaTa-COV19/QaTa-COV19-v2/Test Set/Ground-truths/*.png"
    _, test_images, _, test_masks = read_images(test_img_path, test_img_path, test_mask_path, test_mask_path, test=True)
    test_images = np.array(resize_images(test_images, False))
    test_masks = np.array(resize_images(test_masks, True))
    # testing_dataset = make_dataset(test_images, test_masks, False, 0.01)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    # test_dataset = SAMDataset(dataset=testing_dataset, processor=processor)
    # test_dataloader = DataLoader(test_dataset, drop_last=False, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    original_sam_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = lora_config(model, ranking=3)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    trained_model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Original SAM total params: {original_sam_total_params}")
    print(f"LoRA-SAM total params: {trained_model_parameters}")
    print(model)
    #load_model(model, device)


    #plot_image(trained_model, training_dataset, processor, device)
    small_testing_dataset = make_dataset(test_images, test_masks, True, subset_size) #Smaller sample
    full_testing_dataset = make_dataset(test_images, test_masks, False, None) #Full dataset
    small_test_iou = calculate_IOU(model, small_testing_dataset, device, processor)
    full_test_iou = calculate_IOU(model, full_testing_dataset, device, processor)
    print(f"Average IoUs over {len(small_testing_dataset)} test sample: {small_test_iou}")
    print(f"Average IoUs over {len(full_testing_dataset)} test sample: {full_test_iou}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LoRA_SAM model evaluation')
    parser.add_argument('subset_size', default=0.05, type=float, help='Size of the subset of the dataset to use for evaluation (must be a float between (0.0,1.0))')
    parser.add_argument('model_path', type=str, help='Path of the trained model')
    args = parser.parse_args()
    
    main(args.subset_size, args.model_path)