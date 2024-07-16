from transformers import SamProcessor
from transformers import SamModel
from statistics import mean
from tqdm import tqdm
from scripts.SAM_Dataset import *
from scripts.LoRA_Config import *
import argparse
import random
import torch
import monai


def train_model(model, train_dataloader, loss_func, optimizer, num_epochs):
    mean_epoch_loss = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False,
                            )
            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = loss_func(predicted_masks, ground_truth_masks.unsqueeze(1))
            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()
            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
        mean_epoch_loss.append(mean_epoch_loss)

def main(num_epochs):
    #Read in Images
    train_img_path = "/home/cahsi/Josh/Semantic_Segmentation_Research/LoRA_SAM/qatacov19-dataset/QaTa-COV19/QaTa-COV19-v2/Train Set/Images/*.png"
    test_img_path = "/home/cahsi/Josh/Semantic_Segmentation_Research/LoRA_SAM/qatacov19-dataset/QaTa-COV19/QaTa-COV19-v2/Test Set/Images/*.png"
    train_mask_path = "/home/cahsi/Josh/Semantic_Segmentation_Research/LoRA_SAM/qatacov19-dataset/QaTa-COV19/QaTa-COV19-v2/Train Set/Ground-truths/*.png"
    test_mask_path = "/home/cahsi/Josh/Semantic_Segmentation_Research/LoRA_SAM/qatacov19-dataset/QaTa-COV19/QaTa-COV19-v2/Test Set/Ground-truths/*.png"
    train_img, test_imgs, train_mask, test_masks = read_images(train_img_path, test_img_path, train_mask_path, test_mask_path)
    #Images Info
    print("Length of training raw images: " + str(len(train_img)) + "      Shape of an training raw image: " + str(train_img[0].shape))
    print("Length of training mask images: " + str(len(train_mask)) + "     Shape of an training mask image: " + str(train_mask[0].shape))
    print("Length of test raw images: " + str(len(test_imgs)) + "       Shape of an test raw image: " + str(test_imgs[0].shape))
    print("Length of test mask images: " + str(len(test_masks)) + "      Shape of an test mask image: " + str(test_masks[0].shape))

    #Resize images to 256x256
    train_images = np.array(resize_images(train_img, False))
    test_imgs = np.array(resize_images(test_imgs, False))
    train_masks = np.array(resize_images(train_mask, True))
    test_masks = np.array(resize_images(test_masks, True))

    #Print shape of resized images
    print("Shape of resized training raw image: " + str(train_images.shape))
    print("Shape of resized training mask image: " + str(train_masks.shape))
    print("Shape of resized testing raw image: " + str(test_imgs.shape))
    print("Shape of resized testing mask image: " + str(test_masks.shape))

    #Create training dataset
    training_dataset = make_dataset(train_images, train_masks)

    #Check dataset info
    print("Train Dataset Info:\n" + str(training_dataset))

    #Initialize the processor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    #Create instance of the SAMDataset class
    train_dataset = SAMDataset(dataset=training_dataset, processor=processor)

    #Load SAM model
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    
    #Original number of parameters
    original_sam_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    #Implement LoRA to model
    model = lora_config(model, rank=3)

    #Print difference in parameters before/after LoRA
    print(f"Original SAM total params: {original_sam_total_params}")
    sam_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA-SAM total params: {sam_total_params}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM Model")
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of epochs for training"
    )
    args = parser.parse_args()
    main(args.num_epochs)