from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import SamProcessor
from transformers import SamModel
from inference import calculate_IOU, load_model
from SAM_Dataset import *
from LoRA_Config import *
from torch.optim import Adam
from statistics import mean
from tqdm import tqdm
import torch.multiprocessing as mp
import argparse
import torch
import monai
import sys
import os

#Setup process group for DDP
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train_model(model, train_dataloader, loss_func, optimizer, num_epochs, rank, ranking):
    mean_epoch_loss = []

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    model.train()

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(rank),
                            input_boxes=batch["input_boxes"].to(rank),
                            multimask_output=False,
                            )
            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(rank)
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
    
    if rank == 0:
        #Save the model
        ckp = model.module.state_dict()
        PATH = f"checkpoint.pt_epochs_{num_epochs}_meanloss_{mean(epoch_losses)}"
        torch.save(ckp, PATH)
        print(f"Training checkpoint saved at {PATH}")


def main(rank:int, world_size:int, num_epochs:int, batch_size:int):
    #Create process group
    ddp_setup(rank, world_size)

    #Read in Images
    train_img_path = "/home/cahsi/Josh/Semantic_Segmentation_Research/LoRA_SAM/qatacov19-dataset/QaTa-COV19/QaTa-COV19-v2/Train Set/Images/*.png"
    #test_img_path = "/home/cahsi/Josh/Semantic_Segmentation_Research/LoRA_SAM/qatacov19-dataset/QaTa-COV19/QaTa-COV19-v2/Test Set/Images/*.png"
    train_mask_path = "/home/cahsi/Josh/Semantic_Segmentation_Research/LoRA_SAM/qatacov19-dataset/QaTa-COV19/QaTa-COV19-v2/Train Set/Ground-truths/*.png"
    #test_mask_path = "/home/cahsi/Josh/Semantic_Segmentation_Research/LoRA_SAM/qatacov19-dataset/QaTa-COV19/QaTa-COV19-v2/Test Set/Ground-truths/*.png"
    train_img, _, train_mask, _ = read_images(train_img_path, train_img_path, train_mask_path, train_img_path, test=False)
    #Images Info
    print("Length of training raw images: " + str(len(train_img)) + "      Shape of an training raw image: " + str(train_img[0].shape))
    print("Length of training mask images: " + str(len(train_mask)) + "     Shape of an training mask image: " + str(train_mask[0].shape))
    #print("Length of test raw images: " + str(len(test_images)) + "       Shape of an test raw image: " + str(test_images[0].shape))
    #print("Length of test mask images: " + str(len(test_masks)) + "      Shape of an test mask image: " + str(test_masks[0].shape))

    #Resize images to 256x256
    train_images = np.array(resize_images(train_img, False))
    #test_images = np.array(resize_images(test_images, False))
    train_masks = np.array(resize_images(train_mask, True))
    #test_masks = np.array(resize_images(test_masks, True))
    #Shape of resized images
    print("Shape of resized training raw image: " + str(train_images.shape))
    print("Shape of resized training mask image: " + str(train_masks.shape))
    #print("Shape of resized testing raw image: " + str(test_images.shape))
    #print("Shape of resized testing mask image: " + str(test_masks.shape))

    #Create training dataset
    training_dataset = make_dataset(train_images, train_masks, False, 0.01)
    #Check dataset info
    print("Train Dataset Info:\n" + str(training_dataset))

    #Initialize the processor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    #Create instance of the SAMDataset class
    train_dataset = SAMDataset(dataset=training_dataset, processor=processor)

########################################################################################################################################################################
    #IF drop_last=False IT GIVES THE FOLLOWING ERROR: IndexError: index 1 is out of bounds for dimension 0 with size 1
########################################################################################################################################################################
    #Create a Dataloader instance for the training
    sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank, shuffle=False, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=batch_size, drop_last=False, shuffle=False, sampler=sampler)
    
    #Load SAM model
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    
    #Original number of parameters
    original_sam_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    #Implement LoRA to model
    ranking = 3
    model = lora_config(model, ranking)

    #Print difference in parameters before/after LoRA
    print(f"Original SAM total params: {original_sam_total_params}")
    sam_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA-SAM total params: {sam_total_params}")

    #Initialize the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    #Train the model
    train_model(model, train_dataloader, seg_loss, optimizer, num_epochs, rank, ranking)

    #Destroy process group
    destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LoRA_SAM distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.total_epochs, args.batch_size), nprocs=world_size)