from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import SamProcessor
from transformers import SamModel
from SAM_Dataset import *
from LoRA_Config import *
from torch.optim import Adam
from statistics import mean
from tqdm import tqdm
import torch.multiprocessing as mp
import argparse
import torch
import monai
import glob
import os

#Setup process group for DDP
def ddp_setup(rank, world_size):
    """
    This function sets up the process group for distributed data parallel training.
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train_model(model, train_dataloader, loss_func, optimizer, num_epochs, rank):
    """
    This function trains the LoRA-SAM model using distributed data parallel training
    Args:
        model: The LoRA-SAM model to be trained
        train_dataloader: Dataloader instance for the training dataset
        loss_func: Loss function to be used for training
        optimizer: Optimizer to be used for training
        num_epochs: Number of epochs to train for
        rank: Unique identifier of each process
    """
    mean_epoch_loss = []

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
        if rank == 0:
            print(f'EPOCH: {epoch}')
            print(f'Mean loss: {mean(epoch_losses)}')
        mean_epoch_loss.append(mean(epoch_losses))
    
    if rank == 0:
        #Save the model
        ckp = model.module.state_dict()
        PATH = f"checkpoint.pt_epochs_{num_epochs}_meanloss_{mean(epoch_losses)}"
        torch.save(ckp, PATH)
        print(f"Training checkpoint saved at {PATH}")


def main(rank: int, world_size: int, num_epochs: int, batch_size: int, num_workers: int, lora_rank: int, data_root: str):
    """
    Main function for the distributed training job. This function is called by each process
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
        num_epochs: Number of epochs to train for
        batch_size: Batch size to be used for training
        num_workers: Number of workers to be used for the dataloader
        data_root: Path to the dataset folder
    """
    #Create process group
    ddp_setup(rank, world_size)

    #Read in Images
    train_img_path = os.path.join(data_root, "Train Set/Images/*.png")
    train_mask_path = os.path.join(data_root, "Train Set/Ground-truths/*.png")

    # Get list of paths
    train_img_list = sorted(glob.glob(train_img_path))
    train_mask_list = sorted(glob.glob(train_mask_path))
    
    # Sanity check to ensure we have data
    if len(train_img_list) == 0:
        raise ValueError(f"No images found at {train_img_path}. Check your --data_root path!")

    #Initialize the processor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    #Create training dataset
    training_dataset = SAMDataset(img_paths=train_img_list, mask_paths=train_mask_list, processor=processor)

    #Create a Dataloader instance for the training
    sampler = DistributedSampler(training_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)

    #Load SAM model
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    
    #Original number of parameters
    original_sam_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    #Implement LoRA to model
    model = lora_config(model, lora_rank)

    #Print difference in parameters before/after LoRA
    if rank == 0:
        print(f"Original SAM total params: {original_sam_total_params}")
        sam_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LoRA-SAM total params: {sam_total_params}")

    #Initialize the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    #Train the model
    train_model(model, train_dataloader, seg_loss, optimizer, num_epochs, rank)

    #Destroy process group
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LoRA_SAM distributed training job')
    parser.add_argument('total_epochs', type=int, default=10)
    parser.add_argument('batch_size', type=int, default=4)
    parser.add_argument('num_workers', type=int, default=4)
    parser.add_argument('lora_rank', type=int, default=3)
    parser.add_argument('--data_root', type=str, default="./data", help="Path to the dataset folder")
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.total_epochs, args.batch_size, args.num_workers, args.lora_rank, args.data_root), nprocs=world_size)