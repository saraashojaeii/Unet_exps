from sklearn.model_selection import train_test_split
import numpy as np
import os
from torch.utils.data import DataLoader
import time
from tqdm.notebook import tqdm
import wandb
import random
import torch
from data import *
from network import *
from losses import *
from utils import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

data_path = '/root/home/MD/'
arg_dataset = 'mass'

train_images, train_masks = data_pred(data_path, 'train', arg_dataset)
val_images, val_masks = data_pred(data_path, 'val', arg_dataset)

train_dataset = DataPrep(train_images, train_masks, transform=transform)
val_dataset = DataPrep(val_images, val_masks, transform=transform)

BATCH_SIZE = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

arg_logging = True

if arg_logging:
    wandb.init(project='Unet_experiments_massachusetts', entity="saraa_team", name="BCE")


unet_model = UNet()
optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001)
criterion = BCE()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet_model.to(device)

EPOCHS = 200
t1 = time.time()
for epoch in range(EPOCHS):
    unet_model.train()  # Make sure to use 'unet_model' which is now correctly moved to the device

    total_train_loss = 0
    train_count = 0
        
    total_val_loss = 0
    val_count = 0
    val_average = 0
        
    val_comm = 0
    val_corr = 0
    val_qual = 0
    total_val_comm = 0
    total_val_corr = 0
    total_val_qual = 0   

    for batch in tqdm(train_loader):
    # for batch in train_loader:
        inputs, target1 = batch
        inputs, target1 = inputs.to(device), target1.to(device)

        optimizer.zero_grad()

        mask_output = unet_model(inputs)  
        loss = criterion(mask_output, target1)

        total_train_loss += (loss.item()) 
        train_count += 1

        loss.backward()
        optimizer.step()

    train_average = total_train_loss / train_count   
    print(f"Epoch {epoch+1}/{EPOCHS}, Total Loss: {loss/len(train_loader)}, Mask Loss: {loss}")

    unet_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs, target1 = batch
            inputs, target1 = inputs.to(device), target1.to(device)

            mask_output = unet_model(inputs)
            val_loss = criterion(mask_output, target1)

            val_count += 1
            total_val_loss += val_loss.item()

            mask = mask_output.squeeze().cpu().numpy()
            val_y = target1.squeeze().cpu().numpy()  
        
            comm, corr, qual = relaxed_f1(mask, val_y, 3)
        
            val_comm += comm
            val_corr += corr
            val_qual += qual
        
            total_val_comm += val_comm
            total_val_corr += val_corr
            total_val_qual += val_qual
        
            val_comm = 0
            val_corr = 0
            val_qual = 0


    avg_val_loss = total_val_loss / val_count
    print(f"Validation: Epoch {epoch+1}/{EPOCHS}, Total Loss: {val_loss}, Mask Loss: {val_loss} ")

    val_average = total_val_loss / val_count

    val_comm_avg = total_val_comm / val_count
    val_corr_avg = total_val_corr / val_count
    val_qual_avg = total_val_qual / val_count
    val_f1 = 2 * (val_comm_avg * val_corr_avg)/(val_comm_avg + val_corr_avg)
  
    if arg_logging:
      wandb.log({"Epoch": (epoch+1), "Training Loss": train_average, "Validation Loss": val_average, "val_comm_avg": val_comm_avg, "val_corr_avg": val_corr_avg, "val_qual_avg": val_qual_avg, "val_f1": val_f1})
      os.makedirs('../saved_models', exist_ok=True)
      torch.save(unet_model.state_dict(), f'../saved_models/unet_BCE_epoch{epoch+1}.pth')
      artifact = wandb.Artifact(f'unet_BCE_epoch{epoch+1}', type='model')
      artifact.add_file(f'../saved_models/unet_BCE_epoch{epoch+1}.pth')
      wandb.log_artifact(artifact)

t2 = time.time()
print((t2 - t1))
