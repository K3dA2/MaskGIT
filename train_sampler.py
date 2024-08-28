import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import datetime
import os
import torch.nn.utils as utils
from models.transformer import Transformer, Config
from utils.vqsampler_dataloader import get_data_loader
import torch.nn.functional as F
from utils.utils import count_parameters
from models.model import VQVAE


def training_loop(n_epochs, optimizer, model, device, data_loader, valid_loader, max_grad_norm=1.0, epoch_start=0, mask_token=1024):
    model.train()
    best_loss_valid = float('inf')
    for epoch in range(epoch_start, n_epochs):
        loss_train = 0.0
        loss_valid = 0.0

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit='batch')
        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device).long()  # Ensure x is of type torch.long
            #y = y.to(device)

            mask = torch.bernoulli(0.5 * torch.ones(x.shape, device=x.device))
            mask = mask.round().to(dtype=torch.int64).to(x.device)

            mask_tokens = torch.ones(x.shape[0], 1, device=x.device).long() * mask_token
            mask_indices = mask * x + (1 - mask) * mask_tokens
            mask_indices = mask_indices.long()  # Ensure mask_indices is of type torch.long

            _, loss = model(mask_indices, x)

            optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            loss_train += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        model.eval()
        with torch.no_grad():
            for valid_tensors, valid_labels in valid_loader:
                valid_tensors = valid_tensors.to(device).long()  # Ensure valid_tensors is of type torch.long
                mask = torch.bernoulli(0.5 * torch.ones(valid_tensors.shape, device=valid_tensors.device))
                mask = mask.round().to(dtype=torch.int64).to(valid_tensors.device)

                mask_tokens = torch.ones(valid_tensors.shape[0], 1, device=valid_tensors.device).long() * mask_token
                mask_indices = mask * valid_tensors + (1 - mask) * mask_tokens
                mask_indices = mask_indices.long()  # Ensure mask_indices is of type torch.long

                _, valid_loss = model(mask_indices, valid_tensors)
                loss_valid += valid_loss.item()
        
        loss_valid /= len(valid_loader)

        if loss_valid < best_loss_valid:
            best_loss_valid = loss_valid
            model_filename = 'vqvae-transformer.pth'
            model_path = os.path.join('weights', model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

        with open("vqvae-sampler.txt", "a") as file:
            file.write(f"{loss_train / len(data_loader)}\n")

        with open("vqvae-val-sampler.txt", "a") as file:
            file.write(f"{loss_valid}\n")

        print('{} Epoch {}, Training loss {}, Validation loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(data_loader), loss_valid))

        model.train()
    

if __name__ == "__main__":
    train_path = 'train.txt'
    val_path = 'val.txt'
    model_path = 'weights/vqvae-transformer.pth'
    epoch = 0

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    config = Config()
    model = Transformer(config)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    print(f"Number of parameters: {count_parameters(model)}")
    data_loader = get_data_loader(train_path, file_type='txt', batch_size=8, prepend_value=512)
    val_loader = get_data_loader(val_path, file_type='txt', batch_size=8, prepend_value=512)

    
    # Optionally load model weights if needed
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    
    training_loop(
        n_epochs=200,
        optimizer=optimizer,
        model=model,
        device=device,
        data_loader=data_loader,
        valid_loader=val_loader,
        epoch_start=epoch + 1,
        mask_token=1024
    )
