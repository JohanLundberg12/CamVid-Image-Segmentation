from time import time
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm


class AEModel:

    def __init__(self, model: nn.Module) -> None:
        self.model: nn.Module = model

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, optimizer: optim.Optimizer, loss_fn: Callable, scaler: torch.cuda.amp.GradScaler, log_name: str) -> Tuple[List[float], List[float]]:
        
        # Summary writer for Tensorboard
        writer = SummaryWriter(log_dir='runs/' + log_name)

        # Add graph to Tensorboard
        # writer.add_graph(self.model)

        # Get device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using {device} as backend')
        # and move our model over to the selected device
        self.model.to(device)
        # activate training mode
        self.model.train()

        # Just to have a nice progress bar
        num_training_steps = epochs * len(train_loader)
        progress_bar = tqdm(range(num_training_steps))

        # Saving losses through epochs
        train_losses = list()
        valid_losses = list()

        for epoch in range(1, epochs+1):

            start = time()

            train_loss: float = 0.0
            valid_loss: float = 0.0

            # set model to be trainable
            self.model.train()

            # iterate over batches
            for batch_idx, (data, targets) in enumerate(train_loader):
                
                # move data to device
                data = data.to(device)
                targets = targets.float().to(device)
                
                # forward
                with torch.cuda.amp.autocast():
                    predictions = self.model(data)
                    loss = loss_fn(predictions, targets)
                    
                # backward
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Update progress bar
                progress_bar.update(1)

                # Add batch loss to total loss
                train_loss += loss.item()

            # Calculate average loss
            train_loss /= len(train_loader)

            # Calculate validation loss
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(val_loader):
                    data = data.to(device)
                    targets = targets.float().to(device)
                    predictions = self.model(data)
                    loss = loss_fn(predictions, targets)
                    valid_loss += loss.item()
            
            # Calculate average loss
            valid_loss /= len(val_loader)

            stop = time()

            # print training and validation loss for epoch
            print(f'Epoch: {epoch}, avg train-loss: {round(train_loss, 4)}, avg val-loss: {round(valid_loss, 4)}, time: {stop-start}')

            # Save losses
            train_losses.append(train_loss), valid_losses.append(valid_loss)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', valid_loss, epoch)

        writer.close()

        return train_losses, valid_losses
