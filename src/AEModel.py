from pathlib import Path
from time import time
from typing import Callable, List, Tuple

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from config import CAMVID_DIR
from metrics import iou
from utils import Color_map, mask_to_rgb, write_3d_array

code2id, id2code, name2id, id2name = Color_map(CAMVID_DIR / 'class_dict.csv')


class AEModelTrainer:

    def __init__(self, model: nn.Module) -> None:
        self.model: nn.Module = model

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, optimizer: optim.Optimizer,
              loss_fn: Callable, scaler: torch.cuda.amp.GradScaler, log_name: str) -> Tuple[List[float], List[float]]:
        """
        Train the model for a given number of epochs.

        Parameters
        ----------
        train_loader : DataLoader
            Dataloader with training data
        val_loader : DataLoader
            Dataloader with validation data
        epochs : int
            Number of epochs to train for
        optimizer : optim.Optimizer
            Optimizer to use for training
        loss_fn : Callable
            Loss function to use for training
        scaler : torch.cuda.amp.GradScaler
            Gradient scaler to use for training
        log_name : str
            Name of the log file to use

        Returns
        -------
        Tuple[List[float], List[float]]
            Training and validation losses
        """

        # Summary writer for Tensorboard
        writer = SummaryWriter(log_dir='runs/' + log_name)

        # Add graph to Tensorboard
        # writer.add_graph(self.model)

        # Get device
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
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

        # Saving iou through epochs
        train_iou = list()
        valid_iou = list()

        # Saving preds through epochs
        #all_preds = list()  # shape: (epoch, image, pixel)

        for epoch in range(1, epochs + 1):

            start = time()

            train_loss: float = 0.0
            valid_loss: float = 0.0

            #epoch_preds = list()

            # set model to be trainable
            self.model.train()

            # iterate over batches
            for batch_idx, (data, targets, rgb) in enumerate(train_loader):

                # move data to device
                data = data.to(device)
                targets = targets.float().to(device)

                # forward
                with torch.cuda.amp.autocast():
                    predictions = self.model(data)
                    loss = loss_fn(predictions, targets)

                # Add train iou
                # selecting the most probable class
                train_iou.extend(iou(predictions.argmax(1), targets.argmax(1)))

                # backward
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Update progress bar
                progress_bar.update(1)

                # Add batch loss to total loss
                train_loss += loss.item() * data.size(0)

            # Calculate average loss
            train_loss /= len(train_loader)

            # Calculate average iou
            avg_train_iou = sum(train_iou) / len(train_iou)

            # Calculate validation loss
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (data, targets, rbg) in enumerate(val_loader):
                    # move data to device
                    data = data.to(device)
                    targets = targets.float().to(device)
                    # forward
                    predictions = self.model(data)
                    # calculate loss
                    loss = loss_fn(predictions, targets)
                    # add batch loss to total loss
                    valid_loss += loss.item() * data.size(0)
                    # calculate iou score
                    valid_iou.extend(
                        iou(predictions.argmax(1), targets.argmax(1)))

                    # epoch_preds.extend(
                    #     predictions.argmax(1).flatten(
                    #         start_dim=1).cpu().tolist())

            # Calculate average loss
            valid_loss /= len(val_loader)

            # average jaccard score mIOU
            avg_valid_iou = sum(valid_iou) / len(valid_iou)

            # all_preds.append(epoch_preds)

            stop = time()

            # print training and validation loss for epoch
            print(f'\nEpoch: {epoch}\navg train-loss: {round(train_loss, 4)}\navg val-loss: {round(valid_loss, 4)}\nmIoU-train: {avg_train_iou}\nmIoU-val: {avg_valid_iou}\ntime: {stop-start}\n')

            # Save losses
            train_losses.append(train_loss), valid_losses.append(valid_loss)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('mIoU/train', avg_train_iou, epoch)
            writer.add_scalar('Loss/val', valid_loss, epoch)
            writer.add_scalar('mIoU/val', avg_valid_iou, epoch)

        writer.close()

        #all_preds = np.array(all_preds)
        #write_3d_array(all_preds, 'valid_preds/' + log_name + ".txt")

        isExist = os.path.exists('models')
        if not isExist:
            os.makedirs('models')

        torch.save(self.model.state_dict(), 'models/' + log_name + '.pt')

        return train_losses, valid_losses

    def predict(self, test_loader: DataLoader, model_path: Path,
                loss_fn: Callable) -> List[Tuple[torch.Tensor, torch.Tensor, List[float]]]:
        """
        Predict the labels for the test set and compute the jaccard score.

        Parameters
        ----------
        test_loader : DataLoader
            Dataloader with test data

        Returns
        -------
        List[Tuple[torch.Tensor, torch.Tensor, List[float]]]
            List of tuples containing predictions, targets and jaccard scores
        """

        self.model.load_state_dict(torch.load(f'{model_path}.pt'))

        # Get device
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using {device} as backend')

        # Cast model to device
        self.model.to(device)

        # Saving results
        iou_scores = list()
        preds = list()
        test_loss = 0.0

        # activate evaluation mode
        self.model.eval()

        # Iterate over batches
        with torch.no_grad():
            # Create empty list for predictions and targets
            predictions = list()
            targets = list()
            for batch_idx, (data, targets, rbg) in enumerate(test_loader):
                # move data to device
                data = data.to(device)
                targets = targets.to(device)
                # forward
                predictions = self.model(data)
                # calculate loss
                loss = loss_fn(predictions, targets)
                # add batch loss to total loss
                test_loss += loss.item() * data.size(0)
                # calculate iou score
                iou_scores.extend(
                    iou(predictions.argmax(1), targets.argmax(1)))

                preds.extend(mask_to_rgb(predictions.cpu().numpy(), id2code))

        # average jaccard score mIOU
        avg_test_iou = sum(iou_scores) / len(iou_scores)

        # Calculate average loss
        test_loss /= len(test_loader)

        return preds, avg_test_iou, test_loss
