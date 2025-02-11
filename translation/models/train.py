import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


PROJECT_DIR = os.path.expanduser("~/Vietnamese-Poem-Generation/")
sys.path.append(PROJECT_DIR)
from constants import LOG_DIR, STORAGE_DIR


# Training function
def train_model(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        num_epochs: int,
        device: torch.device,
        model_name: str,
        threshold: float = 1e-6,
        patience: int = 5,  # Early stopping if no improvement after X epochs.
    ) -> None:
    """
    Train a Seq2Seq model with logging, tqdm progress tracking, early stopping, and metric calculation.

    Args:
        model (nn.Module): Seq2Seq model to train.
        train_loader (DataLoader): Training dataset loader.
        val_loader (DataLoader): Validation dataset loader.
        criterion (nn.Module): Loss function (CrossEntropyLoss).
        optimizer (Optimizer): Optimizer (Adam, SGD, etc.).
        scheduler (lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to run training (cuda or cpu).
        model_name (str): Name of the model.
        threshold (float): Minimum improvement in validation loss to save the model.
        patience (int): Early stopping threshold.

    Returns:
        None
    """
    writer = SummaryWriter(log_dir=LOG_DIR)
    save_path = os.path.join(STORAGE_DIR, f"{model_name}.pt")

    model.to(device)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for src_ids, tgt_ids in progress_bar:
            src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)

            optimizer.zero_grad()
            outputs = model(src_ids, tgt_ids, training=True)

            loss = criterion(outputs.view(-1, outputs.shape[-1]), tgt_ids.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        print(f"Epoch {epoch + 1} | Train Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src_ids, tgt_ids in val_loader:
                src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
                outputs = model(src_ids, tgt_ids, training=False)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), tgt_ids.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f}")

        # Early stopping & Save best model
        if val_loss < best_loss - threshold:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered due to no improvement")
                break

        # Scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

    writer.close()
    print("Training complete!")
