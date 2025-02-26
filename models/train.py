import os
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.expanduser('~/vietnamese-poem-generation')
sys.path.append(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from constants import *
sys.path.append(UTILS_DIR)
from tokenization import *
from dataset import *


def train_model(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        num_epochs: int,
        device: torch.device,
        model_name: str,
        sub_batch_size: int = 4,
        threshold: float = 1e-6,
        patience: int = 5,  
    ) -> None:
    """
    Train a Seq2Seq model with mini-batch training to reduce memory usage.
    """
    
    writer = SummaryWriter(log_dir=LOG_DIR)
    save_path = os.path.join(STORAGE_DIR, f"{model_name}.pt")

    model.to(device)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for input_seqs, target_seqs, padding_masks in progress_bar:
            input_seqs, target_seqs, padding_masks = input_seqs.to(device), target_seqs.to(device), padding_masks.to(device)
            optimizer.zero_grad()

            num_splits = max(1, input_seqs.size(0) // sub_batch_size)
            input_split = torch.chunk(input_seqs, num_splits, dim=0)
            tgt_split = torch.chunk(target_seqs, num_splits, dim=0)
            padding_mask_split = torch.chunk(padding_masks, num_splits, dim=0)

            total_loss = 0.0
            for mini_input, mini_tgt_in, mini_padding_mask in zip(input_split, tgt_split, padding_mask_split):
                seq_len = mini_input.size(1)
                src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                output = model(src=mini_input, src_mask=src_mask, src_pad_mask=mini_padding_mask)
                output = output.permute(0, 2, 1) # [batch_size, seq_len, vocab_size]
                loss = criterion(output, mini_tgt_in)
                
                loss.backward()
                total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Clip gradient to prevent exploding
            optimizer.step()
            running_loss += total_loss / num_splits
            progress_bar.set_postfix(loss=f"{total_loss / num_splits:.4f}")

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        print(f"Epoch {epoch + 1} | Train Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss - threshold:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered due to no improvement.")
                break

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

    writer.close()
    print("Training complete!")

