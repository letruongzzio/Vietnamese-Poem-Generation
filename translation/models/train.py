import os
import sys
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


PROJECT_DIR = os.path.expanduser("~/Vietnamese-Poem-Generation/")
sys.path.append(PROJECT_DIR)
from constants import LOG_DIR, STORAGE_DIR


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
        sub_batch_size: int = 4,  # 🔹 Kích thước mini-batch để chia nhỏ
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

        for src_ids, tgt_ids in progress_bar:
            src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
            optimizer.zero_grad()

            num_splits = max(1, src_ids.size(0) // sub_batch_size)
            src_split = torch.chunk(src_ids, num_splits, dim=0)
            tgt_split = torch.chunk(tgt_ids, num_splits, dim=0)

            total_loss = 0.0
            for mini_src, mini_tgt in zip(src_split, tgt_split):
                if hasattr(model, "teacher_forcing_ratio"):
                    outputs = model(mini_src, mini_tgt, training=True)
                else:
                    outputs = model(mini_src, mini_tgt)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), mini_tgt.view(-1))
                
                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            running_loss += total_loss / num_splits
            progress_bar.set_postfix(loss=f"{total_loss / num_splits:.4f}")

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        print(f"Epoch {epoch + 1} | Train Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src_ids, tgt_ids in val_loader:
                src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)

                num_splits = max(1, src_ids.size(0) // sub_batch_size)
                src_split = torch.chunk(src_ids, num_splits, dim=0)
                tgt_split = torch.chunk(tgt_ids, num_splits, dim=0)

                total_val_loss = 0.0
                for mini_src, mini_tgt in zip(src_split, tgt_split):
                    if hasattr(model, "teacher_forcing_ratio"):
                        outputs = model(mini_src, mini_tgt, training=True)
                    else:
                        outputs = model(mini_src, mini_tgt)
                    loss = criterion(outputs.view(-1, outputs.shape[-1]), mini_tgt.view(-1))
                    total_val_loss += loss.item()

                val_loss += total_val_loss / num_splits

        val_loss /= len(val_loader)
        writer.add_scalar("Loss/val", val_loss, epoch)
        print(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss - threshold:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered due to no improvement.")
                break

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

    writer.close()
    print("Training complete!")
