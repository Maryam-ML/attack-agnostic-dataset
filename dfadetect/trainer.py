"""A generic training wrapper."""
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from dfadetect import cnn_features

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LOGGER = logging.getLogger(__name__)


@dataclass
class NNDataSetting:
    use_cnn_features: bool


class Trainer:
    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 1e-3, "weight_decay": 1e-4},
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.epoch_test_losses: List[float] = []


def forward_and_loss(model, criterion, batch_x, batch_y):
    batch_out = model(batch_x)
    batch_loss = criterion(batch_out, batch_y)
    return batch_out, batch_loss


class GDTrainer(Trainer):

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        nn_data_setting: NNDataSetting,
        cnn_features_setting: cnn_features.CNNFeaturesSetting,
        test_len: Optional[float] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        logging_prefix: str = "",
        pos_weight: Optional[torch.FloatTensor] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        if test_dataset is not None:
            train = dataset
            test = test_dataset
        else:
            test_len = int(len(dataset) * test_len)
            train_len = len(dataset) - test_len
            train, test = torch.utils.data.random_split(dataset, [train_len, test_len])

        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )

        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.epochs, eta_min=1e-6
        )

        use_amp = self.device != "cpu" and torch.cuda.is_available()
        scaler = GradScaler(enabled=use_amp)

        best_model = None
        best_val_loss = float("inf")

        LOGGER.info(f"Starting training for {self.epochs} epochs!")

        for epoch in range(self.epochs):
            LOGGER.info(f"Epoch num: {epoch}")

            running_loss = 0.0
            num_correct = 0.0
            num_total = 0.0
            model.train()

            for i, (batch_x, _, batch_y) in enumerate(train_loader):
                batch_size = batch_x.size(0)
                num_total += batch_size

                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device, non_blocking=True)

                if nn_data_setting.use_cnn_features:
                    batch_x = cnn_features.prepare_feature_vector(
                        batch_x, cnn_features_setting=cnn_features_setting
                    )

                optim.zero_grad(set_to_none=True)

                with autocast(enabled=use_amp):
                    batch_out, batch_loss = forward_and_loss(model, criterion, batch_x, batch_y)

                scaler.scale(batch_loss).backward()
                scaler.step(optim)
                scaler.update()

                batch_pred = (torch.sigmoid(batch_out) >= 0.5).int()
                num_correct += (batch_pred == batch_y.int()).sum().item()
                running_loss += batch_loss.item() * batch_size

                if i % 100 == 0:
                    LOGGER.info(
                        f"[Epoch {epoch:04d}] [Step {i:05d}] | "
                        f"Loss: {running_loss / num_total:.4f} | "
                        f"Acc: {num_correct / num_total * 100:.2f}%"
                    )

            running_loss /= max(num_total, 1)
            train_accuracy = (num_correct / max(num_total, 1)) * 100
            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: "
                f"train/{logging_prefix}__loss: {running_loss:.6f}, "
                f"train/{logging_prefix}__accuracy: {train_accuracy:.4f}"
            )

            test_running_loss = 0.0
            num_correct = 0.0
            num_total = 0.0
            model.eval()

            with torch.no_grad():
                for batch_x, _, batch_y in test_loader:
                    batch_size = batch_x.size(0)
                    num_total += batch_size

                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device, non_blocking=True)

                    if nn_data_setting.use_cnn_features:
                        batch_x = cnn_features.prepare_feature_vector(
                            batch_x, cnn_features_setting=cnn_features_setting
                        )

                    with autocast(enabled=use_amp):
                        batch_out = model(batch_x)
                        batch_loss = criterion(batch_out, batch_y)

                    test_running_loss += batch_loss.item() * batch_size
                    batch_pred = (torch.sigmoid(batch_out) >= 0.5).int()
                    num_correct += (batch_pred == batch_y.int()).sum().item()

            test_running_loss /= max(num_total, 1)
            test_acc = 100 * (num_correct / max(num_total, 1))
            self.epoch_test_losses.append(test_running_loss)

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: "
                f"test/{logging_prefix}__loss: {test_running_loss:.6f}, "
                f"test/{logging_prefix}__accuracy: {test_acc:.4f}"
            )

            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            LOGGER.info(f"Epoch [{epoch+1}/{self.epochs}] lr: {current_lr:.8f}")

            if best_model is None or test_running_loss < best_val_loss:
                best_val_loss = test_running_loss
                best_model = deepcopy(model.state_dict())

                if checkpoint_dir is not None:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_val_loss": best_val_loss,
                        },
                        os.path.join(checkpoint_dir, "best_checkpoint.pth"),
                    )
                    LOGGER.info(
                        f"New best checkpoint saved at epoch {epoch+1} "
                        f"with val_loss={best_val_loss:.6f}"
                    )

            LOGGER.info(
                f"[{epoch:04d}]: train_loss={running_loss:.6f} | "
                f"train_acc={train_accuracy:.4f} | "
                f"val_loss={test_running_loss:.6f} | "
                f"val_acc={test_acc:.4f}"
            )

            torch.cuda.empty_cache()

        model.load_state_dict(best_model)
        return model