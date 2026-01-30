"""A generic training wrapper."""
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from dfadetect import cnn_features
from dfadetect.datasets import TransformDataset


LOGGER = logging.getLogger(__name__)



# =========================
# Data settings
# =========================
@dataclass
class NNDataSetting:
    use_cnn_features: bool


# =========================
# Padding collate function (CRITICAL FIX)
# =========================
def pad_collate_fn(batch):
    batch_x, batch_meta, batch_y = zip(*batch)

    # Convert numpy arrays to tensors if needed
    batch_x_tensors = []
    for x in batch_x:
        # Convert to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Ensure x is 1D audio (flatten if needed)
        if x.dim() > 1:
            x = x.flatten()
        
        batch_x_tensors.append(x)
    
    batch_x = batch_x_tensors

    # Find max length
    max_len = max(x.shape[-1] for x in batch_x)

    padded_x = []
    for x in batch_x:
        # Ensure x is at least 1D
        if x.dim() == 0:
            x = x.unsqueeze(0)
        
        # Make it 2D: [1, length] for audio
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        pad_amount = max_len - x.shape[-1]
        padded_x.append(F.pad(x, (0, pad_amount)))

    # Stack: [batch, 1, length]
    stacked_x = torch.stack(padded_x)
    
    # Ensure it's 3D [batch, channels=1, length], not 4D or 5D
    if stacked_x.dim() > 3:
        # Squeeze extra dimensions
        stacked_x = stacked_x.squeeze()
        if stacked_x.dim() == 2:  # If squeezed too much
            stacked_x = stacked_x.unsqueeze(1)  # Add channel dim back
    
    # Convert labels - handle multiple dataset label formats
    label_list = []
    for y in batch_y:
        if isinstance(y, str):
            y_lower = y.lower()
            
            # Bonafide/Real labels -> 0
            if y_lower in ['bonafide', 'real', '-']:
                label_list.append(0)
            # Explicit spoof/fake labels -> 1
            elif y_lower in ['spoof', 'fake']:
                label_list.append(1)
            # ASVspoof attack codes (A01, A02, etc.) -> 1 (spoof)
            elif y.startswith('A') and len(y) >= 2 and y[1:].isdigit():
                label_list.append(1)
            # Any other attack identifier -> 1 (spoof)
            else:
                # Assume unknown labels are attacks/spoofs
                label_list.append(1)
        elif isinstance(y, torch.Tensor):
            label_list.append(y.item() if y.dim() == 0 else y)
        else:
            label_list.append(int(y))
    
    batch_y_tensor = torch.tensor(label_list, dtype=torch.long)

    return stacked_x, batch_meta, batch_y_tensor


# =========================
# Base Trainer
# =========================
class Trainer:
    """Lightweight wrapper for training models with gradient descent."""

    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 1e-3},
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.epoch_test_losses: List[float] = []



# =========================
# GMM Trainer
# =========================
class GMMTrainer(Trainer):

    def train(
        self,
        model: torch.nn.Module,
        dataset: TransformDataset,
        logging_prefix: str = "",
        test_len: float = 0.2,
    ) -> torch.nn.Module:

        model = model.to(self.device)
        model.train()

        test_len = int(len(dataset) * test_len)
        train_len = len(dataset) - test_len
        train, test = torch.utils.data.random_split(dataset, [train_len, test_len])

        # GMM usually processes files one by one (batch_size=1)
        train_loader = DataLoader(train, batch_size=1, shuffle=True)
        test_loader = DataLoader(test, batch_size=1)

        optimizer = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)
        LOGGER.info(f"Starting training of {logging_prefix} for {self.epochs} epochs!")

        for epoch in range(1, self.epochs + 1):
            train_iter = iter(train_loader)

            while True:
                batch = []
                for _ in range(self.batch_size):
                    try:
                        audio_file, _ = next(train_iter)
                        audio_file = audio_file.view(audio_file.shape[-2:]).T
                        batch.append(audio_file)
                    except StopIteration:
                        break

                if not batch:
                    break

                batch = torch.cat(batch).to(self.device)
                pred = model(batch)
                loss = -pred.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model._build_distributions()

            LOGGER.info(
                f"Epoch [{epoch}/{self.epochs}]: train/{logging_prefix}__loss: {loss.item()}"
            )

            test_losses = []
            for audio_file, _ in test_loader:
                audio_file = audio_file.view(audio_file.shape[-2:]).T.to(self.device)
                test_losses.append((-model(audio_file).mean()).item())

            test_loss = float(np.mean(test_losses))
            LOGGER.info(
                f"Epoch [{epoch}/{self.epochs}]: test/{logging_prefix}__loss: {test_loss}"
            )
            self.epoch_test_losses.append(test_loss)

        model.eval()
        return model



# =========================
# Forward + loss helper
# =========================
def forward_and_loss(model, criterion, batch_x, batch_y, **kwargs):
    batch_out = model(batch_x)
    batch_loss = criterion(batch_out, batch_y)
    return batch_out, batch_loss



# =========================
# Gradient Descent Trainer
# =========================
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
    ):
        model = model.to(self.device) # Ensure model is on the correct device

        if test_dataset is not None:
            train = dataset
            test = test_dataset
        else:
            test_len_count = int(len(dataset) * test_len)
            train_len = len(dataset) - test_len_count
            train, test = torch.utils.data.random_split(
                dataset, [train_len, test_len_count]
            )

        # 🔴 Updated DataLoader with collate_fn and corrected num_workers
        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,  # Optimized for your system
            collate_fn=pad_collate_fn,
        )

        test_loader = DataLoader(
            test,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=4,
            collate_fn=pad_collate_fn,
        )

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        best_model = None
        best_acc = 0.0

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

                batch_x = batch_x.to(self.device)

                if nn_data_setting.use_cnn_features:
                    batch_x = cnn_features.prepare_feature_vector(
                        batch_x,
                        cnn_features_setting=cnn_features_setting,
                    )

                # Ensure batch_y is the right shape [batch_size, 1]
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)
                
                batch_y = batch_y.float().to(self.device)

                batch_out, batch_loss = forward_and_loss(
                    model, criterion, batch_x, batch_y
                )

                batch_pred = (torch.sigmoid(batch_out) > 0.5).int()
                num_correct += (batch_pred == batch_y.int()).sum().item()

                running_loss += batch_loss.item() * batch_size

                optim.zero_grad()
                batch_loss.backward()
                optim.step()

            running_loss /= max(num_total, 1)
            train_acc = 100 * num_correct / max(num_total, 1)

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: "
                f"train/{logging_prefix}__loss={running_loss:.4f}, "
                f"train/{logging_prefix}__acc={train_acc:.2f}"
            )

            # -------- Evaluation --------
            model.eval()
            test_loss = 0.0
            num_correct_test = 0.0
            num_total_test = 0.0

            with torch.no_grad():
                for batch_x, _, batch_y in test_loader:
                    batch_size = batch_x.size(0)
                    num_total_test += batch_size

                    batch_x = batch_x.to(self.device)

                    if nn_data_setting.use_cnn_features:
                        batch_x = cnn_features.prepare_feature_vector(
                            batch_x,
                            cnn_features_setting=cnn_features_setting,
                        )

                    if batch_y.dim() == 1:
                        batch_y = batch_y.unsqueeze(1)
                    
                    batch_y = batch_y.float().to(self.device)

                    batch_out = model(batch_x)
                    loss = criterion(batch_out, batch_y)

                    test_loss += loss.item() * batch_size
                    batch_pred = (torch.sigmoid(batch_out) > 0.5).int()
                    num_correct_test += (batch_pred == batch_y.int()).sum().item()

            test_loss /= max(num_total_test, 1)
            test_acc = 100 * num_correct_test / max(num_total_test, 1)

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: "
                f"test/{logging_prefix}__loss={test_loss:.4f}, "
                f"test/{logging_prefix}__acc={test_acc:.2f}"
            )

            if best_model is None or test_acc > best_acc:
                best_acc = test_acc
                best_model = deepcopy(model.state_dict())

        model.load_state_dict(best_model)
        return model
