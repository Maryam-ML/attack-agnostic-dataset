import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
import yaml
import numpy as np

# Ensure your custom path is prioritized
sys.path.insert(0, '/kaggle/working/attack-agnostic-dataset')

from dfadetect.agnostic_datasets.attack_agnostic_dataset import AttackAgnosticDataset
from dfadetect.cnn_features import CNNFeaturesSetting
from dfadetect.datasets import apply_feature_and_double_delta, lfcc, mfcc
from dfadetect.models import models
from dfadetect.models.gaussian_mixture_model import GMMDescent, flatten_dataset
from dfadetect.trainer import GDTrainer, GMMTrainer, NNDataSetting, pad_collate_fn
from dfadetect.utils import set_seed
from experiment_config import feature_kwargs


LOGGER = logging.getLogger()

def init_logger(log_file):
    LOGGER.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)

def save_model(model: torch.nn.Module, model_dir: Union[Path, str], name: str) -> None:
    full_model_dir = Path(f"{model_dir}/{name}")
    full_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{full_model_dir}/ckpt.pth")

# ==========================================
# NN TRAINING FUNCTION
# ==========================================
def train_nn(
    datasets_paths: List[Union[Path, str]],
    batch_size: int,
    epochs: int,
    device: str,
    model_config: Dict,
    cnn_features_setting: CNNFeaturesSetting,
    model_dir: Optional[Path] = None,
    amount_to_use: Optional[int] = None,
) -> None:

    LOGGER.info("Loading data...")
    model_name, model_parameters = model_config["name"], model_config["parameters"]
    optimizer_config = model_config["optimizer"]

    use_cnn_features = False if model_name == "rawnet" else True
    nn_data_setting = NNDataSetting(use_cnn_features=use_cnn_features)
    
    timestamp = time.time()
    folds_number = 3

    for fold in range(folds_number):
        # Setup Datasets
        data_train = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            fold_num=fold,
            fold_subset="train",
            reduced_number=amount_to_use,
            oversample=True,
        )

        data_test = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            fold_num=fold,
            fold_subset="test",
            reduced_number=amount_to_use,
            oversample=True,
        )

        current_model = models.get_model(
            model_name=model_name, config=model_parameters, device=device,
        ).to(device)

        LOGGER.info(f"Training '{model_name}' fold {fold} on {len(data_train)} files.")

        # 🔴 REFRESHED TRAINER CALL
        # Note: pad_collate_fn is handled inside trainer.py's GDTrainer.train()
        trainer_instance = GDTrainer(
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            optimizer_kwargs=optimizer_config,
        )

        current_model = trainer_instance.train(
            dataset=data_train,
            model=current_model,
            test_dataset=data_test,
            nn_data_setting=nn_data_setting,
            logging_prefix=f"fold_{fold}",
            cnn_features_setting=cnn_features_setting,
        )

        if model_dir is not None:
            save_name = f"aad__{model_name}_fold_{fold}__{timestamp}"
            save_model(current_model, model_dir, save_name)

# ==========================================
# GMM TRAINING FUNCTION
# ==========================================
def train_gmm(
    datasets_paths: List[Union[Path, str]],
    feature_fn: Callable,
    feature_kwargs: dict,
    clusters: int,
    batch_size: int,
    device: str,
    model_dir: Optional[Path] = None,
    use_double_delta: bool = True,
    amount_to_use: Optional[int] = None,
    real_epochs: int = 3,
    fake_epochs: int = 1
) -> None:

    for fold in range(3):
        # Bonafide Training
        real_dataset_train = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0], 
            fold_num=fold, fold_subset="train",
            oversample=False, undersample=False,
            return_label=False, reduced_number=amount_to_use
        )
        real_dataset_train.get_bonafide_only()

        # Spoof Training
        fake_dataset_train = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            fold_num=fold, fold_subset="train",
            oversample=False, undersample=False,
            return_label=False, reduced_number=amount_to_use
        )
        fake_dataset_train.get_spoof_only()

        # Extract Features
        real_dataset_train, fake_dataset_train = apply_feature_and_double_delta(
            [real_dataset_train, fake_dataset_train],
            feature_fn=feature_fn,
            feature_kwargs=feature_kwargs,
            use_double_delta=use_double_delta
        )

        # Train Real GMM
        LOGGER.info(f"GMM Fold {fold}: Training real model...")
        initial_real = flatten_dataset(real_dataset_train, device, 10)
        real_model = GMMDescent(clusters, initial_real, covariance_type="diag").to(device)
        real_model = GMMTrainer(device=device, epochs=real_epochs, batch_size=batch_size).train(
            real_model, real_dataset_train, test_len=0.05
        )

        # Train Fake GMM
        LOGGER.info(f"GMM Fold {fold}: Training fake model...")
        initial_fake = flatten_dataset(fake_dataset_train, device, 10)
        fake_model = GMMDescent(clusters, initial_fake, covariance_type="diag").to(device)
        fake_model = GMMTrainer(device=device, epochs=fake_epochs, batch_size=batch_size).train(
            fake_model, fake_dataset_train, test_len=0.05
        )

        if model_dir is not None:
            save_model(real_model, model_dir, f"real_{fold}")
            save_model(fake_model, model_dir, f"fake_{fold}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["data"].get("seed", 42))
    device = "cuda" if not args.cpu and torch.cuda.is_available() else "cpu"

    model_dir = Path(args.ckpt)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not args.use_gmm:
        cnn_features_setting = config["data"].get("cnn_features_setting", {})
        cnn_features_setting = CNNFeaturesSetting(**cnn_features_setting)

        train_nn(
            datasets_paths=[args.asv_path],
            device=device,
            amount_to_use=args.amount,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_dir=model_dir,
            model_config=config["model"],
            cnn_features_setting=cnn_features_setting,
        )
    else:
        feature_fn = lfcc if args.lfcc else mfcc
        train_gmm(
            datasets_paths=[args.asv_path],
            feature_fn=feature_fn,
            feature_kwargs=feature_kwargs(args.lfcc),
            clusters=args.clusters,
            batch_size=args.batch_size,
            device=device,
            model_dir=model_dir,
            amount_to_use=args.amount
        )

# ... (parse_args function remains as is) ...

if __name__ == "__main__":
    main(parse_args())