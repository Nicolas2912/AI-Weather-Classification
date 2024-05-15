import os
import sys
from datetime import datetime

import structlog
import argparse

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to the sys.path
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import WeatherDataset
from src.training.train_model import WeatherClassifier
from src.training.train_model import DEFAULT_DATASET_PATH

# TO RUN IN CONSOLE:
# 1. (WINDOWS): set PYTHONPATH=<path_to_project_root>
# 2. (LINUX): export PYTHONPATH=<path_to_project_root>
# 3. python models/optimize_hyperparameters.py --data_folder <path_to_dataset> --n_trials <number_of_trials>
# --epochs <number_of_epochs> --device <device> --mode <mode> --study_name <study_name>

# ! FOR OPTUNA:
# 1. pip install optuna-dashboard
# 2. cd .\models\
# 3. optuna-dashboard --storage sqlite:///example.db

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Set up logging
logger = structlog.get_logger()


def arguments():
    parser_opt = argparse.ArgumentParser(description='Optimize hyperparameters for the weather classification model.')
    parser_opt.add_argument('--data', type=str,
                            default=DEFAULT_DATASET_PATH,
                            help='Path to the dataset folder.')

    parser_opt.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials.')
    parser_opt.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser_opt.add_argument('--device', type=str, default='cuda', help='Device to use for training.', choices=["cpu", "cuda", "mps"])
    parser_opt.add_argument('--mode', type=str, default='small', choices=["small", "large"],
                            help='Mode for optimization: small or large.')
    parser_opt.add_argument('--study_name', type=str,
                            default=f'weather_classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                            help='Name of the Optuna study.')

    args_opt = parser_opt.parse_args()

    return args_opt.data, args_opt.n_trials, args_opt.epochs, args_opt.device, args_opt.mode, args_opt.study_name


if __name__ == "__main__":
    data, n_trials, epochs, device, mode, study_name = arguments()
    logger.info("PARAMETERS:")
    print(f"Data path: {data}")
    print(f"Number of trials: {n_trials}")
    print(f"Number of epochs: {epochs}")
    print(f"Device: {device}")
    print(f"Mode: {mode}")
    print(f"Study name: {study_name}")
    print()

    W = WeatherDataset(data_folder=data)
    device = torch.device(device)

    # Get data
    tr, val, te = W.get_data_loaders()

    # Get model
    model = WeatherClassifier(compute_device=device, num_classes=len(W.dataset.classes))

    logger.info("START OPTIMIZING HYPERPARAMETERS")

    # Optimize hyperparameters
    if mode == "small":
        model.optimize_hyperparameters_lr_wd(tr, val, epochs, device, n_trials, study_name)

    elif mode == "large":
        model.optimize_hyperparameters_large(data, device, n_trials)

    print("=======FINISHED=======")
