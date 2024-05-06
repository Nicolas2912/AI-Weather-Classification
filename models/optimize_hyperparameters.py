import os
from datetime import datetime

import matplotlib.pyplot as plt
import optuna
import structlog
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
from typing import Tuple

from utils.data_loader import WeatherDataset
from models.train_model import WeatherClassifier

# ! FOR OPTUNA:
# 1. pip install optuna-dashboard
# 2. cd .\models\
# 3. optuna-dashboard --storage sqlite:///example.db

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Set up logging
log = structlog.get_logger()

parser = argparse.ArgumentParser(description='Optimize hyperparameters for the weather classification model.')
parser.add_argument('--data_folder', type=str,
                    default=r'C:\Users\Anwender\Desktop\Nicolas\Dokumente\FH Bielefeld\Optimierung und Simulation\2. Semester\SimulationOptischerSysteme\AI-Weather-Classification\dataset',
                    help='Path tothedatasetfolder.')

parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for training.')
parser.add_argument('--mode', type=str, default='small', choices=["small", "large"],
                    help='Mode for optimization: small or large.')
parser.add_argument('--study_name', type=str, default=f'weather_classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                    help='Name of the Optuna study.')

args = parser.parse_args()

data_folder = args.data_folder
n_trials = args.n_trials
epochs = args.epochs
device = args.device
mode = args.mode
study_name = args.study_name

if __name__ == "__main__":
    W = WeatherDataset(data_folder=args.data_folder)
    device = torch.device(args.device)

    tr, val, te = W.get_data_loaders()

    model = WeatherClassifier(device=device, num_classes=len(W.dataset.classes))

    if mode == "small":
        model.optimize_hyperparameters_lr_wd(tr, val, epochs, device, n_trials, study_name)

    elif mode == "large":
        model.optimize_hyperparameters_large(data_folder, device, n_trials)
