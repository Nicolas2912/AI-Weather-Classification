# data_loader.py

# 1. Import necessary modules and packages
import argparse
import torch
from models.model_architecture import WeatherPredictionModel
from utils.data_loader import get_data_loaders
from models.train_model import train_model
from utils.evaluation_metrics import evaluate_model

# 2. Set up device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Load data using data loaders
train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)

# 4. Initialize the model
model = WeatherPredictionModel().to(device)

# 5. Train the model
train_model(model, train_loader, val_loader, num_epochs=10)

# 6. Evaluate the trained model
evaluate_model(model, test_loader)

# 7. Save the trained model if needed
torch.save(model.state_dict(), "trained_model.pth")
