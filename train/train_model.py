import os
import time
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


# TODO: Add this information to README.md
# ! TO RUN IN TERMINAL (WINDOWS): set PYTHONPATH=%PYTHONPATH%;<PATH_TO_PROJECT>
# THEN RUN: python models/train_model.py --data_path <PATH_TO_DATASET> -- device <DEVICE> --epochs <EPOCHS> --model_path <MODEL_PATH>
# ! TO RUN IN TERMINAL (LINUX/MACOS): export PYTHONPATH=$PYTHONPATH:<PATH_TO_PROJECT>
# THEN RUN: python models/train_model.py --data_path <PATH_TO_DATASET> -- device <DEVICE> --epochs <EPOCHS> --model_path <MODEL_PATH>

def arguments():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Weather classification')
    parser.add_argument('--data_path', type=str,
                        default=r'C:\Users\Anwender\Desktop\Nicolas\Dokumente\FH Bielefeld\Optimierung und Simulation\2. Semester\SimulationOptischerSysteme\AI-Weather-Classification\dataset',
                        help='Path to the dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--epochs', type=int, default=28, help='Number of epochs to train the model')
    parser.add_argument('--model_path', type=str, default='trained_model.pth', help='Path to save the trained model')

    args = parser.parse_args()
    data_path = args.data_path
    device = args.device
    epochs = args.epochs
    model_path = args.model_path

    return data_path, device, epochs, model_path


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = structlog.get_logger()


class WeatherClassifier(nn.Module):
    """
    A convolutional neural network classifier designed for classifying weather conditions from images.

    Attributes:
        num_classes (int): Number of distinct weather conditions to classify.
        device (torch.device): Device on which the model computations will be performed.

    Methods:
        forward(x): Defines the forward pass of the model.
        train_model(...): Trains the model with given parameters.
        test(...): Evaluates the model's performance on the test dataset.
        predict_image(...): Makes predictions on a single image.
        _print_model_size(): Prints the size of the model in terms of parameters.
    """

    def __init__(self, num_classes: int, device: torch.device, image_size: tuple = (512, 512)):
        super(WeatherClassifier, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.image_size = image_size

        print("=" * 200, "\n")

        # Initialize network architecture
        self._initialize_layers()

        # Setup the classifier part of the network
        self._setup_classifier()

        logger.info("Model initialized successfully.")

        # Move the model to the specified device
        self.to(device)
        logger.info("Model moved to device.", device=device)

    def _initialize_layers(self):
        """ Initialize the convolutional layers of the network """
        num_layers = 3
        channels = [3, 64, 512, 1024, 4096]
        kernel_size_conv = 3
        stride_conv = 2
        padding_conv = 1
        kernel_size_pool = 3
        stride_pool = 2
        activation_function = nn.LeakyReLU()

        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size_conv, stride=stride_conv,
                          padding=padding_conv),
                activation_function,
                nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_pool)
            ])

        self.features = nn.Sequential(*layers)
        self.features.to(self.device)
        self._calculate_feature_size()

        logger.info("Convolutional Layers Setup")
        table = tabulate({
            'Number of Layers': [num_layers],
            'Channels': [channels],
            'Kernel Size Convolutional': [kernel_size_conv],
            'Stride Convolutional': [stride_conv],
            'Padding Convolutional': [padding_conv],
            'Kernel Size Pooling': [kernel_size_pool],
            'Stride Pooling': [stride_pool],
            'Activation Function': [activation_function]
        }, headers='keys', tablefmt='pretty')
        print(table)
        print()

    def _setup_classifier(self):
        """ Setup the classifier part of the network """
        dropout_rate = 0.357273264918355
        num_feature = 1024
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, num_feature),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_feature, self.num_classes)
        )

        logger.info("Classifier Setup")
        table = tabulate({
            'Dropout Rate': [dropout_rate],
            'Number of features': [num_feature]
        }, headers='keys', tablefmt='pretty')
        print(table)
        print()

    def _calculate_feature_size(self):
        """ Calculate the output size of the feature maps to connect to the classifier """
        dummy_input = torch.ones(1, 3, int(self.image_size[0]), int(self.image_size[1])).to(self.device)
        dummy_output = self.features(dummy_input)
        self.num_features = dummy_output.view(dummy_output.size(0), -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the model on input data x.

        Args:
            x (torch.Tensor): Input data tensor containing image pixel values.

        Returns:
            torch.Tensor: The model's output predictions.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, learning_rate: float,
                    model_path: str, device: torch.device,
                    weight_decay: float = 1e-5):
        """
        Trains the model using the provided data loaders and training parameters.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            epochs (int): Number of epochs to train the model.
            learning_rate (float): Learning rate for the optimizer.
            model_path (str): Path where the trained model will be saved.
            device (torch.device): Device on which the model should be trained.
            weight_decay (float): Weight decay to prevent overfitting during optimization.

        Returns:
            None: This method saves the trained model to the specified path and plots the training results.
        """
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        criterion_name = self.criterion.__class__.__name__
        optimizer_name = self.optimizer.__class__.__name__

        table = tabulate({
            'Criterion': [criterion_name],
            'Optimizer': [optimizer_name],
            'Learning Rate': [learning_rate],
            'Weight Decay': [weight_decay]
        }, headers='keys', tablefmt='pretty')
        logger.info("Training Parameters")
        print(table)
        print()

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

        logger.info("Training Started")

        start = time.time()
        for epoch in range(epochs):
            self.run_epoch(train_loader, val_loader, device)

            # Print epoch results in tabular format
            table = tabulate({
                'Epoch': [epoch],
                'Train Loss': [self.train_loss_history[-1]],
                'Val Loss': [self.val_loss_history[-1]],
                'Train Acc': [self.train_acc_history[-1]],
                'Val Acc': [self.val_acc_history[-1]]
            }, headers='keys', tablefmt='pretty')
            logger.info("Epoch Results")
            print(table)

        logger.info(f"Training finished in {round(time.time() - start, 3)} seconds.")

        torch.save(self.state_dict(), model_path)
        logger.info("Model saved successfully.", model_path=model_path)
        self.plot_results()

    def run_epoch(self, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
        """
        Executes a training and validation run for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (torch.device): Device to run the training on (e.g., 'cpu' or 'cuda').

        Returns:
            None
        """
        train_loss, train_accuracy = self.train_one_epoch(train_loader, device)
        val_loss, val_accuracy = self.validate_one_epoch(val_loader, device)

        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.train_acc_history.append(train_accuracy)
        self.val_acc_history.append(val_accuracy)

    def train_one_epoch(self, train_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
        """
        Conducts training on the training dataset for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            device (torch.device): Device on which to perform the training operations.

        Returns:
            tuple: Contains average training loss and training accuracy for the epoch.
        """
        self.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return train_loss / len(train_loader.dataset), 100 * correct / total

    def validate_one_epoch(self, val_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
        """
        Validates the model on the validation dataset for one epoch.

        Args:
            val_loader (DataLoader): DataLoader for the validation data.
            device (torch.device): Device on which to perform the validation operations.

        Returns:
            tuple: Contains average validation loss and validation accuracy for the epoch.
        """
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return val_loss / len(val_loader.dataset), 100 * correct / total

    def plot_results(self):
        """
        Plots training and validation loss and accuracy over all epochs.

        Returns:
            None: Generates plots which are then displayed.
        """
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))
        ax1.plot(self.train_loss_history, label='Training Loss')
        ax1.plot(self.val_loss_history, label='Validation Loss')
        ax1.set_title('Loss over Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(self.train_acc_history, label='Training Accuracy')
        ax2.plot(self.val_acc_history, label='Validation Accuracy')
        ax2.set_title('Accuracy over Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

    def optimize_hyperparameters_lr_wd(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int,
                                       device: torch.device, n_trials: int = 50,
                                       study_name: str = "weather-classification"):
        """
        Optimize hyperparameters for learning rate and weight decay using Optuna.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            epochs (int): Number of epochs to run for each trial.
            device (torch.device): Device to perform the optimization on.
            n_trials (int): Number of trials to run.

        Returns:
            None: Outputs the best hyperparameters to the console.
        """

        def objective(trial):
            """ Objective function for hyperparameter optimization. """
            # Define hyperparameters
            lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-2)

            # Initialize model, criterion, optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

            # Training and validation logic
            for epoch in range(epochs):
                self.train()  # set the model to training mode
                train_loss = 0.0
                for inputs, labels in tqdm(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)  # move data to device

                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)

                self.eval()  # set the model to evaluation mode
                with torch.no_grad():
                    val_loss = 0.0
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)  # move data to device
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)

            val_loss = val_loss / len(val_loader.dataset)
            return val_loss

        # Create a study object and optimize the objective function
        study = optuna.create_study(
            storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
            study_name=study_name,
            direction='minimize'
        )
        study.optimize(objective, n_trials=n_trials)

        # Get the best parameters
        best_params = study.best_params

        print(f'Best parameters: {best_params}')

    def test(self, test_loader: DataLoader, model_path: str, device: torch.device):
        """
        Tests the model's performance on a provided test dataset.

        Args:
            test_loader (DataLoader): DataLoader for the test data.
            model_path (str): Path to the trained model file.
            device (torch.device): Device to perform the test on.

        Returns:
            None: This method prints the accuracy of the model on the test dataset.
        """
        self.load_state_dict(torch.load(model_path))
        self.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        pred_acc = round((100 * correct / total), 3)
        table = tabulate({
            'Number of Test Images': [len(test_loader.dataset)],
            'Accuracy': [f'{pred_acc}%']
        }, headers='keys', tablefmt='pretty')
        logger.info("Test Results")
        print(table)
        print()

    def predict_image(self, one_image_loader, model_path: str, class_to_idx_mapping: dict):
        """
        Predicts the class of a single image using the trained model.

        Args:
            one_image_loader (DataLoader): DataLoader containing one image.
            model_path (str): Path to the trained model file.
            class_to_idx_mapping (dict): Mapping from class indices to class labels.

        Returns:
            list: Predictions for the image.
        """
        self.load_state_dict(torch.load(model_path))
        self.eval()
        predictions = []
        idx_to_class_mapping = {value: key for key, value in class_to_idx_mapping.items()}
        image_names = []
        for l in one_image_loader.dataset.imgs:
            image_names.append(l[0].split("\\")[-1])

        with torch.no_grad():
            for inputs, labels in one_image_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.append(predicted)

        predictions = [idx_to_class_mapping[i.item()] for i in predictions]

        for real, pred in zip(image_names, predictions):
            print(f"Image: {real}, Prediction: {pred}")

        return predictions

    def optimize_hyperparameters_large(self, data_path, device: torch.device, n_trials=100):
        def objective(trial):
            # Define hyperparameters
            lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-2)
            batch_size = trial.suggest_int('batch_size', 2, 32)
            activation_name = trial.suggest_categorical('activation', ['ReLU', 'SiLU', 'LeakyReLU'])
            dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
            epochs = trial.suggest_int('epochs', 5, 50)

            # Set activation function
            activation = None
            if activation_name == 'ReLU':
                activation = nn.ReLU()
            elif activation_name == 'SiLU':
                activation = nn.SiLU()
            elif activation_name == 'LeakyReLU':
                activation = nn.LeakyReLU()

            kernel_size_pool = 3
            stride_pool = 2
            input_size = 512
            input_size_copy = input_size
            filter_size = kernel_size_conv = 3
            stride = stride_conv = 2  # change this parameter
            padding = padding_conv = 1
            num_layers = 2

            # Set layers
            channels = [3, 64, 512, 1024, 4096]

            layers = []
            for i in range(num_layers):
                layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size_conv, stride=stride_conv,
                                        padding=padding_conv))
                layers.append(activation)
                layers.append(nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_pool))

            self.features = nn.Sequential(*layers)
            self.to(device)

            for i in range(num_layers):
                # Convulution layer
                output_size_conv = (input_size - filter_size + 2 * padding) / stride + 1
                input_size = output_size_conv
                output_size_pool = (input_size - filter_size) / stride + 1
                input_size = output_size_pool

            # Pass a dummy input through the features module
            dummy_input = torch.ones(1, 3, int(input_size_copy), int(input_size_copy)).to(device)
            dummy_output = self.features(dummy_input)

            num_features = dummy_output.view(dummy_output.size(0), -1).size(1)

            # classifier layer?
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, 1024),
                activation,
                nn.Dropout(p=dropout),
                nn.Linear(1024, 11),
            )
            self.to(device)

            # Training and validation logic
            criterion = nn.CrossEntropyLoss()
            W = WeatherDataset(data_folder=data_path)

            train_loader, val_loader, test_loader = W.get_data_loaders(batch_size=batch_size)
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

            # start training loop
            val_accuracys = {}
            for epoch in range(epochs):
                self.train()  # set the model to training mode
                train_loss = 0.0
                for inputs, labels in tqdm(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)  # move data to device

                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)

                self.eval()  # set the model to evaluation mode
                correct_val = 0
                total_val = 0
                correct_test = 0
                total_test = 0
                with torch.no_grad():
                    val_loss = 0.0
                    test_loss = 0.0
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)  # move data to device
                        outputs = self(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)  # move data to device
                        outputs = self(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total_test += labels.size(0)
                        correct_test += (predicted == labels).sum().item()
                        loss = criterion(outputs, labels)
                        test_loss += loss.item() * inputs.size(0)

                val_accuracy = 100 * correct_val / total_val  # calculate validation accuracy
                val_accuracys[epoch] = val_accuracy  # store validation accuracy
                test_accuracy = 100 * correct_test / total_test  # calculate validation accuracy

            print("Val accuracy: " + str(val_accuracy))
            print("Val accuracys: " + str(val_accuracys))
            print("Test accuracy: " + str(test_accuracy))
            return test_accuracy

        # Create a study object and optimize the objective function
        study = optuna.create_study(
            storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
            study_name="weather-classification_" + str(datetime.now()),
            direction='maximize'
        )
        study.optimize(objective, n_trials=n_trials)

        # Get the best parameters
        best_params = study.best_params

        print(f'Best parameters: {best_params}')


if __name__ == "__main__":
    # path_dataset = "/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/dataset"
    one_image_data = "/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/utils/one_image"
    # model_path = "trained_model.pth"

    path_dataset_win = r"C:\Users\Anwender\Desktop\Nicolas\Dokumente\FH Bielefeld\Optimierung und Simulation\2. Semester\SimulationOptischerSysteme\AI-Weather-Classification\dataset"
    # one_image_data_win = r"C:\Users\Anwender\Desktop\Nicolas\Dokumente\FH Bielefeld\Optimierung und Simulation\2. Semester\SimulationOptischerSysteme\AI-Weather-Classification\test_image"

    path_dataset, device, epochs, model_path = arguments()


    logger.info("Parameters:")
    print(f"Path to dataset: {path_dataset}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Model path: {model_path}")
    print()

    device = torch.device(device)
    W = WeatherDataset(data_folder=path_dataset, resize_format=(512, 512))
    image_size = W.resize_format
    model = WeatherClassifier(num_classes=len(W.dataset.classes), device=device, image_size=image_size)

    # Get data loaders
    tr, val, test = W.get_data_loaders(batch_size=5)

    # Train model
    logger.info("START TRAINING WEATHER CLASSIFIER MODEL")
    model.train_model(tr, val, epochs, 2.1788228027184658e-05, model_path, device, 1.592566270355879e-06)

    # Test model
    model.test(test, model_path, device)

    logger.info("Model trained and tested successfully.")
    print("-" * 100)
    print("=========FINISHED=========")
