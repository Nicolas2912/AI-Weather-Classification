import os
import sys
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to the sys.path
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import WeatherDataset


# TODO: Add this information to README.md
# ! TO RUN IN TERMINAL (WINDOWS): set PYTHONPATH=%PYTHONPATH%;<PATH_TO_PROJECT>
# THEN RUN: python models/train_model.py --data_path <PATH_TO_DATASET> -- device <DEVICE> --epochs <EPOCHS> --model_path <MODEL_PATH>
# ! TO RUN IN TERMINAL (LINUX/MACOS): export PYTHONPATH=$PYTHONPATH:<PATH_TO_PROJECT>
# THEN RUN: python models/train_model.py --data_path <PATH_TO_DATASET> -- device <DEVICE> --epochs <EPOCHS> --model_path <MODEL_PATH>

def arguments():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Weather classification')
    parser.add_argument('--data', type=str,
                        default=r'C:\Users\Anwender\Desktop\Nicolas\Dokumente\FH Bielefeld\Optimierung und Simulation\2. Semester\SimulationOptischerSysteme\AI-Weather-Classification\dataset',
                        help='Path to the dataset')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on',
                        choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--epochs', type=int, default=28, help='Number of epochs to train the model')
    parser.add_argument('--model_path', type=str, default='trained_model.pth',
                        help='Path to save the trained model')
    parser.add_argument('--verbose', type=bool, default=True, help='Print model details', choices=[True, False])

    args = parser.parse_args()
    data = args.data

    # make data path to raw string
    data = r"{}".format(data)

    compute_device = args.device
    training_epochs = args.epochs
    model_file_path = args.model_path
    verbosity = args.verbose

    return data, compute_device, training_epochs, model_file_path, verbosity


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

    def __init__(self, num_classes: int, compute_device: torch.device, image_size: tuple = (512, 512),
                 verbosity: bool = True):
        super(WeatherClassifier, self).__init__()
        self.device = compute_device
        self.num_classes = num_classes
        self.image_size = image_size
        self.verbose = verbosity

        if self.verbose:
            print("=" * 200)

        # Initialize network architecture
        self._initialize_layers()

        # Set up the classifier part of the network
        self._setup_classifier()

        logger.info("Model initialized successfully.")

        # Move the model to the specified device
        self.to(self.device)
        logger.info("Model moved to device.", device=self.device)

        # Init attributes for training
        self.criterion = None
        self.optimizer = None
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def _initialize_layers(self):
        """ Initialize the convolutional layers of the network """
        self.num_layers = 3
        self.channels = [3, 64, 512, 1024, 4096]
        self.kernel_size_conv = 3
        self.stride_conv = 2
        self.padding_conv = 1
        self.kernel_size_pool = 3
        self.stride_pool = 2
        self.activation_function = nn.LeakyReLU()

        layers = []
        for i in range(self.num_layers):
            layers.extend([
                nn.Conv2d(self.channels[i], self.channels[i + 1], kernel_size=self.kernel_size_conv,
                          stride=self.stride_conv,
                          padding=self.padding_conv),
                self.activation_function,
                nn.MaxPool2d(kernel_size=self.kernel_size_pool, stride=self.stride_pool)
            ])

        self.features = nn.Sequential(*layers)
        self.features.to(self.device)
        self._calculate_feature_size()

        if self.verbose:
            logger.info("Convolutional Layers Setup")
            table = tabulate({
                'Number of Layers': [self.num_layers],
                'Channels': [self.channels],
                'Kernel Size Convolutional': [self.kernel_size_conv],
                'Stride Convolutional': [self.stride_conv],
                'Padding Convolutional': [self.padding_conv],
                'Kernel Size Pooling': [self.kernel_size_pool],
                'Stride Pooling': [self.stride_pool],
                'Activation Function': [self.activation_function]
            }, headers='keys', tablefmt='pretty')
            print(table)
            print()

    def _setup_classifier(self):
        """ Set up the classifier part of the network """
        dropout_rate = 0.357273264918355
        self.num_feature = 1024
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, self.num_feature),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_feature, self.num_classes)
        )

        if self.verbose:
            logger.info("Classifier Setup")
            table = tabulate({
                'Dropout Rate': [dropout_rate],
                'Number of features': [self.num_feature]
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

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, training_epochs: int, learning_rate: float,
                    model_file_path: str, compute_device: torch.device, weight_decay: float = 1e-5):
        """
        Trains the model using the provided data loaders and training parameters.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            training_epochs (int): Number of epochs to train the model.
            learning_rate (float): Learning rate for the optimizer.
            model_file_path (str): Path where the trained model will be saved.
            compute_device (torch.device): Device on which the model should be trained.
            weight_decay (float): Weight decay to prevent overfitting during optimization.

        Returns:
            None: This method saves the trained model to the specified path and plots the training results.
        """
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        criterion_name = self.criterion.__class__.__name__
        optimizer_name = self.optimizer.__class__.__name__

        if self.verbose:
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
        for epoch in range(training_epochs):
            self.run_epoch(train_loader, val_loader, device)

            if self.verbose:
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

        try:
            torch.save(self.state_dict(), model_file_path)
            logger.info("Model saved successfully.", model_path=model_file_path)
        except Exception as e:
            logger.error(f"Error saving model: {e}")

        if self.verbose:
            self.plot_results()

    def run_epoch(self, train_loader: DataLoader, val_loader: DataLoader, compute_device: torch.device):
        """
        Executes a training and validation run for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            compute_device (torch.device): Device to run the training on (e.g., 'cpu' or 'cuda').

        Returns:
            None
        """
        train_loss, train_accuracy = self.train_one_epoch(train_loader, compute_device)
        val_loss, val_accuracy = self.validate_one_epoch(val_loader, compute_device)

        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.train_acc_history.append(train_accuracy)
        self.val_acc_history.append(val_accuracy)

    def train_one_epoch(self, train_loader: DataLoader, compute_device: torch.device) -> Tuple[float, float]:
        """
        Conducts training on the training dataset for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            compute_device (torch.device): Device on which to perform the training operations.

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
            correct += (predicted.eq(labels)).sum().item()

        # noinspection PyTypeChecker
        return train_loss / len(train_loader.dataset), 100 * correct / total

    def validate_one_epoch(self, val_loader: DataLoader, compute_device: torch.device) -> Tuple[float, float]:
        """
        Validates the model on the validation dataset for one epoch.

        Args:
            val_loader (DataLoader): DataLoader for the validation data.
            compute_device (torch.device): Device on which to perform the validation operations.

        Returns:
            tuple: Contains average validation loss and validation accuracy for the epoch.
        """
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(compute_device), labels.to(compute_device)
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.eq(labels)).sum().item()

        # noinspection PyTypeChecker
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

    def optimize_hyperparameters_lr_wd(self, train_loader: DataLoader, val_loader: DataLoader, training_epochs: int,
                                       compute_device: torch.device, n_trials: int = 50,
                                       study_name: str = "weather-classification"):
        """
        Optimize hyperparameters for learning rate and weight decay using Optuna.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            training_epochs (int): Number of epochs to run for each trial.
            compute_device (torch.device): Device to perform the optimization on.
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
            for epoch in range(training_epochs):
                self.train()  # set the model to training mode
                train_loss = 0.0
                for inputs, labels in tqdm(train_loader):
                    inputs, labels = inputs.to(compute_device), labels.to(compute_device)  # move data to device

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
                        inputs, labels = inputs.to(compute_device), labels.to(compute_device)  # move data to device
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)

            # noinspection PyTypeChecker
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
        best_params_keys = best_params.keys()
        best_params_values = best_params.values()

        if self.verbose:
            table = tabulate({
                'Best Parameters': best_params_keys,
                'Values': best_params_values
            }, headers='keys', tablefmt='pretty')
            logger.info("Best Parameters")
            print(table)
            print()

        print(f'Best parameters: {best_params}')

    def test(self, test_loader: DataLoader, model_file_path: str, compute_device: torch.device):
        """
        Tests the model's performance on a provided test dataset.

        Args:
            test_loader (DataLoader): DataLoader for the test data.
            model_path (str): Path to the trained model file.
            device (torch.device): Device to perform the test on.

        Returns:
            None: This method prints the accuracy of the model on the test dataset.
        """
        self.load_state_dict(torch.load(model_file_path))
        self.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(compute_device), labels.to(compute_device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.eq(labels)).sum().item()

        pred_acc = round((100 * correct / total), 3)

        # noinspection PyTypeChecker
        table = tabulate({
            'Number of Test Images': [len(test_loader.dataset)],
            'Accuracy': [f'{pred_acc}%']
        }, headers='keys', tablefmt='pretty')
        logger.info("Test Results")
        print(table)
        print()

    def optimize_hyperparameters_large(self, data_path: str, compute_device: torch.device, n_trials: int = 100):
        """
        Optimizes hyperparameters of a neural network using Optuna over a defined number of trials.

        Performs hyperparameter optimization by testing different configurations of learning rates, weight decay, batch
        sizes, activations, and dropout rates. Evaluates model performance based on accuracy metrics from validation and
        test data sets. Results, including the best parameters found, are printed and logged.

        Args:
            data_path (str): Path to the dataset.
            compute_device (torch.device): Compute device (CPU or GPU).
            n_trials (int, optional): Number of optimization trials. Defaults to 100.

        Note:
            Assumes the availability of a specific dataset structure and device compatibility.
        """

        def objective(trial) -> float:
            # Define hyperparameters
            lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-2)
            batch_size = trial.suggest_int('batch_size', 2, 32)
            activation_name = trial.suggest_categorical('activation', ['ReLU', 'SiLU', 'LeakyReLU'])
            dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
            training_epochs = trial.suggest_int('epochs', 5, 50)

            # Set activation function
            activation = None
            if activation_name == 'ReLU':
                activation = nn.ReLU()
            elif activation_name == 'SiLU':
                activation = nn.SiLU()
            elif activation_name == 'LeakyReLU':
                activation = nn.LeakyReLU()

            layers = []
            for i in range(self.num_layers):
                layers.append(nn.Conv2d(self.channels[i], self.channels[i + 1], kernel_size=self.kernel_size_conv,
                                        stride=self.stride_conv,
                                        padding=self.padding_conv))
                layers.append(activation)
                layers.append(nn.MaxPool2d(kernel_size=self.kernel_size_pool, stride=self.stride_pool))

            self.features = nn.Sequential(*layers)
            self.to(device)

            input_size = self.image_size[0]

            # Calculate the size of the feature maps
            for i in range(self.num_layers):
                output_size_conv = (input_size - self.kernel_size_conv + 2 * self.padding_conv) / self.stride_conv + 1
                input_size = output_size_conv
                output_size_pool = (input_size - self.kernel_size_conv) / self.stride_pool + 1
                input_size = output_size_pool

            # Pass a dummy input through the features module
            dummy_output = None
            try:
                dummy_input = torch.ones(1, 3, int(input_size), int(input_size)).to(compute_device)
                dummy_output = self.features(dummy_input)
            except Exception as e:
                logger.error(f"Error: {e}")

            num_features = dummy_output.view(dummy_output.size(0), -1).size(1)

            # classifier layer
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, self.num_features),
                activation,
                nn.Dropout(p=dropout),
                nn.Linear(self.num_feature, self.num_classes),
            )
            self.to(compute_device)

            # Training and validation logic
            criterion = nn.CrossEntropyLoss()
            weather_model = WeatherDataset(data_folder=data_path)

            train_loader, val_loader, test_loader = weather_model.get_data_loaders(batch_size=batch_size)

            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

            # start training loop
            val_accuracies = {}
            test_accuracy = 0
            for epoch in range(training_epochs):
                self.train()  # set the model to training mode
                train_loss = 0.0
                for inputs, labels in tqdm(train_loader):
                    inputs, labels = inputs.to(compute_device), labels.to(compute_device)  # move data to device

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
                        inputs, labels = inputs.to(compute_device), labels.to(compute_device)  # move data to device
                        outputs = self(inputs)
                        _, predicted = torch.max(outputs.data, 1)  # get the predicted class
                        total_val += labels.size(0)
                        correct_val += (predicted.eq(labels)).sum().item()
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(compute_device), labels.to(compute_device)  # move data to device
                        outputs = self(inputs)
                        _, predicted = torch.max(outputs.data, 1)  # get the predicted class
                        total_test += labels.size(0)
                        correct_test += (predicted.eq(labels)).sum().item()
                        loss = criterion(outputs, labels)
                        test_loss += loss.item() * inputs.size(0)

                val_accuracy = 100 * correct_val / total_val  # calculate validation accuracy
                val_accuracies[epoch] = val_accuracy  # store validation accuracy
                test_accuracy = 100 * correct_test / total_test  # calculate validation accuracy

            # log accuracies
            if self.verbose:
                logger.info(f"Validation accuracies: {val_accuracies}")
                logger.info(f"Test accuracy: {test_accuracy}")

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
        best_params_keys = best_params.keys()
        best_params_values = best_params.values()

        if self.verbose:
            table = tabulate({
                'Best Parameters': best_params_keys,
                'Values': best_params_values
            }, headers='keys', tablefmt='pretty')
            logger.info("Best Parameters")
            print(table)
            print()


if __name__ == "__main__":
    # path_dataset = "/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/dataset"
    one_image_data = "/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/utils/one_image"
    # model_path = "trained_model.pth"

    path_dataset_win = r"C:\Users\Anwender\Desktop\Nicolas\Dokumente\FH Bielefeld\Optimierung und Simulation\2. Semester\SimulationOptischerSysteme\AI-Weather-Classification\dataset"
    # one_image_data_win = r"C:\Users\Anwender\Desktop\Nicolas\Dokumente\FH Bielefeld\Optimierung und Simulation\2. Semester\SimulationOptischerSysteme\AI-Weather-Classification\test_image"

    # Get arguments
    path_dataset, device, epochs, model_path, verbose = arguments()

    logger.info("Parameters:")
    print(f"Path to dataset: {path_dataset}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Model path: {model_path}")
    print(f"Verbose: {verbose}")
    print()

    device = torch.device(device)
    W = WeatherDataset(data_folder=path_dataset, resize_format=(512, 512))
    model = WeatherClassifier(num_classes=len(W.dataset.classes), compute_device=device, image_size=W.resize_format,
                              verbosity=verbose)

    # Get data loaders
    tr, val, test = W.get_data_loaders(batch_size=5)

    # Train model
    logger.info("START TRAINING WEATHER CLASSIFIER MODEL")
    model.train_model(train_loader=tr, val_loader=val, training_epochs=epochs, learning_rate=2.1788228027184658e-05,
                      model_file_path=model_path, compute_device=device, weight_decay=1.592566270355879e-06)

    # Test model
    model.test(test, model_path, device)

    logger.info("Model trained and tested successfully.")
    print("-" * 100)
    print("=========FINISHED=========")
