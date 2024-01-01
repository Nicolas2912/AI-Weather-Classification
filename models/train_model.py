import torch
from torch import nn
import optuna
from datetime import datetime

from utils.data_loader import WeatherDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet


class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        self.fc = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x

    def train_model(self, train_loader, val_loader, epochs, learning_rate, model_path, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        train_loss_history = []
        val_loss_history = []

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.half().to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                outputs = outputs.float()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_loss_history.append(train_loss)

            self.eval()  # set the model to evaluation mode
            with torch.no_grad():
                val_loss = 0.0
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)  # move data to device
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    val_loss_history.append(val_loss)

            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)

            print(
                f'\nEpoch: {epoch + 1}/{epochs}.. Training Loss: {train_loss:.3f}.. Validation Loss: {val_loss:.3f}\n')

        plt.figure(figsize=(10, 5))
        plt.plot(range(epochs), train_loss_history, label='Training Loss')
        plt.plot(range(epochs), val_loss_history, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(
            '/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/experiments/loss.png')
        plt.show()

        torch.save(self.state_dict(), model_path)

    def to_fp16(self):
        self.half()
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

        return self

    def test(self, test_loader, model_path, device):
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

        print(f'Accuracy of the model on {total} test images: {100 * correct / total}%')


class WeatherClassifier(nn.Module):
    def __init__(self, num_classes):
        super(WeatherClassifier, self).__init__()

        # parameters to adjust
        kernel_size_conv = 3
        stride_conv = 2 # change this parameter
        padding_conv = 1

        kernel_size_pool = 2
        stride_pool = 2

        # feature layer?
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_pool),
            nn.Conv2d(64, 128, kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_pool)
        )

        length = 14
        height = 14
        # self.avgpool = nn.AdaptiveAvgPool2d((length, height))

        # classifier layer?
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.45),
            nn.Linear(128 * length * height, 1024),
            nn.SiLU(),
            nn.Dropout(p=0.45),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # maybe sigmoid for more than 2 weather situations in one image
        return x

    def train_model(self, train_loader, val_loader, epochs, learning_rate, model_path, device, weight_decay=1e-5):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_loss_history = []
        val_loss_history = []
        val_acc_history = []

        # print(f"Start train")

        for epoch in (range(epochs)):
            self.train()  # set the model to training mode
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader):

                inputs, labels = inputs.float().to(device), labels.to(device)  # move data to device

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            self.eval()  # set the model to evaluation mode
            correct = 0
            total = 0
            with torch.no_grad():
                val_loss = 0.0
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)  # move data to device
                    outputs = self(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = 100 * correct / total  # calculate validation accuracy
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_accuracy)  # store validation accuracy

            print(
                f'\nEpoch: {epoch + 1}/{epochs}.. Training Loss: {train_loss:.3f}.. Validation Loss: {val_loss:.3f}.. Validation accuracy {100 * correct / total}%\n')

        accuracy = 100 * correct / total
        # save the model
        torch.save(self.state_dict(), model_path)

        # plot the training and validation loss and accuracy
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # create two subplots
        ax1.set_title(
            f"Learning rate: {round(learning_rate, 5)}; Epochs: {epochs};\nAccuracy: {round(val_accuracy, 3)}%")
        ax1.plot(range(epochs), train_loss_history, label='Training Loss')
        ax1.plot(range(epochs), val_loss_history, label='Validation Loss')
        ax1.grid(True)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(range(epochs), val_acc_history, label='Validation Accuracy')  # plot validation accuracy
        ax2.grid(True)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # time now
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        plt.savefig(
            f'/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/experiments/loss_{current_time}.png')
        plt.show()

    def optimize_hyperparameters(self, train_loader, val_loader, epochs, model_path, device, n_trials=100):
        def objective(trial):
            # Define hyperparameters
            lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-2)
            # batch_size = trial.suggest_int('batch_size', 1, 128)

            # Initialize model, criterion, optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

            # Get data loaders with the suggested batch size
            # train_loader_optuna, val_loader_optuna, _ = WeatherDataset.get_data_loaders(batch_size=batch_size)

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
            study_name="weather-classification_2",
            direction='minimize'
        )
        study.optimize(objective, n_trials=n_trials)

        # Get the best parameters
        best_params = study.best_params

        print(f'Best parameters: {best_params}')

    def test(self, test_loader, model_path, device):
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
        print(f'Accuracy of the model on {len(test_loader.dataset)} test images: {pred_acc}%')

    def predict(self, test_loader, device):
        # Load the trained model
        self.load_state_dict(torch.load("trained_model.pth"))
        self.eval()  # set the model to evaluation mode

        correct = 0
        total = 0
        num_images_to_plot = 6  # number of images to plot
        plotted_images = 0

        # Create a figure with 10 subplots for each side (predicted and actual)
        fig, axs = plt.subplots(num_images_to_plot, 2, figsize=(10, 50))

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Plot the images with their predicted and actual labels
                for i in range(inputs.size(0)):
                    input_data = inputs[i].cpu().numpy().transpose((1, 2, 0))  # convert the tensor to numpy array

                    # Subplot for the predicted label
                    axs[plotted_images, 0].imshow(input_data)
                    axs[plotted_images, 0].set_title(f'Predicted: {predicted[i]}',
                                                     color='green' if predicted[i] == labels[i] else 'red')

                    # Subplot for the actual label
                    axs[plotted_images, 1].imshow(input_data)
                    axs[plotted_images, 1].set_title(f'Actual: {labels[i]}')

                    plotted_images += 1
                    if plotted_images == num_images_to_plot:
                        break

                if plotted_images == num_images_to_plot:
                    break

        plt.tight_layout()
        plt.show()

        print(f'Accuracy of the network on the test images: {100 * correct / total}%')

    def predict_image(self, one_image_loader, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()
        predictions = []

        with torch.no_grad():
            for inputs, labels in one_image_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.append(predicted)
                # print(f"Prediction: {predicted}")

        return predictions


if __name__ == "__main__":
    path_dataset = "/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/dataset"
    one_image_data = "/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/utils/one_image"
    model_path = "trained_model.pth"

    # mpd backend
    device = torch.backends.mps.is_available()
    if device:
        device = torch.device("mps")
        # print("MPS device found.")



    # device = torch.device("cpu")
    W = WeatherDataset(data_folder=path_dataset)
    tr, val, test = W.get_data_loaders(batch_size=16)

    # model = CustomEfficientNet(num_classes=11).to_fp16().to(device)
    # num_classes_all = len(W.dataset.classes)
    # model = WeatherClassifier(num_classes_all)
    # model.train_model(tr, val, 3, 0.00010758019268037226, model_path, device, 8.465089413204625e-06)
    # model.test(test, model_path, device)
    # W_one = WeatherDataset(data_folder=one_image_data)
    # W = WeatherDataset(data_folder=path_dataset)
    # # one_img_loader = W_one.one_image_loader()
    # tr, val, test = W.get_data_loaders(batch_size=64)
    # # num_classes_one = len(W_one.dataset.classes)
    num_classes_all = len(W.dataset.classes)
    #
    model = WeatherClassifier(num_classes_all)
    model.to(device)
    # # tune hyperparameters
    #model.optimize_hyperparameters(train_loader=tr, val_loader=val, epochs=10, model_path=model_path, device=device,
    #                               n_trials=10)
    #
    print("Start train")
    model.train_model(train_loader=tr, val_loader=val, epochs=10, learning_rate=0.00010758019268037226, model_path=model_path,
                        device=device, weight_decay=8.465089413204625e-06)
    print("End Train")
    #
    model.test(test, model_path, device)
    # predictions = model.predict_image(one_image_loader=one_img_loader, model_path=model_path)
    # class_to_idx_dict = W.dataset.class_to_idx
    # prediction = model.test(test, model_path, device)

    # idx_to_class_dict = {value: key for key, value in class_to_idx_dict.items()}

    # print(predictions)
    #
    # for pred in predictions:
    #     pred_num = pred[0].item()
    #     print(f"Prediction: {idx_to_class_dict[pred_num]}")

    # W = WeatherDataset(data_folder=path_dataset)
    # print(W.dataset.class_to_idx)
    # tr, val, test = W.get_data_loaders(batch_size=32, train_ratio=0.7)
    # print("Data loaded finished")
    #
    # device = torch.device("cpu")
    # AI = WeatherClassifier(num_classes=len(W.dataset.classes)).to(device)
    # AI.train_model(tr, val, epochs=2, learning_rate=0.01, model_path=model_path, device=device)
    # AI.test(test, model_path, device)
