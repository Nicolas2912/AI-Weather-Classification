__project_name__ = "Weather Classifier"
__version__ = 0

print("=" * 10, f"STARTING PROJECT: {__project_name__} (V. {__version__})", "=" * 10)
print("\n")

# Import necessary modules and packages
import time
import argparse
import torch
from utils.data_loader import WeatherDataset
from models.train_model import WeatherClassifier

# TODO: Setting up input arguments and check if this is all correct
# Add arguments
args = argparse
parser = args.ArgumentParser(prog="Weather classification")
parser.add_argument("-gpu", "--gpu", type=bool, help="Boolean if you want to use gpu on Mac M1 or "
                                                     "NVIDIA Cuda graphics card", default=True)
parser.add_argument("-tr_tst", "--train_test", type=str, help="Decide whether to train or to test",
                    default="train", choices=["train", "test"])
parser.add_argument("-data_dir", "--data_directory", type=str, help="Directory path of the data",
                    default="/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/dataset")
parser.add_argument("-epo", "--epochs", type=int, help="Number of epochs to train", default=10)
parser.add_argument("-mod_path", "--model_path", type=str, help="Path to save the model (.pth) to",
                    default="trained_model.pth")
parser.add_argument("-img", "--test_images", type=str, help="Path of the image(s) for testing",
                    default="/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/utils/one_image")

args = parser.parse_args()

gpu = args.gpu
train_test = args.train_test
data_dir = args.data_directory
epochs = args.epochs
model_path = args.model_path
test_images = args.test_images

if gpu not in [False, True]:
    raise ValueError(f"Parameter gpu is {gpu}. GPU has to be True or False")
if epochs < 0:
    raise ValueError(f"Number of epochs must be greater 0")

# TODO: Use tabulate package here!
print(f"Parameters:")
print(
    f"GPU: {gpu}\nTrain or test: {train_test}\nDirectory of data: {data_dir}\nEpochs: {epochs}\nModel path: {model_path}")

print("-" * 100)

if __name__ == "__main__":
    # load data
    W = WeatherDataset(data_folder=data_dir)
    tr, val, test = W.get_data_loaders(batch_size=32, train_ratio=0.7)

    # settng up device
    if gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # train model
    if train_test == "train":
        s_train = time.time()
        model = WeatherClassifier(len(W.dataset.classes))

        print("Begin training\n")
        model.train_model(train_loader=tr, val_loader=val, epochs=epochs, learning_rate=0.001, model_path=model_path,
                          device=device)
        print(f"Model trained successfully in {time.time}s\n")

        print("Begin testing.")
        s_test = time.time()
        model.test(test_loader=test, model_path=model_path, device=device)
        print(f"Testing finished in {time.time() - s_test}s\n")

    # test model with custom images
    elif train_test == "test" and test_images != "None":
        Test_Dataset = WeatherDataset(data_folder=test_images)
        image_loader = Test_Dataset.one_image_loader()
        model = WeatherClassifier(11)
        predictions = model.predict_image(image_loader, model_path)

        prediction_dict = Test_Dataset.dataset.class_to_idx
        prediction_dict_rev = {value: key for key, value in prediction_dict.items()}

        # TODO: Plot images in matplotlib with picture (test image) and text (prediction)

    print("Processed finished")
    print("-" * 100)

    """
    gpu = True

    # 2. Set up device (CPU or GPU; Mac (M1) or NVIDIA)
    if torch.backends.mps.is_available() and gpu:
        device = torch.device("mps")
        print("MPS device found.")
    elif torch.cuda.is_available() and gpu:
        device = torch.device("cuda")
        print("CUDA device found.")
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU instead.")
    
    # 3. Load data using data loaders
    path_dataset = "/Users/nicolasschneider/MeineDokumente/FH_Bielefeld/Optimierung_und_Simulation/2. Semester/SimulationOptischerSysteme/AI-Weather-Classification/dataset"
    W_data = WeatherDataset(data_folder=path_dataset)
    train_loader, val_loader, test_loader = WeatherDataset.get_data_loaders(batch_size=32)

    # 4. Initialize the model
    num_classes = len(train_loader.dataset)
    model = WeatherClassifier().to(device)

    # 5. Train the model
    model.train_model(train_loader, val_loader, 100, 0.001, model_path)
    """
