# AI-Weather-Classification

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Optimize Hyperparameters](#optimize-hyperparameters)
  - [Evaluation](#evaluation)
  - [Prediction](#prediction)


## Project Description
This project is an implementation of a neural network using PyTorch to classify the weather conditions in images. 
It uses machine learning techniques to accurately predict weather conditions based on the input images.

## Installation
To install the project, make sure you have installed a conda.

Look at the [conda documentation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for more information.

Now, follow these steps to install the project:

1. Clone the repository:
```bash
git clone https://github.com/Nicolas2912/AI-Weather-Classification.git
cd AI-Weather-Classification
```

2. Create a new conda environment:
```bash
conda create -n weather-classification python=3.9.13
```

3. Activate the conda environment:
```bash
conda activate weather-classification
```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To **train** you own model run the following command:
```bash
python src/training/train_model.py
```

This will train a model using the images in the `dataset` folder with the default arguments. 
The model with name `weather-model` will be saved in the `models` folder. Following arguments can be specified:

- `--data`: Path to the dataset folder. Default: `dataset` directory within the project root
- `--device`: Device to train the model on. Default: `cuda` if available, else `cpu`. Allowed values: `cpu`, `cuda`, `mps`.
- `--epochs`: Number of epochs to train the model. Default: `28`.
- `--model_path`: Path to save the trained model. Default: `models/weather-model.pth`.
- `--verbose`: Print model training progress and other model details. Default: `True`.

After finishing the training, the model will also be evaluated on a test dataset. The evaluation results will be printed
to the console. The training results are saved in the `experiments` folder.

Example usage with arguments:
```bash
python src/training/train_model.py --data dataset --device cuda --epochs 30 --model_path models/my-weather-model.pth --verbose True
```

### Optimize Hyperparameters

To **optimize** the hyperparameters of the model run the following command:
```bash
python src/training/optimize_hyperparameters.py
```

This will optimize the hyperparameters of the model using `optuna`. 

Following arguments can be specified:

- `--data`: Path to the dataset folder. Default: `dataset` directory within the project root.
- `--n_trials`: Number of optimization trials to run. Default: `50`.
- `--epochs`: Number of epochs for training during each trial. Default: `10`.
- `--device`: Device to train the model on. Default: `cuda` if available, else `cpu`. Allowed values: `cpu`, `cuda`, `mps`.
- `--mode`: Mode for optimization, indicating the scale of the optimization process. Default: `small`. Allowed values: `small`, `large`.
- `--study_name`: Name of the study to save the optimization results. Default: Automatically generated name based on the current date and time.

If `mode==large` you don't need to specify the `epochs` since it is a hyperparameter that will be optimized. 

You can inspect the optimization process with`optuna-dashboard`. Make sure `optuna-dashboard` is installed. You can 
install it with:
```bash
pip install optuna-dashboard
```

After the package is installed, you can start the dashboard with:
```bash
optuna-dashboard sqlite:///db.sqlite3
```

Example usage with arguments:
```bash
python src/training/optimize_hyperparameters.py --data dataset --n_trials 100 --epochs 20 --device cuda --mode small --study_name my-study
```

### Evaluation

To **evaluate** the model run the following command:
```bash
python src/evaluation/evaluation.py
```

This will evaluate the model on the test dataset. The evaluation results will be printed to the console and the 
confusion matrix will be saved in the `experiments` folder.

Following arguments can be specified:

- `--data`: Path to the dataset folder. Default: `dataset` directory within the project root.
- `--model`: Path to the model file. Default: `models/weather-model.pth`.
- `--device`: Device to run the evaluation on. Default: `cuda` if available, else `cpu`. Allowed values: `cpu`, `cuda`, `mps`.

Example usage with arguments:
```bash
python src/evaluation/evaluation.py --data dataset --model models/my-weather-model.pth --device cuda
```

### Prediction

To **predict** the weather condition of an image run the following command:
```bash
python src/prediction/predict.py --image_path path/to/image.jpg
```

This will predict the weather condition of the image or images specified by `image_path` using the trained model. The 
prediction results will be printed to the console.

Following arguments can be specified:

- `--model`: Path to the trained model file. Default: `models/weather-model.pth`.
- `--images`: Path the directory containing classes of the images. Default: `test_image` directory within the project root.
- `--dataset`: Path to the dataset directory used to initialize the dataset structure and classes. Default: `dataset` directory within the project root.
- `--device`: Device to run the prediction on. Default: `cuda` if available, else `cpu`. Allowed values: `cpu`, `cuda`, `mps`.

Example usage with arguments:
```bash
python src/prediction/predict.py --model models/my-weather-model.pth --images test_image --dataset dataset --device cuda
```
