from src.training.train_model import WeatherClassifier
import torch
from torchview import draw_graph

# Set device
device = torch.device("cpu")

# Create model instance
model = WeatherClassifier(11, device)

# Create a dummy input tensor appropriate for the model input
dummy_input = torch.randn(1, 3, 512, 512).to(device)

# Generate the visual graph
model_graph = draw_graph(model, dummy_input)

# Save the visual graph as PNG
model_graph.visual_graph.render('model_graph', format='pdf', cleanup=True)
