import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        """
        Args:
        - input_dim (int): Dimension of the input features.
        - hidden_dim (list of int): List containing the dimensions of hidden layers.
        - output_dim (int): Dimension of the output layer.
        """
        super(MLP, self).__init__()
        
        # Create a list of layers
        layers = []
        prev_dim = hidden_dim[0]

        # Hidden layers
        for h_dim in hidden_dim[1:]:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())  # Activation function
            prev_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Sequentially apply layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)