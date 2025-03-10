from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNNModel(nn.Module):
    """
    A simple deep neural network model for binary classification.
    
    This model consists of three fully connected layers with ReLU activations
    for the hidden layers and sigmoid activation for the output layer.
    
    Attributes:
        fc1 (nn.Linear): First fully connected layer
        fc2 (nn.Linear): Second fully connected layer
        out (nn.Linear): Output layer
        
    Args:
        input_dim (int, optional): Dimension of the input features. Defaults to 5.
    """
    def __init__(self, input_dim=5):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        """
        Forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output probabilities of shape (batch_size, 1)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x 