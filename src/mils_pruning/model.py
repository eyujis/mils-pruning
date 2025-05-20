# src/mils_pruning/model.py

import torch
import torch.nn as nn


class Binarize(torch.autograd.Function):
    """Straight-Through Estimator (STE) for sign-based binarization."""
    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Pass-through gradient


class BinaryLinear(nn.Module):
    """Linear layer with binarized weights and BatchNorm."""
    def __init__(self, in_features, out_features):
        super(BinaryLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bn = nn.BatchNorm1d(out_features)

        # Weight initialization
        nn.init.normal_(self.weight, mean=0.0, std=0.1)

    def forward(self, x):
        W_b = Binarize.apply(self.weight)
        out = nn.functional.linear(x, W_b) + self.bias
        return self.bn(out)


class BinarizedMLP(nn.Module):
    """
    Binarized Multi-Layer Perceptron with binarized weights and activations.
    
    Parameters
    ----------
    input_shape : Tuple[int, int]
        Height and width of the input image (e.g., (10, 10)).
    nodes_h1 : int
        Number of hidden units in first hidden layer.
    nodes_h2 : int
        Number of hidden units in second hidden layer.
    """
    def __init__(self, input_shape, nodes_h1, nodes_h2):
        super(BinarizedMLP, self).__init__()
        in_dim = input_shape[0] * input_shape[1]
        self.fc1 = BinaryLinear(in_dim, nodes_h1)
        self.fc2 = BinaryLinear(nodes_h1, nodes_h2)
        self.fc3 = BinaryLinear(nodes_h2, 10)  # Output logits for 10 classes

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = Binarize.apply(x)

        x = self.fc2(x)
        x = Binarize.apply(x)

        x = self.fc3(x)
        return x
