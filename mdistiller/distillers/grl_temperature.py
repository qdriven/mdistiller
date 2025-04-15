import torch
import torch.nn as nn
from torch.autograd import Function

# Copied and adapted from https://github.com/zhengli97/CTKD/blob/main/models/temp_global.py

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        # Return the input tensor directly
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Get the lambda value saved in the forward pass
        lambda_ = ctx.lambda_
        # Reverse the gradient and multiply by lambda
        # Ensure lambda_ is a tensor for multiplication
        grad_input = grad_output.clone()
        return -lambda_ * grad_input, None

class GradientReversal(nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()

    def forward(self, x, lambda_):
        # Apply the Gradient Reversal Function
        return GradientReversalFunction.apply(x, lambda_)


class GlobalTemperature(nn.Module):
    """Learnable global temperature parameter using GRL."""
    def __init__(self, initial_temperature=4.0):
        super(GlobalTemperature, self).__init__()
        # Initialize the global temperature as a learnable parameter
        # Use a single scalar tensor for the parameter
        self.global_t = nn.Parameter(torch.tensor(initial_temperature, dtype=torch.float32), requires_grad=True)
        self.grl = GradientReversal()

    def forward(self, lambda_):
        """Applies GRL to the temperature parameter for gradient calculation,
           but returns the actual clamped temperature value for use in loss functions.
        """
        # Apply GRL. This primarily affects the backward pass.
        # The GRL function itself returns the input value in the forward pass.
        _ = self.grl(self.global_t, lambda_)
        # Return the actual current temperature value, clamped to avoid issues.
        return self.global_t.clamp(min=0.1) 