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
        # Store lambda for backward pass
        ctx.lambda_ = lambda_
        # Simply return the input tensor
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Get the lambda value saved in the forward pass
        lambda_ = ctx.lambda_
        
        # If no gradients are flowing in, return None
        if grad_output is None:
            return None, None
            
        # Convert lambda to tensor if needed
        if not isinstance(lambda_, torch.Tensor):
            lambda_ = torch.tensor(lambda_, device=grad_output.device, dtype=grad_output.dtype)
            
        # 增强梯度反转强度：使用更大的倍数（3倍）
        lambda_ = lambda_ * 3.0
            
        # Reverse gradients by multiplying by -lambda
        grad_input = -lambda_ * grad_output
        
        # 记录反转的梯度值，便于调试
        if torch.is_tensor(grad_input) and grad_input.numel() > 0:
            print(f"GRL grad magnitude: {grad_input.abs().mean().item():.4f}")
        
        # Return gradients for inputs, None for lambda (it's a hyperparameter)
        return grad_input, None

class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer module wrapper
    """
    def __init__(self):
        super(GradientReversal, self).__init__()
        
    def forward(self, x, lambda_=1.0):
        """
        Apply gradient reversal to input tensor
        
        Args:
            x: Input tensor
            lambda_: Coefficient for gradient reversal (default: 1.0)
            
        Returns:
            Tensor with same value as input but with reversed gradients
        """
        # Explicitly check if x requires gradients to ensure we're on the right track
        if not x.requires_grad:
            print(f"Warning: Input to GRL does not require gradients: {x}")
            x.requires_grad_(True)
            
        return GradientReversalFunction.apply(x, lambda_)

class GlobalTemperature(nn.Module):
    """
    Learnable global temperature parameter using Gradient Reversal Layer.
    
    This module maintains a single learnable parameter (temperature) and applies
    gradient reversal during backpropagation, allowing the temperature to be 
    optimized to maximize the distillation loss.
    """
    def __init__(self, initial_temperature=4.0):
        super(GlobalTemperature, self).__init__()
        # Initialize the temperature as an explicitly learnable parameter
        self.global_t = nn.Parameter(torch.tensor(initial_temperature, dtype=torch.float32))
        
        # Ensure requires_grad is enabled
        self.global_t.requires_grad_(True)
        
        # Create the gradient reversal layer
        self.grl = GradientReversal()
        
        print(f"GlobalTemperature module initialized:")
        print(f"  - initial_temperature: {initial_temperature}")
        print(f"  - requires_grad: {self.global_t.requires_grad}")
        print(f"  - device: {self.global_t.device}")

    def forward(self, lambda_=1.0):
        """
        Forward pass applying gradient reversal to the temperature.
        
        Args:
            lambda_: Gradient reversal strength coefficient
            
        Returns:
            Temperature value (with gradient reversal applied for backward pass)
        """
        # Check parameter state
        if not self.global_t.requires_grad:
            print("Warning: global_t lost requires_grad=True! Re-enabling...")
            self.global_t.requires_grad_(True)
            
        # Print debug info occasionally (every 100 calls based on id to avoid tracking state)
        if id(self) % 100 == 0:  # Simple way to print occasionally without maintaining state
            print(f"GlobalTemp forward: t={self.global_t.item():.4f}, requires_grad={self.global_t.requires_grad}")
            
        # Apply gradient reversal - this will negate gradients during backward pass
        t_with_grl = self.grl(self.global_t, lambda_)
        
        # 修改钳位范围，降低上限，提高下限
        clamped_t = torch.clamp(t_with_grl, min=1.0, max=8.0)  # 原来是 0.1-20.0
        
        return clamped_t 
