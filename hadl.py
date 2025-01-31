import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function
from scipy.fft import dct, idct




class DiscreteCosineTransform(Function):
        """
        Custom PyTorch implementation of the Discrete Cosine Transform (DCT) and its inverse (IDCT).
        
        This class provides a differentiable forward and backward pass for DCT, allowing it to be used
        in neural network training.
        """
        
        @staticmethod
        def forward(ctx, input):
            """
            Forward pass: Applies the Discrete Cosine Transform (DCT).
            
            Parameters:
            - input (Tensor): Input tensor of shape [..., Length].
            
            Returns:
            - Tensor: DCT-transformed tensor with the same shape as input.
            """
            # Convert PyTorch tensor to NumPy array
            input_np = input.cpu().numpy()
            
            # Apply DCT using scipy with orthonormalization
            transformed_np = dct(input_np, type=2, norm='ortho', axis=-1)
            
            # Convert back to PyTorch tensor and move to original device
            output = torch.from_numpy(transformed_np).to(input.device)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            """
            Backward pass: Applies the Inverse Discrete Cosine Transform (IDCT) for gradient computation.
            
            Parameters:
            - grad_output (Tensor): Gradient of the loss with respect to the output.
            
            Returns:
            - Tensor: Gradient of the loss with respect to the input.
            """
            # Convert gradient tensor to NumPy array
            grad_output_np = grad_output.cpu().numpy()
            
            # Apply IDCT using scipy with orthonormalization
            grad_input_np = idct(grad_output_np, type=2, norm='ortho', axis=-1)
            
            # Convert back to PyTorch tensor and move to original device
            grad_input = torch.from_numpy(grad_input_np).to(grad_output.device)
            return grad_input


class LowRank(nn.Module):
    """
    This layer performs a low-rank approximation by factorizing the weight matrix into two smaller matrices.
    
    Parameters:
    - in_features (int): Number of input features.
    - out_features (int): Number of output features.
    - rank (int): Rank for the low-rank decomposition.
    - bias (bool, optional): If True, includes a bias term. Default is True.
    """
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LowRank, self).__init__()
        self.in_features = in_features  # Input feature dimension
        self.out_features = out_features  # Output feature dimension
        self.rank = rank  # Rank for the low-rank decomposition
        self.bias = bias  # Whether to include a bias term

        # Initialize factorized weight matrices A and B
        wA = torch.empty(self.in_features, rank)
        wB = torch.empty(self.rank, self.out_features)
        
        self.A = nn.Parameter(nn.init.kaiming_uniform_(wA))  # First weight matrix
        self.B = nn.Parameter(nn.init.kaiming_uniform_(wB))  # Second weight matrix

        # Initialize bias term if enabled
        if self.bias:
            wb = torch.empty(self.out_features)
            self.b = nn.Parameter(nn.init.uniform_(wb))

    def forward(self, x):
        """
        Forward pass of the Low-Rank layer.
        
        Parameters:
        - x (Tensor): Input tensor of shape [Batch, Input Features].
        
        Returns:
        - Tensor: Output tensor of shape [Batch, Output Features].
        """
        out = x @ self.A  # First projection step
        out = out @ self.B  # Second projection step

        # Add bias if enabled
        if self.bias:
            out += self.b
        
        return out

class HADL(nn.Module):
    """
    Haar-DCT Low-Rank framework for time series forecasting.

    This model processes input sequences using Haar Wavelet Transform (Haar) and 
    Discrete Cosine Transform (DCT) before applying a low-rank layer for forecasting.

    Parameters:
    - seq_len (int): Length of the input sequence.
    - pred_len (int): Length of the predicted sequence.
    - features (int): Number of input features (channels).
    - rank (int, optional): Rank of the low-rank decomposition. Default is 30.
    - individual (bool, optional): If True, uses separate low-rank layers for each feature;
      otherwise, a shared low-rank layer is used. Default is False.
    - bias (bool, optional): If True, enables bias in the low-rank layer. Default is True.
    - enable_Haar (bool, optional): If True, applies Haar transform before further processing. Default is True.
    - enable_DCT (bool, optional): If True, applies DCT transformation. Default is True.
    """
    def __init__(self, seq_len, pred_len, features, rank=30, individual=False, bias=True, enable_Haar=True, enable_DCT=True):
        super(HADL, self).__init__()
        
        self.seq_len = seq_len  # Input sequence length
        self.pred_len = pred_len  # Output sequence length (forecast horizon)
        self.rank = rank  # Rank for the low-rank layer
        self.features = features  # Number of features (channels)
        
        # Flags to enable or disable specific transformations
        self.bias = bias  # Controls whether the low-rank layer includes bias
        self.individual = individual  # Controls whether each feature has its own low-rank layer
        self.enable_Haar = enable_Haar  # Flag to enable Haar decomposition
        self.enable_DCT = enable_DCT  # Flag to enable Discrete Cosine Transform
        
        # Define Haar low-pass filter for downsampling
        # Averaging Filter coefficients: [1/sqrt(2), 1/sqrt(2)]
        self.low_pass_filter = torch.tensor([1, 1], dtype=torch.float32) / math.sqrt(2)
        self.low_pass_filter = self.low_pass_filter.reshape(1, 1, -1).repeat(self.features, 1, 1)

        # Determine the input size for the low-rank layer after optional Haar transformation
        if enable_Haar:
            # Haar reduces sequence length by half, but if seq_len is odd, round up to even.
            in_len = (self.seq_len // 2) + 1 if (self.seq_len % 2) != 0 else (self.seq_len // 2)
        else:
            in_len = self.seq_len  # Without Haar, the full sequence length is used.

        # Initialize the Low-Rank layer(s)
        if self.individual:
            # If `individual` is True, create separate low-rank layers for each feature
            self.layers = nn.ModuleList([
                LowRank(in_features=in_len, out_features=self.pred_len, rank=self.rank, bias=self.bias)
                for _ in range(self.features)
            ])
        else:
            # Otherwise, use a single shared low-rank layer for all features
            self.layer = LowRank(in_features=in_len, out_features=self.pred_len, rank=self.rank, bias=self.bias)



    def forward(self, x):
        """
        Forward pass of the HADL model.
        
        Parameters:
        - x (Tensor): Input tensor of shape [Batch, Input length, Channel].
        
        Returns:
        - Tensor: Output tensor of shape [Batch, Pred length, Channel].
        """
        batch_size, _, _ = x.shape

        # Transpose x to shape [Batch, Channel, Input length] for processing
        x = x.permute(0, 2, 1)
        
        # Scaled normalization (zero-centering each sequence)
        seq_mean = torch.mean(x, axis=-1, keepdim=True)
        x = x - seq_mean
        
        # Apply Haar decomposition if enabled
        if self.enable_Haar:
            if self.seq_len % 2 != 0:
                # Pad sequence if length is odd to maintain even-length output after Haar
                x = F.pad(x, (0, 1))
            
            # Ensure low-pass filter is on the correct device
            self.low_pass_filter = self.low_pass_filter.to(x.device)
            
            # Apply Haar low-pass filtering (downsampling)
            x = F.conv1d(input=x, weight=self.low_pass_filter, stride=2, groups=self.features)

        # Apply Discrete Cosine Transform (DCT) if enabled, converting time domain to frequency domain
        if self.enable_DCT:
            x = DiscreteCosineTransform.apply(x) / x.shape[-1]

        # Prediction using the low-rank layer(s)
        if self.individual:
            # If `individual` is True, process each feature separately
            out = torch.empty(batch_size, self.features, self.pred_len, device=x.device)
            for i in range(self.features):
                pred = self.layers[i](x[:, i, :].view(batch_size, 1, -1))
                out[:, i, :] = pred.view(batch_size, -1)
        else:
            # Use shared low-rank layer
            out = self.layer(x)

        # Restore mean to the output sequence
        out = out + seq_mean
        
        # Transpose back to original shape [Batch, Output length, Channel]
        return out.permute(0, 2, 1)