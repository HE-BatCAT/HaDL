# HADL: Haar-DCT Low-Rank Model

## Overview
HADL (Haar-DCT Low-Rank) is a neural network framework designed for forecasting multivariate time series. It utilizes Haar Wavelet Transform (DWT) for noise reduction, Discrete Cosine Transform (DCT) for frequency domain representation, and a low-rank layer for efficient prediction. The model is designed to extract long-term periodic features while reducing computational complexity.

## Features
- **Haar Decomposition**: Reduces noise by extracting approximation coefficients from the input sequence.
- **Discrete Cosine Transform (DCT)**: Converts time-domain data into the frequency domain for better pattern recognition.
- **Low-Rank Layer**: Uses a low-rank matrix decomposition to learn compact and efficient representations.
- **Configurable Components**: Enables or disables Haar and DCT transformations based on user preferences.
- **Per-Feature or Generalized Layers**: Allows for independent layers per feature or a shared low-rank layer across all features.

## Installation
To use HADL, ensure you have the required dependencies installed:
```bash
pip install torch scipy numpy
```

## Model Architecture
The model consists of three key components:
1. **Preprocessing**
   - Mean normalization
   - Haar wavelet decomposition (if enabled)
   - DCT transformation (if enabled)
2. **Low-Rank Prediction Layer**
   - Learns a compressed representation of the input using low-rank matrix factorization.
3. **Postprocessing**
   - Restores the mean to the predicted output.

## Usage
### Model Initialization
```python
from hadl import HADL

model = HADL(seq_len=128, pred_len=24, features=10, rank=30, individual=False, enable_Haar=True, enable_DCT=True)
```

### Forward Pass
```python
import torch

x = torch.randn(32, 128, 10)  # [Batch, Input Length, Features]
out = model(x)
print(out.shape)  # Expected Output: [32, 24, 10]
```

## Components
### HADL Class
```python
class HADL(nn.Module):
    """
    Haar-DCT Low-Rank model for time series forecasting.
    """
    def __init__(self, seq_len, pred_len, features, rank=30, individual=False, bias=True, enable_Haar=True, enable_DCT=True):
        ...
```

### Low-Rank Layer
```python
class LowRank(nn.Module):
    """
    Low-Rank layer for dimensionality reduction.
    """
    def __init__(self, in_features, out_features, rank, bias=True):
        ...
```

### Discrete Cosine Transform
```python
class DiscreteCosineTransform(Function):
    """
    Differentiable Discrete Cosine Transform (DCT) and its inverse.
    """
    @staticmethod
    def forward(ctx, input):
        ...
```

## Configuration Options
| Parameter       | Description                                    | Default |
|---------------|--------------------------------|---------|
| `seq_len`     | Input sequence length                         | -       |
| `pred_len`    | Output prediction length                      | -       |
| `features`    | Number of input features                      | -       |
| `rank`        | Rank of the low-rank layer                    | 30      |
| `individual`  | If True, applies a separate layer per feature | False   |
| `bias`        | Enables bias in the low-rank layer            | True    |
| `enable_Haar` | Enables Haar decomposition                    | True    |
| `enable_DCT`  | Enables Discrete Cosine Transform             | True    |

## License
This project is open-source under the MIT License.
