"""
PyTorch Neural Network Architecture Module for Tabular Churn Classification.
Designed specifically to balance predictive power, regularization against overfitting,
and gradient smooth differentiability for XAI feature attribution (Captum, SHAP, LIME).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularChurnNN(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for Tabular Binary Classification.
    
    Architecture Rationale:
    - Input Dimension: 47 features (one-hot encoded + scaled numericals)
    - Hidden Layer 1: Dense(47 -> 64) -> BatchNorm1d -> ReLU -> Dropout(0.25)
    - Hidden Layer 2: Dense(64 -> 32) -> BatchNorm1d -> ReLU -> Dropout(0.20)
    - Output Layer: Dense(32 -> 1) producing raw un-normalized Logits.
    
    Why Logits Output?
    1. Numerical Stability: Paired with nn.BCEWithLogitsLoss for stable log-sum-exp gradients.
    2. Captum Integration: Integrated Gradients attribution functions seamlessly on raw logits
       without saturation artifacts caused by sigmoid tails.
    """

    def __init__(
        self,
        input_dim: int = 47,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        output_dim: int = 1,
        dropout_rate: float = 0.25,
    ):
        super(TabularChurnNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate * 0.8)  # Slightly lower dropout on inner layer

        # Output Layer (Logits)
        self.out = nn.Linear(hidden_dim_2, output_dim)

        # Weight Initialization (He/Kaiming Normal for ReLU)
        self._init_weights()

    def _init_weights(self):
        """Kaiming Normal initialization for Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning un-normalized logits of shape (batch_size, 1).
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        logits = self.out(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference method returning binary class probabilities [P(No Churn), P(Churn)].
        Required for LIME, SHAP, and evaluation metrics.
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            logits = self.forward(x)
            p_churn = torch.sigmoid(logits)
            p_no_churn = 1.0 - p_churn
            return torch.cat([p_no_churn, p_churn], dim=1)


def get_model_summary(model: nn.Module, input_dim: int = 47) -> dict:
    """Calculates total trainable parameters and layer dimensions."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layers_info = []
    for name, module in model.named_children():
        layers_info.append(f"{name}: {module}")

    return {
        "model_class": model.__class__.__name__,
        "input_dim": input_dim,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "architecture": layers_info,
    }


if __name__ == "__main__":
    # Test instantiation and forward pass
    dummy_input = torch.randn(10, 47)
    model = TabularChurnNN(input_dim=47)
    logits = model(dummy_input)
    probs = model.predict_proba(dummy_input)

    summary = get_model_summary(model)
    print("=" * 60)
    print("        PYTORCH NEURAL NETWORK ARCHITECTURE AUDIT")
    print("=" * 60)
    print(f"Model Class:       {summary['model_class']}")
    print(f"Input Features:    {summary['input_dim']}")
    print(f"Total Parameters:  {summary['total_params']:,}")
    print(f"Trainable Params:  {summary['trainable_params']:,}")
    print(f"Forward Logits:    Shape {tuple(logits.shape)}")
    print(f"Predict Proba:     Shape {tuple(probs.shape)}")
    print("=" * 60)
