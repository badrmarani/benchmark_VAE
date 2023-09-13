import torch
import torch.nn as nn


class LeakySiLU(nn.Module):
    """Differentiable analogue of LeakyReLU"""

    def __init__(self, negative_slope=0.01) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (1 - self.negative_slope) * F.silu(input) + self.negative_slope * input


class NeuralSampler(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        hidden_dim = embedding_dim * 4
        net = []
        dim = embedding_dim
        for i in range(4):
            net += [nn.Linear(dim, hidden_dim)]
            if i != 3:
                net += [nn.ReLU()]
            else:
                net += [nn.Linear(dim, embedding_dim)]
            dim = hidden_dim
        net += [nn.BatchNorm1d(embedding_dim, momentum=0.5, affine=False)]

        self.net = nn.Sequential(*net)

    def forward(self, batch_size: int):
        device = self.parameters().__next__().device

        eps = torch.randn(size=(batch_size, self.embedding_dim)).to(device)
        z = self.net(eps)
        return z
