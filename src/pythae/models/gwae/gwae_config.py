from dataclasses import field
from typing import List, Literal, Union

from pydantic.dataclasses import dataclass

from ..vae import VAEConfig


@dataclass
class GWAEConfig(VAEConfig):
    distance_coef: float = 1.0
    learned_similarity: bool = True
    merged_condition: bool = True
    mixed_potential: bool = True

    mmd_kernel_choice: Literal["rbf", "imq"] = "rbf"
    mmd_kernel_bandwidth: float = 1.0
    mmd_scales: Union[List[float], None] = field(
        default_factory=lambda: [0.1, 0.2, 0.5, 1.0, 2.0, 5, 10.0]
    )

    coef_w: float = 1.0
    coef_d: float = 1.0
    coef_entropy_reg: float = 1.0
    coef_gradient_penalty: float = 10.0

    uses_default_discriminator: bool = True
    discriminator_input_dim: List = None
    max_epochs_discriminator: int = 5
