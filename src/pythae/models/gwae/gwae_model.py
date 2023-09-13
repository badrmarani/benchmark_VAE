import inspect
import logging
from copy import deepcopy
from typing import Optional

import cloudpickle
import torch
from torch import nn

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseDiscriminator, BaseEncoder
from ..base.base_utils import CPU_Unpickler, ModelOutput, hf_hub_is_available
from ..nn.default_architectures import Discriminator_GVAE_MLP
from ..vae import VAE
from .gwae_config import GWAEConfig
from .gwae_utils import NeuralSampler

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class GWAE(VAE):
    """Gromov-Wasserstein Autoencoders
    
    Args:
    """
    def __init__(
        self,
        model_config: GWAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        discriminator: Optional[BaseDiscriminator] = None,
    ):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "GWAE"

        self.coef_w = self.model_config.coef_w
        self.coef_d = self.model_config.coef_d
        self.coef_entropy_reg = self.model_config.coef_entropy_reg

        self.mixed_potential = model_config.mixed_potential
        self.learned_similarity = model_config.learned_similarity
        self.merged_condition = model_config.merged_condition

        if self.merged_condition or self.learned_similarity or self.mixed_potential:
            if discriminator is None:
                # print(model_config.input_dim)
                # print(model_config.latent_dim)
                if model_config.input_dim is None or model_config.latent_dim is None:
                    raise AttributeError(
                        "No input and/or latent dimension provided !"
                        "'latent_dim' parameter and 'input_dim' parameter of GWAE_Config instance "
                        "must be set to a value. Unable to build discriminator automatically."
                    )
                self.model_config.discriminator_input_dim = self.model_config.latent_dim

                self.model_config.discriminator_input_dim = (
                    self.model_config.input_dim,
                    self.model_config.latent_dim,
                )

                discriminator = Discriminator_GVAE_MLP(self.model_config)
                self.model_config.uses_default_discriminator = True
            else:
                self.model_config.uses_default_discriminator = False

            self.set_discriminator(discriminator)
            self.max_epochs_discriminator = self.model_config.max_epochs_discriminator
            self.coef_gradient_penalty = self.model_config.coef_gradient_penalty

        device = self.encoder.parameters().__next__().device
        self.sampler = NeuralSampler(self.model_config.latent_dim).to(device=device)

        self.mmd_scales = self.model_config.mmd_scales
        self.mmd_kernel_bandwidth = self.model_config.mmd_kernel_bandwidth

        self.log_distance_coef = nn.Parameter(
            torch.tensor(self.model_config.distance_coef).log()
        )

    def set_discriminator(self, discriminator: BaseDiscriminator) -> None:
        r"""This method is called to set the discriminator network

        Args:
            discriminator (BaseDiscriminator): The discriminator module that needs to be set to the
                model.

        """
        if not issubclass(type(discriminator), BaseDiscriminator):
            raise BadInheritanceError(
                (
                    "Discriminator must inherit from BaseDiscriminator class from "
                    "pythae.models.base_architectures.BaseDiscriminator. Refer to documentation."
                )
            )

        self.discriminator = discriminator

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]

        batch_size = x.size(0)

        z2 = self.sample_from_prior(batch_size)
        recon_x2 = self.decoder(z2)["reconstruction"]

        pwise_distance_x = (recon_x2.unsqueeze(1) - recon_x2.unsqueeze(0)).view(
            batch_size, batch_size, -1
        )

        scale_z2 = z2.std(dim=0).mean()
        scale_z2.clamp_(min=1e-8)
        pwise_distance_z = (z2.unsqueeze(1) - z2.unsqueeze(0)) * (
            self.log_distance_coef.exp() / scale_z2
        )

        pwise_distance_x = pwise_distance_x.norm(p=2, dim=2)
        pwise_distance_z = pwise_distance_z.norm(p=2, dim=2)

        loss_gw = torch.abs(pwise_distance_x - pwise_distance_z).mean()

        loss_w = (0.5 * (x - recon_x) ** 2).view(batch_size, -1).sum() / batch_size
        if self.learned_similarity:
            disc_x = self.discriminator.x_network(x)
            disc_recon_x = self.discriminator.x_network(recon_x)
            loss_w += (0.5 * (disc_x - disc_recon_x) ** 2).sum() / batch_size

        loss_entropy_reg = log_var.sum() / batch_size

        if self.merged_condition:
            logit_autoencoding = self.discriminator(x, z)
            if self.mixed_potential:
                zq = z.detach()
                xq = self.decoder(zq)["reconstruction"]
                logit_sampling = (
                    self.discriminator(recon_x2, z2) + self.discriminator(xq, zq)
                ) / 2
            else:
                logit_sampling = self.discriminator(recon_x2, z2)
            loss_d = (logit_sampling - logit_autoencoding).mean()
        else:
            loss_d = self.mmd(z, z2)

        autoencoder_loss = (
            loss_gw
            + self.coef_w * loss_w
            + self.coef_d * loss_d
            + self.coef_entropy_reg * loss_entropy_reg
        )

        # discriminator training
        if self.merged_condition:
            encoder_output = self.encoder(x)
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            for _ in range(self.max_epochs_discriminator if self.training else 1):
                with torch.no_grad():
                    z, eps = self._sample_gauss(mu, std)
                    recon_x = self.decoder(z)["reconstruction"]

                    z2 = self.sample_from_prior(batch_size)
                    recon_x2 = self.decoder(z2)["reconstruction"]

                # z = z.detach()
                # recon_x = recon_x.detach()
                # z2 = z2.detach()
                # recon_x2 = recon_x2.detach()



                logit_autoencoding = self.discriminator(x, z)
                if self.mixed_potential:
                    logit_sampling = 0.5 * (
                        self.discriminator(recon_x2, z2)
                        + self.discriminator(recon_x, z)
                    )
                else:
                    logit_sampling = self.discriminator(recon_x2, z2)

                loss_logits = - (logit_sampling - logit_autoencoding).mean()

                loss_gp = (
                    self.gradient_penalty_one_centered(
                        x, z, recon_x2, z2
                    )
                    + 1e-4 * logit_autoencoding.pow(2).mean()
                    + 1e-4 * logit_sampling.pow(2).mean()
                )

        discriminator_loss = loss_logits + self.coef_gradient_penalty * loss_gp

        loss = autoencoder_loss + discriminator_loss

        output = ModelOutput(
            loss=loss,
            autoencoder_loss=autoencoder_loss,
            discriminator_loss=discriminator_loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def logit_gradient(self, x, z, logit):
        batch_size = x.size(0)

        grad = torch.autograd.grad(
            outputs=logit,
            inputs=[x, z],
            grad_outputs=torch.ones_like(logit),
            retain_graph=True,
            create_graph=True,
        )
        grad_x = grad[0].view(batch_size, -1)
        grad_z = grad[1].view(batch_size, -1)
        grad_cat = torch.cat([grad_x, grad_z], dim=1)
        return grad_cat

    def gradient_penalty_one_centered(self, x1, z1, x2, z2):
        batch_size = x1.size(0)

        eps = torch.rand(size=(batch_size,), device=x1.device, dtype=x1.dtype)

        xp = (
            x1
            + eps[(...,) + (None,) * (len(x1.size()) - 1)]
            + x2 * (1 - eps[(...,) + (None,) * (len(x2.size()) - 1)])
        )
        zp = (
            z1
            + eps[(...,) + (None,) * (len(z1.size()) - 1)]
            + z2 * (1 - eps[(...,) + (None,) * (len(z2.size()) - 1)])
        )

        xp.requires_grad_(True)
        zp.requires_grad_(True)

        logit_interpolation = self.discriminator(xp, zp)
        grad_cat = self.logit_gradient(xp, zp, logit_interpolation)
        grad_norm = grad_cat.norm(p=2, dim=1)
        loss_gp = torch.mean((grad_norm - 1) ** 2)
        return loss_gp

    def mmd(self, z1, z2):
        N = z1.size(0)
        assert N == z2.size(0)

        if self.mmd_kernel_choice == "rbf":
            k_z1 = self.rbf_kernel(z1, z1)
            k_z2 = self.rbf_kernel(z2, z2)
            k_z1z2 = self.rbf_kernel(z1, z2)

        else:
            k_z1 = self.imq_kernel(z1, z1)
            k_z2 = self.imq_kernel(z2, z2)
            k_z12 = self.imq_kernel(z1, z2)

        mmd_z1 = (k_z1 - k_z1.diag().diag()).sum() / ((N - 1) * N)
        mmd_z2 = (k_z2 - k_z2.diag().diag()).sum() / ((N - 1) * N)
        mmd_z12 = k_z12.sum() / (N**2)

        mmd_loss = mmd_z1 + mmd_z2 - 2 * mmd_z12
        return mmd_loss

    def imq_kernel(self, z1, z2):
        """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""

        Cbase = (
            2.0 * self.model_config.latent_dim * self.model_config.kernel_bandwidth**2
        )

        k = 0

        for scale in self.mmd_scales:
            C = scale * Cbase
            k += C / (C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2)

        return k

    def rbf_kernel(self, z1, z2):
        """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""

        C = (
            2.0
            * self.model_config.latent_dim
            * self.model_config.mmd_kernel_bandwidth**2
        )

        k = torch.exp(-torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2 / C)

        return k

    def sample_from_prior(self, batch_size: int) -> torch.Tensor:
        return self.sampler(batch_size)

    def autoencoder_loss_function(self):
        pass

    def discriminator_loss_function(self):
        pass


    def save(self, dir_path: str):
        """Method to save the model at a specific location

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """

        # This creates the dir if not available
        super().save(dir_path)
        model_path = dir_path

        model_dict = {"model_state_dict": deepcopy(self.state_dict())}

        if not self.model_config.uses_default_discriminator:
            with open(os.path.join(model_path, "discriminator.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(
                    inspect.getmodule(self.discriminator)
                )
                cloudpickle.dump(self.discriminator, fp)

        torch.save(model_dict, os.path.join(model_path, "model.pt"))

    @classmethod
    def _load_custom_discriminator_from_folder(cls, dir_path):

        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)

        if "discriminator.pkl" not in file_list:
            raise FileNotFoundError(
                f"Missing discriminator pkl file ('discriminator.pkl') in"
                f"{dir_path}... This file is needed to rebuild custom discriminators."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, "discriminator.pkl"), "rb") as fp:
                discriminator = CPU_Unpickler(fp).load()

        return discriminator

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided

        """

        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)

        if not model_config.uses_default_encoder:
            encoder = cls._load_custom_encoder_from_folder(dir_path)

        else:
            encoder = None

        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)

        else:
            decoder = None

        if not model_config.uses_default_discriminator:
            discriminator = cls._load_custom_discriminator_from_folder(dir_path)

        else:
            discriminator = None

        model = cls(
            model_config, encoder=encoder, decoder=decoder, discriminator=discriminator
        )
        model.load_state_dict(model_weights)

        return model

    @classmethod
    def load_from_hf_hub(
        cls, hf_hub_path: str, allow_pickle: bool = False
    ):  # pragma: no cover
        """Class method to be used to load a pretrained model from the Hugging Face hub

        Args:
            hf_hub_path (str): The path where the model should have been be saved on the
                hugginface hub.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl`` and ``discriminator``) if a custom encoder (resp. decoder and/or
                discriminator) was provided
        """

        if not hf_hub_is_available():
            raise ModuleNotFoundError(
                "`huggingface_hub` package must be installed to load models from the HF hub. "
                "Run `python -m pip install huggingface_hub` and log in to your account with "
                "`huggingface-cli login`."
            )

        else:
            from huggingface_hub import hf_hub_download

        logger.info(f"Downloading {cls.__name__} files for rebuilding...")

        config_path = hf_hub_download(repo_id=hf_hub_path, filename="model_config.json")
        dir_path = os.path.dirname(config_path)

        _ = hf_hub_download(repo_id=hf_hub_path, filename="model.pt")

        model_config = cls._load_model_config_from_folder(dir_path)

        if (
            cls.__name__ + "Config" != model_config.name
            and cls.__name__ + "_Config" != model_config.name
        ):
            warnings.warn(
                f"You are trying to load a "
                f"`{ cls.__name__}` while a "
                f"`{model_config.name}` is given."
            )

        model_weights = cls._load_model_weights_from_folder(dir_path)

        if (
            not model_config.uses_default_encoder
            or not model_config.uses_default_decoder
            or not model_config.uses_default_discriminator
        ) and not allow_pickle:
            warnings.warn(
                "You are about to download pickled files from the HF hub that may have "
                "been created by a third party and so could potentially harm your computer. If you "
                "are sure that you want to download them set `allow_pickle=true`."
            )

        else:

            if not model_config.uses_default_encoder:
                _ = hf_hub_download(repo_id=hf_hub_path, filename="encoder.pkl")
                encoder = cls._load_custom_encoder_from_folder(dir_path)

            else:
                encoder = None

            if not model_config.uses_default_decoder:
                _ = hf_hub_download(repo_id=hf_hub_path, filename="decoder.pkl")
                decoder = cls._load_custom_decoder_from_folder(dir_path)

            else:
                decoder = None

            if not model_config.uses_default_discriminator:
                _ = hf_hub_download(repo_id=hf_hub_path, filename="discriminator.pkl")
                discriminator = cls._load_custom_discriminator_from_folder(dir_path)

            else:
                discriminator = None

            logger.info(f"Successfully downloaded {cls.__name__} model!")

            model = cls(
                model_config,
                encoder=encoder,
                decoder=decoder,
                discriminator=discriminator,
            )
            model.load_state_dict(model_weights)

            return model
