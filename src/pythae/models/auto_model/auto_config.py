from pydantic.dataclasses import dataclass

from pythae.config import BaseConfig


@dataclass
class AutoConfig(BaseConfig):
    @classmethod
    def from_json_file(cls, json_path):
        """Creates a :class:`~pythae.config.BaseAEConfig` instance from a JSON config file. It
        builds automatically the correct config for any `pythae.models`.

        Args:
            json_path (str): The path to the json file containing all the parameters

        Returns:
            :class:`BaseAEConfig`: The created instance
        """

        config_dict = cls._dict_from_json(json_path)
        config_name = config_dict.pop("name")

        try:
            from ....pythae import models
            model_config = getattr(models, config_name).from_json_file(json_path)
        except AttributeError:
            raise NameError(
                "Cannot reload automatically the model configuration... "
                f"The model name in the `model_config.json may be corrupted. Got `{config_name}`"
            )

        return model_config
