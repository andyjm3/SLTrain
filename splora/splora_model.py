import torch
from torch import nn
import os
from typing import List
from dataclasses import dataclass
from .splora_linear import SpLoRaLinear
import json
from transformers import AutoModelForCausalLM, AutoConfig


@dataclass
class SpLoRaConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    sp_ratio: float
    target_modules: List[str]
    trainable_scaling: bool = False


class SpLoRaModel(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        target_modules,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        sp_ratio=0.01,
        trainable_scaling=False,
    ):
        if r < 0:
            raise ValueError("r must be nonnegative.")
        if sp_ratio <= 0 or sp_ratio >= 1:
            raise ValueError("sp_ratio must be between 0 and 1.")

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.sp_ratio = sp_ratio
        self.trainable_scaling = trainable_scaling
        self.parameterized_modules = []

        self._config = SpLoRaConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            sp_ratio=sp_ratio,
            target_modules=target_modules,
        )

        # patch methods
        self.forward = self.wrapped_model.forward

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print(f"Reparameterized module: {module_name}")
            new_module = SpLoRaLinear(
                module.in_features,
                module.out_features,
                r=self.r,
                sp_ratio=sp_ratio,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                trainable_scaling=self.trainable_scaling,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype
            )

            module.weight = None
            del module

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

        torch.cuda.empty_cache()

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def save_pretrained(self, path, max_shard_size='100GB'):
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "splora_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, "splora_config.json"), "r") as f:
            splora_config = json.load(f)

        config = AutoConfig.from_pretrained(path)

        base_model = AutoModelForCausalLM.from_config(config)

        if "trainable_scaling" not in splora_config:
            splora_config["trainable_scaling"] = False

        model = cls(base_model, **splora_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        model.wrapped_model.load_state_dict(state_dict, strict=True)
        return model








