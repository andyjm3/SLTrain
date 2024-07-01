import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
import sltrain_linear


class lora_sparse_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lora_B, lora_A, dv, di, bias):
        ctx.save_for_backward(input, lora_B, lora_A, dv, di, bias)

        return sltrain_linear.forward(input, lora_B, lora_A, dv, di, bias)

    @staticmethod
    def backward(ctx, output_grad):
        input, lora_B, lora_A, dv, di, bias = ctx.saved_tensors

        grads = sltrain_linear.backward(
            output_grad, input, lora_B, lora_A, dv, di,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[3],
            bias is not None and ctx.needs_input_grad[5],
            bias,
        )

        return tuple(grads)


class SpLoRaLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int,
            sp_ratio: float = 0.01,
            *,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            trainable_scaling: bool = False,
            bias=True,
            device=None,
            dtype=None,
    ):
        """
        Reparameterized sparse and low rank linear layer
                    x W_a @ W_b * lora_alpha / r + x W_sp + bias
        Notice that scale = lora_alpha / r.
        Notice that this class cannot be wrapped to linear layer and thus cannot be used for fine-tune
        """
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive.")
        if sp_ratio <= 0 or sp_ratio >= 1:
            raise ValueError("sp_ratio must be between 0 and 1.")

        if bias:
            self.bias = Parameter(torch.zeros(out_features, device=device, dtype=dtype, requires_grad=True))
            a = 1/math.sqrt(out_features)
            nn.init.uniform_(self.bias, -a, a)
        else:
            self.register_parameter('bias', None)

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.trainable_scaling = trainable_scaling
        self.sp_ratio = sp_ratio
        self.device = device
        self.dtype = dtype

        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.empty(out_features, r, dtype=dtype, device=device))
        nn.init.zeros_(self.lora_B)
        if trainable_scaling:
            self.scaling = nn.Parameter(torch.tensor([1.], device=device, dtype=dtype), requires_grad=True)
        else:
            self.scaling = self.lora_alpha / self.r

        indices, values, shape = self._init_sparse_parameters()
        self.shape = shape
        self.register_buffer("sparse_index", indices.to(device))
        self.sparse_value = Parameter(values.to(device), requires_grad=True)



    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()

        return self.scaling

    def _init_sparse_parameters(self):
        shape = [self.out_features, self.in_features]
        total_elements = self.in_features * self.out_features
        num_nonzeros = int(self.sp_ratio * total_elements)

        indices = torch.randperm(total_elements)[:num_nonzeros]
        indices, _ = torch.sort(indices)
        indices.to(self.device)

        values = torch.empty(size=(num_nonzeros,), device=self.device, dtype=self.dtype)
        a = 1/math.sqrt(self.in_features)
        nn.init.uniform_(values, -a, a)

        return indices, values, shape


    def forward(self, x: Tensor) :
        return lora_sparse_linear.apply(x, self.lora_B, self.lora_A * self._post_lora_scale(), self.sparse_value,
                             self.sparse_index, self.bias)


    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, rank={self.r}, '
                f'sparsity={self.sp_ratio}, bias={self.bias is not None}')