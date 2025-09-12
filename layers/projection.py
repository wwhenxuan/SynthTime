# -*- coding: utf-8 -*-
"""
Created on 2024/12/17 20:40:15
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import abc
from functools import cached_property
import torch
from einops import einsum, rearrange, repeat
from torch import nn

from typing import Optional, Union


class Transpose(nn.Module):
    """Transpose the dimensions of the input tensor"""

    def __init__(self, *dims, contiguous=False) -> None:
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def get_activation(activation: Union[nn.Module, str]):
    """选择使用的激活函数"""
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", or a callable'
    )


class RMSNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class Normalization(nn.Module):
    """使用的标准化层"""

    def __init__(self, num_features: int, norm: Union[nn.Module, str]):
        super(Normalization, self).__init__()

        self.num_features = num_features
        self.norm_name = norm.lower()

        if self.norm_name == "batchnorm":
            self.normalization = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(num_features), Transpose(1, 2)
            )
        elif self.norm_name == "layernorm":
            self.normalization = nn.LayerNorm(num_features)
        elif self.norm_name == "rmsnorm":
            self.normalization = RMSNorm(hidden_size=num_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """标准化层的正向传播部分"""
        return self.normalization(features)



class Projection(nn.Module, abc.ABC):
    def __init__(self, proj_width: int, num_heads: int, **kwargs):
        super().__init__()
        self.proj_width = proj_width
        self.num_heads = num_heads

    @abc.abstractmethod
    def forward(self, x, seq_id): ...


class RotaryProjection(Projection):
    """Moriai中使用的选转位置编码"""

    def __init__(
        self, *, proj_width: int, num_heads: int, max_len: int = 512, base: int = 10000
    ):
        super().__init__(proj_width, num_heads)
        assert (
            self.proj_width % 2 == 0
        ), f"proj_width must be even, got {self.proj_width}"
        self.register_buffer(
            "theta",
            1.0
            / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float)
                / self.proj_width,
            ),
            persistent=False,
        )
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_freq(max_len=max_len)

    def _init_freq(self, max_len: int):
        if self.cos is None or self.cos.size(-2) < max_len:
            position = torch.arange(
                max_len, device=self.theta.device, dtype=self.theta.dtype
            )
            m_theta = einsum(position, self.theta, "length, width -> length width")
            m_theta = repeat(m_theta, "length width -> length (width 2)")
            self.register_buffer("cos", torch.cos(m_theta), persistent=False)
            self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x):
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)  # noqa

    def forward(self, x, seq_id):
        self._init_freq(max_len=seq_id.max() + 1)
        rot_cos = self.cos[seq_id]
        rot_sin = self.sin[seq_id]
        return rot_cos * x + rot_sin * self._rotate(x)


class QueryKeyProjection(nn.Module):
    """Moriai使用的注意力机制"""

    def __init__(
        self, dim: int, num_heads: int, proj_layer, kwargs=None, partial_factor=None
    ):
        super().__init__()
        if partial_factor is not None:
            assert (
                0.0 <= partial_factor[0] < partial_factor[1] <= 1.0
            ), f"got {partial_factor[0]}, {partial_factor[1]}"
        assert num_heads > 0 and dim % num_heads == 0

        self.head_dim = dim // num_heads
        self.partial_factor = partial_factor
        # Attn 这里使用的是旋转位置编码
        self.query_proj = proj_layer(
            proj_width=self.proj_width,
            num_heads=num_heads,
            **(kwargs or {}),
        )
        self.key_proj = self.query_proj

    @cached_property
    def proj_width(self) -> int:
        if self.partial_factor is None:
            return self.head_dim
        return int(self.head_dim * (self.partial_factor[1] - self.partial_factor[0]))

    @cached_property
    def split_sizes(self):
        if self.partial_factor is None:
            return 0, self.head_dim, 0
        return (
            int(self.partial_factor[0] * self.head_dim),
            self.proj_width,
            int((1.0 - self.partial_factor[1]) * self.head_dim),
        )

    def forward(self, query, key, query_id, kv_id):
        if self.partial_factor is not None:
            queries = list(query.split(self.split_sizes, dim=-1))
            keys = list(key.split(self.split_sizes, dim=-1))
            queries[1] = self.query_proj(queries[1], seq_id=query_id)
            keys[1] = self.key_proj(keys[1], seq_id=kv_id)
            query = torch.cat(queries, dim=-1)
            key = torch.cat(keys, dim=-1)
        else:
            query = self.query_proj(query, seq_id=query_id)
            key = self.key_proj(key, seq_id=kv_id)
        return query, key


if __name__ == '__main__':
    x = torch.randn(size=(10, 64, 512))

    for norm in ["BatchNorm", "LayerNorm", "RMSNorm"]:
        layer = Normalization(num_features=512, norm=norm)
        print(layer(x).size())

