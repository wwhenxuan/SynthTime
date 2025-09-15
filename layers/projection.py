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

from typing import Optional, List, Union


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
    def forward(self, x, seq_id):
        ...


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


class TimeFreqConcatenate(nn.Module):
    """用于整合与拼接来自时域的基本特征"""

    def __init__(self, concatenate: Optional[str] = "alternate") -> None:
        super(TimeFreqConcatenate, self).__init__()

        # 根据输入选择具体的拼接方式
        if concatenate == "alternate":
            self.concat = self.alternate
        elif concatenate == "sequential":
            self.concat = self.sequential
        else:
            raise ValueError

    @staticmethod
    def sequential(time: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """按照顺序拼接来自时域和频域的Token"""
        return torch.concatenate([time, freq], dim=1)

    @staticmethod
    def alternate(time: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """交替拼接来自时域和频域的Token"""
        # 判断数据的基本形状
        batch_size, num_tokens, d_model = time.size()

        # 堆叠张量: [2, batch_size, num_token, d_model]
        stacked = torch.stack((time, freq), dim=0)

        # 交换维度: [batch_size, num_token, 2, d_model]
        stacked = stacked.permute(1, 2, 0, 3)

        # 合并维度: [batch_size, 2 * num_token, d_model]
        return stacked.reshape(batch_size, -1, d_model)

    def forward(self, time: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """将时域和频域中的特征Token进行对其拼接"""
        return self.concat(time=time, freq=freq)


class DepthWiseConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Optional[int] = 3,
        padding: Optional[int] = 1,
        bias: Optional[bool] = False,
    ) -> None:
        super(DepthWiseConv, self).__init__()

        self.depth_wise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.point_wise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depth_wise(x)
        # print("Depth_wise:", out.shape)
        out = self.point_wise(out)
        # print("Point_wise:", out.shape)

        return out


def get_talking_heads(
    d_model: int, n_layers: int
) -> Union[List[nn.Module], nn.ModuleList[nn.Module]]:
    """使用卷积层接收来自Transformer层的输出结果"""
    return nn.ModuleList(
        [
            nn.Sequential(
                Transpose(1, 2),
                nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                Transpose(1, 2),
            )
            for _ in range(n_layers)
        ]
    )


class Flatten_Heads(nn.Module):
    """Integrate the final output of the time series encoder"""

    def __init__(
        self,
        individual: bool,
        n_vars: int,
        nf: int,
        patch_num: int,
        targets_window: int,
        head_dropout: int = 0,
        cls_token: Optional[bool] = False,
    ) -> None:
        super().__init__()
        # Whether to output in a channel-independent manner
        self.individual = individual
        self.n_vars = n_vars
        self.patch_num = patch_num
        # Whether to take the [CLS] Token
        if cls_token:
            self.patch_num += 1

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.linears.append(nn.Linear(nf * self.patch_num, targets_window))
                self.dropouts.append(nn.Dropout(head_dropout))
                self.flattens.append(nn.Flatten(start_dim=-2))
        else:
            self.linear = nn.Linear(nf * self.patch_num, targets_window)
            self.dropout = nn.Dropout(head_dropout)
            self.flatten = nn.Flatten(start_dim=-2)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # [batch_size, n_vars, d_model, patch_num]
        if self.individual is True:
            x_out = []
            for i in range(self.n_vars):
                # 将某一通道的维数展平
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TaskHeads(nn.Module):
    def __init__(
        self,
        task_name: str,
        individual: bool,
        n_vars: int,
        nf: int,
        n_classes: int,
        patch_num: int,
        targets_window: int,
        head_dropout: int = 0,
        cls_token: Optional[bool] = False,
    ) -> None:
        super(TaskHeads, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


# if __name__ == '__main__':
#     x = torch.randn(size=(10, 64, 512))
#
#     for norm in ["BatchNorm", "LayerNorm", "RMSNorm"]:
#         layer = Normalization(num_features=512, norm=norm)
#         print(layer(x).size())

# if __name__ == '__main__':
#     batch_size, num_token, d_model = 2, 3, 4
#     concat = TimeFreqConcatenate("sequential")
#     tensor1 = torch.arange(batch_size * num_token * d_model).view(batch_size, num_token, d_model)
#     tensor2 = torch.ones(batch_size, num_token, d_model) * 100
#
#     print(concat(tensor1, tensor2))

# if __name__ == "__main__":
#     net = DepthWiseConv(3, 10, kernel_size=3, padding=1, bias=False)
#     print(net(torch.rand(size=(10, 3, 24))).size())
