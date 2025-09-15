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
