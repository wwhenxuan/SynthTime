# -*- coding: utf-8 -*-
"""
Created on 2024/12/17 20:40:15
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import torch
from torch import nn

from layers import get_activation

from typing import Optional, Union, Tuple, List


class ResidualEmbedding(nn.Module):
    """基于残差连接方法构建时间序列Patch的嵌入模块"""

    def __init__(
        self,
        patch_len: int,
        d_model: int,
        hidden_features: Optional[int] = 256,
        bias: Optional[bool] = True,
        activation: Optional[str] = "relu",
    ) -> None:
        super(ResidualEmbedding, self).__init__()

        self.hidden_layer = nn.Linear(
            in_features=patch_len, out_features=hidden_features, bias=bias
        )
        self.output_layer = nn.Linear(
            in_features=hidden_features, out_features=d_model, bias=bias
        )
        self.residual_layer = nn.Linear(
            in_features=patch_len, out_features=d_model, bias=bias
        )

        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差嵌入模块的正向传播部分"""
        hid = self.activation(self.hidden_layer(x))
        out = self.output_layer(hid)
        res = self.residual_layer(x)
        out = out + res
        return out
