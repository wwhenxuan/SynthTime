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


class RotaryPositionalEmbeddings(nn.Module):
    """使用的旋转位置编码"""

    def __init__(self, d_model: int, base: int = 10_000) -> None:
        super().__init__()
        self.base = base
        self.d_model = d_model
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor) -> None:
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        seq_len = x.shape[0]

        theta = 1.0 / (
            self.base ** (torch.arange(0, self.d_model, 2).float() / self.d_model)
        ).to(
            x.device
        )  # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)

        seq_idx = (
            torch.arange(seq_len, device=x.device).float().to(x.device)
        )  # Position Index -> [0,1,2...seq-1]

        idx_theta = torch.einsum(
            "n,d->nd", seq_idx, theta
        )  # Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]

        idx_theta2 = torch.cat(
            [idx_theta, idx_theta], dim=1
        )  # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]

        self.cos_cached = idx_theta2.cos()[
            :, None, :
        ]  # Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
        self.sin_cached = idx_theta2.sin()[
            :, None, :
        ]  # cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

    def _neg_half(self, x: torch.Tensor) -> torch.Tensor:
        d_2 = self.d_model // 2  #

        return torch.cat(
            [-x[:, :, d_2:], x[:, :, :d_2]], dim=-1
        )  # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._build_cache(x)

        neg_half_x = self._neg_half(x)

        x_rope = (x * self.cos_cached[: x.shape[0]]) + (
            neg_half_x * self.sin_cached[: x.shape[0]]
        )  # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]

        return x_rope


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


class FlattenHeads(nn.Module):
    """Integrate the final output of the time series encoder"""

    def __init__(
        self,
        individual: bool,
        n_vars: int,
        nf: int,
        patch_num: int,
        targets_window: int,
        head_dropout: float = 0,
        cls_token: Optional[bool] = False,
    ) -> None:
        super().__init__()
        # Whether to output in a channel-independent manner
        self.individual = individual
        self.n_vars = n_vars
        self.patch_num = patch_num
        # Whether to take the [CLS] Token
        if cls_token is True:
            self.patch_num += 1

        if self.individual is True:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch_size, n_vars, d_model, patch_num]
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
    """用于多种不同下游任务的任务头网络"""

    def __init__(
        self,
        configs,
        task_name: str,
        d_model: int,
        patch_num: int,
        dropout: Optional[float] = 0.1,
    ) -> None:
        super(TaskHeads, self).__init__()

        self.task_name = task_name

        self.d_model = d_model
        self.patch_num = patch_num
        self.dropout = dropout
