# -*- coding: utf-8 -*-
"""
Created on 2024/12/17 20:40:15
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import torch
from torch import nn
from layers import get_activation, Normalization

from typing import Optional, Tuple, List, Any


class TimeEncoder(nn.Module):
    """一整个注意力机制模块"""

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(TimeEncoder, self).__init__()
        """存放注意力机制的层，这里还有一个卷积模块"""
        self.attn_layers = nn.ModuleList(attn_layers)
        # TODO: 注意这个卷积层是用来做什么的？
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        # 使用的标准化层，BatchNorm或是LayerNorm
        self.norm = norm_layer

    def forward(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []  # 存放注意力分数的列表
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, n_vars, n_tokens, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(
                    x, n_vars, n_tokens, attn_mask=attn_mask, tau=tau, delta=delta
                )
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class TimeLayer(nn.Module):
    """Moriai使用的一层注意力机制"""

    def __init__(
        self,
        attention: nn.Module,  # 使用的具体注意力机制
        d_model: int,  # 模型的维度
        d_ff: int = None,  # 全连接/卷积网络的维度
        dropout: float = 0.1,  # dropout的比例
        activation: str = "gelu",  # 使用的激活函数
        norm: Optional[str] = "BatchNorm",
        pre_norm: Optional[bool] = True,
        use_conv: bool = True,
    ) -> None:
        super(TimeLayer, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.attention = attention

        """Moriai中使用了两层point wise卷积"""
        self.use_conv = use_conv

        # TODO: 这两个模块后续也要调整
        if use_conv is True:
            self.ff = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1),
                get_activation(activation),
                nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1),
            )
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                get_activation(activation),
                nn.Linear(d_ff, d_model),
            )

        # 创建使用的标准化层
        self.norm = norm
        self.norm1 = Normalization(num_features=d_model, norm=norm)
        self.norm2 = Normalization(num_features=d_model, norm=norm)

        # 设置标准化的具体位置
        self.pre_norm = pre_norm

        self.dropout = nn.Dropout(dropout)

    def forward_pre_norm(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        """"""


    def forward(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        """一层注意力机制的四部分内容：多头注意力机制，标准化，全连接层，标准化"""
        # 1.这里通过具体的注意力机制 获得输出的值和注意力分数
        new_x, attn = self.attention(
            x,
            x,
            x,
            n_vars=n_vars,
            n_tokens=n_tokens,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta,
        )
        # 通过注意力机制引入的残差dropout
        x = x + self.dropout(new_x)

        # 2.通过标准化层
        y = x = self.norm1(x)
        # 3.通过全连接层 但这里给换成了两层卷积
        if self.use_conv is True:
            # 通过卷积网络
            y = self.ff(y.transpose(-1, 1)).transpose(-1, 1)
        else:
            # 通过全连接网络
            y = self.ff(y)

        # 4.再次通过标准化层
        return self.norm2(x + y), attn
