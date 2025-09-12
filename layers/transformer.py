# -*- coding: utf-8 -*-
"""
Created on 2024/12/17 20:40:15
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import torch
from torch import nn
from layers import get_activation, Normalization, SEAttention, DepthWiseConv
from layers.projection import Transpose

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
        # TODO: 在时域中使用注意力机制，在频域中使用自适应滤波器
        time_attention: nn.Module,  # 时域中使用的具体注意力机制
        frequency_filter: nn.Module,  # 频域中使用的滤波器模块
        feature_fusion: nn.Module,  # 时频特征融合模块
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

        # 在时域特征学习中使用的注意力机制
        self.time_attention = time_attention

        # 在频域特征学习中使用的自适应滤波器
        self.frequency_filter = frequency_filter

        # 构建原本的ff模块
        # TODO: 这里直接通过外部传入

        # 创建使用的标准化层
        self.norm = norm
        self.norm1 = Normalization(num_features=d_model, norm=norm)
        self.norm2 = Normalization(num_features=d_model, norm=norm)

        # 设置标准化的具体位置
        self.pre_norm = pre_norm

        self.dropout = nn.Dropout(dropout)

    def forward_pre_norm(
        self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None
    ):
        """"""

    def forward(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        """一层注意力机制的四部分内容：多头注意力机制，标准化，全连接层，标准化"""

        if self.pre_norm:
            pass

        else:
            # 1.这里通过具体的注意力机制 获得输出的值和注意力分数
            new_x, attn = self.time_attention(
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

        new_x, attn = self.time_attention(
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


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, adaptive_filter: bool, dim: int) -> None:
        super().__init__()
        self.complex_weight_high = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02
        )
        self.complex_weight = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02
        )

        # 使用截断正态分布
        nn.init.trunc_normal_(self.complex_weight_high, std=0.02)
        nn.init.trunc_normal_(self.complex_weight, std=0.02)

        self.threshold_param = nn.Parameter(torch.rand(1))  # * 0.5)

        self.adaptive_filter = adaptive_filter

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(
            B, 1
        )  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        adaptive_mask = (
            (normalized_energy > self.threshold_param).float() - self.threshold_param
        ).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm="ortho")

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class TimeFreq(nn.Module):
    def __init__(
        self,
        patch_num: int,
        d_model: int,
        d_ff: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_conv: Optional[int] = True,
        bias: Optional[bool] = False,
        reduction: Optional[int] = 8,
        point_wise_bias: Optional[bool] = False,
    ) -> None:
        super(TimeFreq, self).__init__()

        self.patch_num = patch_num
        self.token_num = patch_num * 2

        if use_conv:
            self.body = nn.Sequential(
                Transpose(1, 2),
                DepthWiseConv(
                    in_channels=d_model,
                    out_channels=d_ff,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=bias,
                ),
                SEAttention(channel=d_ff, reduction=reduction),
                DepthWiseConv(
                    in_channels=d_ff,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=bias,
                ),
                Transpose(1, 2),
            )
        else:
            self.body = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff, bias=bias),
                SEAttention(channel=d_ff, reduction=reduction),
                nn.Linear(in_features=d_ff, out_features=d_model, bias=bias),
            )

        self.conv1x1 = nn.Conv1d(
            in_channels=self.token_num,
            out_channels=self.patch_num,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=point_wise_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """特征融合模块的正向传播部分"""
        batch_size, token_num, d_model = x.size()

        assert token_num == self.token_num

        # 将时频混合的token输入到注意力卷积模块中进行特征学习
        time_freq_maps = self.body(x)

        # 通过1x1卷积对时频特征进行融合并输出
        return self.conv1x1(time_freq_maps)
