# -*- coding: utf-8 -*-
"""
Created on 2024/12/17 20:40:15
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import abc
import torch
import torch.nn as nn
from math import sqrt

from einops import repeat, rearrange
from layers.projection import DepthWiseConv, Transpose
from utils import TimerMultivariateMask, TimerCovariateMask

from typing import Optional

# -*- coding: utf-8 -*-
"""
Created on 2024/9/16 17:22
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
from layers import Transpose, get_activation_fn
from layers import PositionalEmbedding
from typing import Optional, Union, Tuple


class TSTEncoder(nn.Module):
    """Time series encoder backbone of SymTime"""

    def __init__(
        self,
        patch_len: int = 16,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: int = None,
        d_v: int = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        store_attn: bool = False,
        pre_norm: bool = False,
        forward_layers: int = 6,
    ) -> None:
        super().__init__()
        self.forward_layers = forward_layers
        self.W_P = nn.Linear(patch_len, d_model)
        # Positional encoding
        self.pe = PositionalEmbedding(d_model=d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Create the [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_mask = nn.Parameter(torch.ones(1, 1).bool(), requires_grad=False)

        # Create the encoder layer of the model backbone
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=act,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for _ in range(n_layers)
            ]
        )

        # model params init
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """model params init through apply methods"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: Tensor,  # x: [batch_size, patch_num, patch_len]
        attn_mask: Optional[Tensor] = None,  # attn_mask: [batch, num_patch]
    ) -> Tensor:
        batch_size = x.size(0)

        # Input patching embedding
        x = self.W_P(x)  # x: [batch_size, patch_num, model_dim]

        # Add the [CLS] token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        # adjust the attn mask
        if attn_mask is not None:
            attn_mask = torch.cat(
                [self.cls_mask.expand(batch_size, -1), attn_mask], dim=1
            )

        # Add the positional embedding
        x = self.pe(x)
        x = self.dropout(x)  # x: [batch_size, patch_num, d_model]

        for mod in self.layers[: self.forward_layers]:
            x = mod(x, attn_mask=attn_mask)

        return x


class TSTEncoderLayer(nn.Module):
    """Patch-based Transformer module sublayer"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int = None,
        d_v: int = None,
        d_ff: int = 256,
        store_attn: int = False,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        bias: bool = True,
        activation: str = "gelu",
        pre_norm: bool = False,
    ) -> None:
        super(TSTEncoderLayer, self).__init__()

        assert (
            not d_model % n_heads
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        # If not specified, the number of heads is divided
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Create the multi-head attention
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout
        )

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        # use pre-norm or not
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.attn = None

    def forward(
        self, src: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor] or Tensor]:
        """Multi-Head attention sublayer"""

        # Whether to use pre-norm for attention layer
        if self.pre_norm:
            src = self.norm_attn(src)

        # Multi-Head attention
        src2, attn = self.self_attn(src, src, src, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn

        # Add: residual connection with residual dropout
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Whether to use pre-norm for ffn layer
        if self.pre_norm:
            src = self.norm_ffn(src)

        # Position-wise Feed-Forward
        src2 = self.ff(src)

        # Add: residual connection with residual dropout
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        return src


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism layer"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int = None,
        d_v: int = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.sdp_attn = _ScaledDotProductAttention(
            d_model, n_heads, attn_dropout=attn_dropout
        )

        # Project output
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout)
        )

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        bs = q.size(0)
        if k is None:
            k = q
        if v is None:
            v = q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(k).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(v).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Apply Scaled Dot-Product Attention (multiple heads)
        output, attn_weights = self.sdp_attn(q_s, k_s, v_s, attn_mask=attn_mask)

        # back to the original inputs dimensions
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        )
        output = self.to_out(output)

        return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        res_attention: bool = False,
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim**-0.5), requires_grad=False)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor, Tensor] or Tuple[Tensor, Tensor]]:
        """
        :param q: [batch_size, n_heads, num_token, d_k]
        :param k: [batch_size, n_heads, d_k, num_token]
        :param v: [batch_size, n_heads, num_token, d_k]
        :param attn_mask: [batch_size, n_heads, num_token]
        """

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale

        # Attention mask (optional)
        if (
            attn_mask is not None
        ):  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            attn_mask = rearrange(attn_mask, "b i -> b 1 i 1") * rearrange(
                attn_mask, "b i -> b 1 1 i"
            )
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class FrequencyFilter(nn.Module):
    """频域滤波模块"""

    def __init__(
        self, adaptive_filter: bool, d_model: int, norm: Optional[str] = "ortho"
    ) -> None:
        super().__init__()
        self.complex_weight_high = nn.Parameter(
            torch.randn(d_model, 2, dtype=torch.float32) * 0.02
        )
        self.complex_weight = nn.Parameter(
            torch.randn(d_model, 2, dtype=torch.float32) * 0.02
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


class TimeFreqFusion(nn.Module):
    """时域和频域的特征学习模块"""

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
        super(TimeFreqFusion, self).__init__()

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


class SEAttention(nn.Module):
    """1dSE通道注意力机制"""

    def __init__(self, channel: int, reduction: Optional[int] = 8) -> None:
        super(SEAttention, self).__init__()
        # 通过全局平均池化操作获取每个通道的重要性
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # 根据全局池化的结果计算每个通道的重要性
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def init_weights(self):
        # 初始化网络模块参数
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 获取数据的批量和通道数
        bitch_size, channels, _ = x.size()
        # 进行Squeeze挤压操作
        y = self.avg_pool(x).view(bitch_size, channels)
        # Excitation激励操作
        y = self.fc(y).view(bitch_size, channels, 1)
        return x * y.expand_as(x)
