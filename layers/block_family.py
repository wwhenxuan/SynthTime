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
from layers import QueryKeyProjection, RotaryProjection
from layers.projection import DepthWiseConv, Transpose
from utils import TimerMultivariateMask, TimerCovariateMask

from typing import Optional


class AttentionBias(nn.Module, abc.ABC):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    @abc.abstractmethod
    def forward(self, query_id, kv_id):
        ...


class BinaryAttentionBias(AttentionBias):
    """Moriai中使用的注意力机制的偏置"""

    def __init__(self, dim: int, num_heads: int):
        super().__init__(dim, num_heads)
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.num_heads)

    def forward(self, query_id, kv_id):
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        weight = rearrange(self.emb.weight, "two num_heads -> two num_heads 1 1")
        bias = ~ind * weight[:1] + ind * weight[1:]
        return bias


class TemporalAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
        d_model=512,
        n_heads: int = 8,
        max_len=100,  # 最大长度又影响什么
        covariate=False,  # 这个变量的具体作用是什么
        flash_attention=False,  # 什么是Flash Attention
    ):
        super(TemporalAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.covariate = covariate
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(
            dim=d_model,
            num_heads=n_heads,
            proj_layer=RotaryProjection,
            kwargs=dict(max_len=max_len),
            partial_factor=(0.0, 0.5),
        )
        self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=n_heads)

    def forward(
        self, queries, keys, values, attn_mask, n_vars, n_tokens, tau=None, delta=None
    ):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # [B, H, L, E]
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        seq_id = torch.arange(n_tokens * n_vars)
        seq_id = repeat(seq_id, "n -> b h n", b=B, h=H)

        queries, keys = self.qk_proj(queries, keys, query_id=seq_id, kv_id=seq_id)

        scale = self.scale or 1.0 / sqrt(E)

        var_id = repeat(torch.arange(n_vars), "C -> (C n_tokens)", n_tokens=n_tokens)
        var_id = repeat(var_id, "L -> b h L", b=B, h=1).to(queries.device)

        attn_bias = self.attn_bias(var_id, var_id)

        # TODO: 是否要将被掩码部分的token数值设置为0
        if self.mask_flag:
            if attn_mask is None:
                if self.covariate:
                    attn_mask = TimerCovariateMask(
                        B, n_vars, n_tokens, device=queries.device
                    )
                else:
                    attn_mask = TimerMultivariateMask(
                        B, n_vars, n_tokens, device=queries.device
                    )
            attn_mask = attn_bias.masked_fill(attn_mask.mask, float("-inf"))
        else:
            attn_mask = attn_bias

        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask
            )
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            scores += attn_mask

            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    """Timer中使用的具体的注意力机制"""

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        # 模型中使用的具体注意力机制
        self.inner_attention = attention
        """将输入从模型的维度投影到具体的多头注意力机制中"""
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        """整合多头注意力机制的输出将其重新投影到模型维度中"""
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries,
        keys,
        values,
        attn_mask,
        n_vars=None,
        n_tokens=None,
        tau=None,
        delta=None,
    ):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 将对应的queries, keys, values 映射到对应的维度上
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 在模型内部计算注意力机制
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            n_vars=n_vars,
            n_tokens=n_tokens,
            tau=tau,
            delta=delta,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


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
