# -*- coding: utf-8 -*-
"""
Created on 2024/12/17 20:40:15
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import torch
from torch import nn
from layers import (
    get_activation,
    get_talking_heads,
    Normalization,
    TimeFreqConcatenate,
    TemporalAttention,
    FrequencyFilter,
    TimeFreqFusion,
)

from typing import Optional, Union, Tuple, List, Any


class TransformerEncoder(nn.Module):
    """一整个注意力机制模块"""

    def __init__(
        self,
        transformer_layer: Union[nn.Module, List[nn.Module]],
        d_model: int,  # 模型的维度
        n_layers: int,
        backbone_norm: Optional[str] = None,
        talking_heads: Optional[bool] = False,
    ) -> None:
        super(TransformerEncoder, self).__init__()
        """存放注意力机制的层，这里还有一个卷积模块"""
        self.transformer_layers = nn.ModuleList(transformer_layer)

        # TODO: 注意这个卷积层是用来做什么的？  先保留吧
        self.talking_heads = (
            get_talking_heads(d_model=d_model, n_layers=n_layers)
            if talking_heads
            else None
        )

        # 使用的标准化层，BatchNorm或是LayerNorm
        self.norm = (
            Normalization(num_features=d_model, norm=backbone_norm)
            if backbone_norm is not None
            else None
        )

    def forward(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        if self.talking_heads is not None:
            for i, (transformer_layer, talking_layer) in enumerate(
                zip(self.transformer_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x = transformer_layer(
                    x, n_vars, n_tokens, attn_mask=attn_mask, tau=tau, delta=delta
                )
                x = talking_layer(x)

            # TODO: 为什么要进行这一步
            # x = self.transformer_layers[-1](x, n_vars, n_tokens, tau=tau, delta=None)
        else:
            for transformer_layer in self.transformer_layers:
                x = transformer_layer(
                    x, n_vars, n_tokens, attn_mask=attn_mask, tau=tau, delta=delta
                )

        if self.norm is not None:
            x = self.norm(x)

        return x


class TransformerLayer(nn.Module):
    """Moriai使用的一层注意力机制"""

    def __init__(
        self,
        # TODO: 在时域中使用注意力机制，在频域中使用自适应滤波器
        time_attention: Union[nn.Module, TemporalAttention],  # 时域中使用的具体注意力机制
        frequency_filter: Union[nn.Module, FrequencyFilter],  # 频域中使用的滤波器模块
        feature_fusion: Union[nn.Module, TimeFreqFusion],  # 时频特征融合模块
        d_model: int,  # 模型的维度
        concatenate: Optional[str] = "alternate",  # 使用的时频特征合并模块
        dropout: float = 0.1,  # dropout的比例
        norm: Optional[str] = "BatchNorm",
        pre_norm: Optional[bool] = True,
    ) -> None:
        super(TransformerLayer, self).__init__()

        # 在时域特征学习中使用的注意力机制
        self.time_attention = time_attention

        # 在频域特征学习中使用的自适应滤波器
        self.frequency_filter = frequency_filter

        # 构建原本的ff模块
        # TODO: 这里直接通过外部传入

        # 构建时频域特征融合模块
        self.feature_fusion = feature_fusion

        # 用于时域频域token融合的模块
        self.concatenate = TimeFreqConcatenate(concatenate=concatenate)

        # 创建使用的标准化层
        self.norm = norm
        self.norm1 = Normalization(num_features=d_model, norm=norm)
        self.norm2 = Normalization(num_features=d_model, norm=norm)

        # 设置标准化的具体位置
        self.pre_norm = pre_norm

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, n_vars, n_tokens, attn_mask=None, tau=None, delta=None):
        """一层注意力机制的四部分内容：多头注意力机制，标准化，全连接层，标准化"""

        # 判断数据的基本形状
        batch_size, num_tokens, d_model = x.size()

        # 构造时频域双重的残差Token
        res_token = (
            torch.stack((x, x), dim=0)
            .permute(1, 2, 0, 3)
            .reshape(batch_size, -1, d_model)
        )

        if self.pre_norm:
            # 对数据进行标准化
            x = self.norm1(x)

            # 通过时域注意力机制来学习和提取模型在时域的基本特征
            time, attn = self.time_attention(
                x,
                x,
                x,
                n_vars=n_vars,
                n_tokens=n_tokens,
                attn_mask=attn_mask,
                tau=tau,
                delta=delta,
            )

            # 通过频域滤波模块来学习和提取模型在频域的基本特征
            freq = self.frequency_filter(x)

            # 对时频特征的token进行特征拼接
            concatenate_token = self.concatenate(time, freq)

            # 引入Transformer的残差连接
            x = res_token + self.dropout(concatenate_token)

            # 进行第二层标准化
            y = self.norm2(x)

            # 计算时频特征融合特征
            fusion_token = self.feature_fusion(y)

            # 构建残差连接模块
            return x + self.dropout(fusion_token)

        else:
            # 通过时域注意力机制来学习和提取模型在时域的基本特征
            time, attn = self.time_attention(
                x,
                x,
                x,
                n_vars=n_vars,
                n_tokens=n_tokens,
                attn_mask=attn_mask,
                tau=tau,
                delta=delta,
            )

            # 通过频域滤波模块来学习和提取模型在频域的基本特征
            freq = self.frequency_filter(x)

            # 对时频特征的token进行特征拼接
            concatenate_token = self.concatenate(time, freq)

            # 引入Transformer的残差连接
            x = res_token + self.dropout(concatenate_token)

            # 通过第一个标准化层
            y = self.norm1(x)

            # 进行时域和频域的特征融合
            fusion_token = self.feature_fusion(y)

            # 引入第二个残差连接
            outputs = y + self.dropout(fusion_token)

            # 返回本次传播特征
            return self.norm2(outputs)
