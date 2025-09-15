# -*- coding: utf-8 -*-
"""
Created on 2025/09/12 10:21:23
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan
"""
import torch
from torch import nn

from layers import (
    TransformerLayer,
    TransformerEncoder,
    ResidualEmbedding,
    FlattenHeads,
    TemporalAttention,
    FrequencyFilter,
    TimeFreqFusion,
)


class Model(nn.Module):

    def __init__(self, configs) -> None:
        super(Model, self).__init__()

        # 待处理的下游任务名称
        self.task_name = configs.task_name

        # 输入时间序列数据的长度
        self.seq_len = configs.seq_len

        # 划分Patch的长度
        self.patch_len = configs.patch_len

        # 划分Patch单独步长
        self.stride = configs.stride

        # 这里计算能够构造Patch的数目
        self.patch_num = configs.seq_len // configs.patch_len * configs.num_vars

        # 模型的维度
        self.d_model, self.d_ff = configs.d_model, configs.d_ff
        self.n_heads = configs.n_heads

        # 模型的层数
        self.n_layers = configs.n_layers

        # 构建用于嵌入的线性层
        self.embedding = ResidualEmbedding(
            patch_len=configs.patch_len,
            d_model=configs.d_model,
            hidden_features=configs.embed_hidden,
        )

        # 构建使用的Transformer模块
        self.backbone = TransformerEncoder(
            transformer_layer=TransformerLayer(
                time_attention=TemporalAttention(d_model=configs.d_model,
                                                 n_heads=configs.n_heads,
                                                 attention_dropout=configs.attention_dropout,
                                                 max_len=configs.max_len,
                                                 scale=None, ),
                frequency_filter=FrequencyFilter(
                    adaptive_filter=configs.adaptive_filter,
                    d_model=configs.d_model,
                    norm=configs.rfft_norm,
                ),
                feature_fusion=TimeFreqFusion(
                    patch_num=configs.patch_num,
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    kernel_size=configs.kernel_size,
                    padding=configs.padding,
                    use_conv=configs.use_conv,
                    bias=configs.fusion_bias,
                    reduction=configs.reduction,
                    point_wise_bias=configs.point_wise_bias,
                ),
                d_model=configs.d_model,
                concatenate=configs.concatenate,
                dropout=configs.dropout,
                norm=configs.norm,
                pre_norm=configs.pre_norm,
            ),
            d_model=configs.d_model,
            n_layers=configs.n_layers,
            backbone_norm=configs.backbone_norm,  # TODO: 这两个是用于增强模型的性能的
            talking_heads=configs.talking_heads,
        )

        # TODO: 这里添加任务头
        self.task_layer = ...

    def patching(self, x: torch.Tensor) -> torch.Tensor:
        """将输入的多通道信号划分为不同的片段"""
        # x = x.permute(0, 2, 1)  # [batch_size, num_channels, seq_len]
        x = x.unfold(
            dimension=-1, size=self.patch_len, step=self.stride,
        )  # [batch_size, num_channels, num_vars, patch_len]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def pretrain(self, x: torch.Tensor) -> torch.Tensor:
        """用于模型预训练的接口配置"""

    def long_term_forecast(self, x: torch.Tensor) -> torch.Tensor:
        pass
