# -*- coding: utf-8 -*-
"""
Created on 2025/09/12 10:21:23
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan
"""
import torch
from torch import nn

from layers import TransformerLayer, TransformerEncoder, ResidualEmbedding


class Model(nn.Module):

    def __init__(self, configs) -> None:
        super(Model, self).__init__()

        # 待处理的下游任务名称
        self.task_name = configs.task_name

        # 划分Patch的长度
        self.patch_len = configs.patch_len

        # 模型的维度
        self.d_model, self.d_ff = configs.d_model, configs.d_ff
        self.n_heads = configs.n_heads

        # 模型的层数
        self.n_layers = configs.n_layers

        # 构建用于嵌入的线性层
        self.embedding = ResidualEmbedding(
            patch_len=self.patch_len,
            d_model=configs.d_model,
            hidden_features=configs.embed_hidden,
        )

        # 构建使用的Transformer模块
        self.backbone = TransformerEncoder()
