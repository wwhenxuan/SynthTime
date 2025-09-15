# -*- coding: utf-8 -*-
"""
Created on 2025/09/12 10:21:23
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan
"""
import numpy as np
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
                time_attention=TemporalAttention(
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    attention_dropout=configs.attention_dropout,
                    max_len=configs.max_len,
                    scale=None,
                ),
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
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            # The long- and short-term time series forecasting tasks
            self.flatten_head = FlattenHeads(
                individual=self.individual,
                n_vars=configs.enc_in,
                patch_num=self.patch_num,
                nf=self.d_model,
                targets_window=configs.pred_len,
                head_dropout=self.dropout,
                cls_token=False,
            )
        elif self.task_name == "classification":
            # We use the conv and linear for the classification heads
            if configs.classification_conv1d is True:
                configs.enc_in = configs.out_channels

            self.classifier = nn.Sequential(
                nn.Conv1d(
                    in_channels=configs.enc_in,
                    out_channels=configs.out_channels,
                    kernel_size=(3,),
                    stride=(1,),
                    padding=1,
                )
                if configs.classification_conv1d is True
                else nn.Identity(),
                nn.GELU(),
                nn.LayerNorm(self.d_model * (self.patch_num * configs.enc_in + 1)),
                nn.Linear(
                    in_features=self.d_model * (self.patch_num * configs.enc_in + 1),
                    out_features=configs.num_classes,
                ),
            )

        elif self.task_name == "imputation":
            self.flatten_head = FlattenHeads(
                individual=self.individual,
                n_vars=configs.enc_in,
                patch_num=self.patch_num,
                nf=self.d_model,
                targets_window=self.seq_len,
                head_dropout=self.out_dropout,
                cls_token=False,
            )

        elif self.task_name == "anomaly_detection":
            self.flatten_head = FlattenHeads(
                individual=self.individual,
                n_vars=configs.enc_in,
                patch_num=self.patch_num,
                nf=self.d_model,
                targets_window=self.seq_len,
                head_dropout=self.out_dropout,
                cls_token=False,
            )

        else:
            raise ValueError("task name wrong!")

    def patching(self, x: torch.Tensor) -> torch.Tensor:
        """将输入的多通道信号划分为不同的片段"""
        # x = x.permute(0, 2, 1)  # [batch_size, num_channels, seq_len]
        x = x.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride,
        )  # [batch_size, num_channels, num_vars, patch_len]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def pretrain(self, x: torch.Tensor) -> torch.Tensor:
        """用于模型预训练的接口配置"""

    def forcast(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Forward for long and short term forecasting"""
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # time series decompsition
        if self.use_avg is True:
            seasonal_part, trend_part = self.decompsition(x_enc)
            x_enc = seasonal_part.permute(0, 2, 1)
            # Mapping trend part to target length
            trend_part = trend_part.permute(0, 2, 1)
            trend_part = self.projection_trend(trend_part)
            trend_part = trend_part.permute(0, 2, 1)
        else:
            x_enc = x_enc.permute(0, 2, 1)

        # do patching
        x_enc = self.patching(ts=x_enc)  # [batch_size, num_vars, patch_num, patch_len]
        batch_size, num_vars, patch_num, patch_len = x_enc.size()

        x_enc = torch.reshape(x_enc, [batch_size * num_vars, patch_num, patch_len])
        x_dec = self.time_encoder(x_enc)
        # 从通道独立恢复为原来的输入形式
        x_dec = torch.reshape(
            x_dec, [batch_size, num_vars, x_dec.shape[-2], x_dec.shape[-1]]
        )
        x_dec = self.flatten_head(x_dec).permute(
            0, 2, 1
        )  # [batch_size, pred_len, num_vars]

        # add the trend part of the decompsition
        if self.use_avg is True:
            x_dec = x_dec + trend_part

        # De-Normalization from Non-stationary Transformer
        x_dec = x_dec * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        x_dec = x_dec + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return x_dec

    def classification(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Forward for classification task"""
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = x_enc.permute(0, 2, 1)  # [batch_size, num_vars, seq_len]

        # Adjusting the input channels through Conv1d
        if self.use_conv1d is True:
            x_enc = self.conv1d(x_enc)  # [batch_size, out_channels, seq_len]
        # do patching and reshape
        x_enc = self.patching(ts=x_enc)  # [batch_size, num_vars, patch_num, patch_len]
        batch_size, num_vars, patch_num, patch_len = x_enc.size()
        # Learning feature through the backbone of Transformer
        x_enc = torch.reshape(
            x_enc, shape=(batch_size, num_vars * patch_num, patch_len)
        )
        x_dec = self.time_encoder(x_enc)
        # Output processing
        x_dec = self.act(x_dec)
        x_dec = torch.reshape(x_dec, shape=(batch_size, -1))
        x_dec = self.ln_proj(x_dec)
        outputs = self.classifier(x_dec)
        return outputs

    def imputation(self, x_enc: torch.Tensor) -> torch.Tensor:
        """进行时间序列填补任务的接口"""
        #  pre-interpolation from Peri-midFormer
        x_enc_np = x_enc.detach().cpu().numpy()
        zero_indices = np.where(x_enc_np[:, :, :] == 0)
        interpolated_x_enc = np.copy(x_enc_np)
        for sample_idx, time_idx, channel_idx in zip(*zero_indices):
            non_zero_indices = np.nonzero(x_enc_np[sample_idx, :, channel_idx])[0]
            before_non_zero_idx = (
                non_zero_indices[non_zero_indices < time_idx][-1]
                if len(non_zero_indices[non_zero_indices < time_idx]) > 0
                else None
            )
            after_non_zero_idx = (
                non_zero_indices[non_zero_indices > time_idx][0]
                if len(non_zero_indices[non_zero_indices > time_idx]) > 0
                else None
            )
            if before_non_zero_idx is not None and after_non_zero_idx is not None:
                interpolated_value = (
                    x_enc_np[sample_idx, before_non_zero_idx, channel_idx]
                    + x_enc_np[sample_idx, after_non_zero_idx, channel_idx]
                ) / 2
            elif before_non_zero_idx is None:
                interpolated_value = x_enc_np[
                    sample_idx, after_non_zero_idx, channel_idx
                ]
            elif after_non_zero_idx is None:
                interpolated_value = x_enc_np[
                    sample_idx, before_non_zero_idx, channel_idx
                ]
            interpolated_x_enc[sample_idx, time_idx, channel_idx] = interpolated_value

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # time series decompsition
        if self.use_avg is True:
            seasonal_part, trend_part = self.decompsition(x_enc)
            x_enc = seasonal_part.permute(0, 2, 1)
            # Mapping trend part to target length
            trend_part = trend_part.permute(0, 2, 1)
            trend_part = self.projection_trend(trend_part)
            trend_part = trend_part.permute(0, 2, 1)
        else:
            x_enc = x_enc.permute(0, 2, 1)

        # do patching and reshape
        x_enc = self.patching(ts=x_enc)  # [batch_size, n_vars, patch_num, patch_len]
        batch_size, n_vars, patch_num, patch_len = x_enc.size()
        # 以通道独立的方式来处理数据
        x_enc = torch.reshape(x_enc, shape=(batch_size * n_vars, patch_num, patch_len))
        # 经过大模型正向传播部分
        x_dec = self.time_encoder(x_enc)  # [batch_size * n_vars, patch_num, d_model]
        x_dec = torch.reshape(
            x_dec, shape=(batch_size, n_vars, x_dec.size(-2), self.d_model)
        )

        # 恢复为模型原本的输出维度
        x_dec = self.flatten_head(x_dec).permute(
            0, 2, 1
        )  # [batch_size, pred_len, num_vars]

        # add the trend part of the decompsition
        if self.use_avg is True:
            x_dec = x_dec + trend_part

        # De-Normalization from Non-stationary Transformer
        x_dec = x_dec * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        x_dec = x_dec + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return x_dec

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        """进行异常检测的接口"""
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # time series decompsition
        if self.use_avg is True:
            seasonal_part, trend_part = self.decompsition(x_enc)
            x_enc = seasonal_part.permute(0, 2, 1)
            # Mapping trend part to target length
            trend_part = trend_part.permute(0, 2, 1)
            trend_part = self.projection_trend(trend_part)
            trend_part = trend_part.permute(0, 2, 1)
        else:
            x_enc = x_enc.permute(0, 2, 1)

        # do patching and reshape
        x_enc = self.patching(ts=x_enc)  # [batch_size, n_vars, patch_num, patch_len]
        batch_size, n_vars, patch_num, patch_len = x_enc.size()
        # 以通道独立的方式来处理数据
        x_enc = torch.reshape(x_enc, shape=(batch_size * n_vars, patch_num, patch_len))
        # 经过大模型正向传播部分
        x_dec = self.time_encoder(x_enc)  # [batch_size * n_vars, patch_num, d_model]
        x_dec = torch.reshape(
            x_dec, [batch_size, n_vars, x_dec.shape[-2], x_dec.shape[-1]]
        )

        # 恢复为模型原本的输出维度
        x_dec = self.flatten_head(x_dec).permute(
            0, 2, 1
        )  # [batch_size, pred_len, num_vars]

        # add the trend part of the decompsition
        if self.use_avg is True:
            x_dec = x_dec + trend_part

        # De-Normalization from Non-stationary Transformer
        x_dec = x_dec * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        x_dec = x_dec + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        return x_dec
