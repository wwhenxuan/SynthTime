# -*- coding: utf-8 -*-
"""
Created on 2024/9/23 17:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
获取使用的损失函数的接口模块
"""
from typing import Optional, Union, Tuple, Dict, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from torch import fft

from accelerate import Accelerator

from colorama import Fore, Style, init

init(autoreset=True)


def get_criterion(
    name: str = "MSE", accelerator: Accelerator = None, args=None
) -> nn.Module:
    """获取神经网络损失函数的接口配置"""
    if accelerator is not None:
        accelerator.print(
            Fore.RED + f"Now is loading criterion: {name} " + Style.RESET_ALL + "->",
            end=" ",
        )
    if name == "MSE":
        fun = nn.MSELoss()
    elif name == "MAE":
        fun = nn.L1Loss()
    elif name == "CEL":
        fun = nn.CrossEntropyLoss()
    elif name == "Huber":
        fun = nn.SmoothL1Loss()
    elif name == "Cos":
        fun = nn.CosineEmbeddingLoss()
    elif name == "CLIP":
        fun = ClipLoss(accelerator=accelerator, local_loss=args.local_loss)
    else:
        raise ValueError("损失函数名称填写错误!")
    if accelerator is not None:
        accelerator.print(Fore.GREEN + "Done!" + Style.RESET_ALL)
    return fun


class ClipLoss(nn.Module):
    """创建经典的CLIP或SimCLR对比损失对象"""

    def __init__(self, accelerator: Accelerator, local_loss: bool = False) -> None:
        super(ClipLoss, self).__init__()
        # 创建多卡并行使用的Accelerator对象
        self.accelerator = accelerator
        # 是否仅使用本地的损失
        self.local_loss = local_loss

        # 获得多卡并行的进程数目
        self.world_size = accelerator.num_processes

        # 获取本次损失计算的设备信息
        self.device = self.accelerator.device

    def get_ground_truth(self, num_logits: int) -> Tensor:
        """calculated ground-truth and cache if enabled"""
        labels = torch.arange(num_logits, device=self.device, dtype=torch.long)
        if self.world_size > 1 and self.local_loss:
            # TODO: 这里为什么要这样做
            labels = labels + num_logits * self.accelerator.process_index
        return labels

    def get_logits(
        self, time_features: Tensor, graph_features: Tensor, logit_scale: Tensor
    ) -> Tuple[Tensor, Tensor]:
        all_time_features, all_graph_features = self.concat_all_gather(
            time_features=time_features, graph_features=graph_features
        )

        if self.local_loss is True:
            # 仅在当前的设备上计算对比损失
            logits_per_time = logit_scale * time_features @ all_graph_features.T
            logits_per_graph = logit_scale * graph_features @ all_time_features.T
        else:
            # 利用所有设备上的特征作为总损失
            logits_per_time = logit_scale * all_time_features @ all_graph_features.T
            logits_per_graph = logits_per_time.T

        return logits_per_time, logits_per_graph

    def concat_all_gather(
        self, time_features: Tensor, graph_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """用于汇聚所有的对比损失特征"""
        all_time_features = self.accelerator.gather(tensor=time_features)
        all_graph_features = self.accelerator.gather(tensor=graph_features)

        return all_time_features, all_graph_features

    def forward(
        self,
        time_features: Tensor,
        graph_features: Tensor,
        logit_scale: Tensor,
        output_dict: bool = False,
    ) -> Tensor | Dict[str, Tensor]:
        """通过特征的正向传播来计算损失"""

        logits_per_time, logits_per_graph = self.get_logits(
            time_features, graph_features, logit_scale
        )

        # 获得ground true标签用于计算对比损失
        labels = self.get_ground_truth(logits_per_time.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_time, labels)
            + F.cross_entropy(logits_per_graph, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class MaskedFilterLoss(nn.Module):
    """
    本文提出的全新的预训练损失，
    原本的时间序列掩码是将被掩码的时间序列全部设置为0，
    该方法在一定程度上破环了时间序列数据的连续性和时序相依关系。

    因此本文通过构造随机的（待定）维纳滤波器对被掩码部分的时间序列进行频率滤波，
    并使模型来恢复和重构被滤波的部分，而非从0开始重构。
    这种方式能够使模型根据相邻的时间序列Patch的周期与趋势特征，
    来恢复被过滤掉的周期组分，
    因此该方法不会破坏时间序列数据的连续性和时序相依关系。

    TODO: 分析这个损失在学习上是否存在难度
    """

    def __init__(
        self,
        accelerator: Accelerator,
        weight: Optional[float] = 0.5,
        return_loss_dict: Optional[bool] = True,
    ) -> None:
        super(MaskedFilterLoss, self).__init__()

        self.accelerator = accelerator
        self.device = self.accelerator.device

        # 用于分配时序和频域重构损失的权重
        self.weight = weight

        # 是否要以字典的形式返回所有的损失
        self.return_loss_dict = return_loss_dict

    def __str__(self) -> str:
        return "MaskedFilterLoss"

    @staticmethod
    def reconstruct_loss(
        self,
        logist: torch.Tensor,
        labels: torch.Tensor,
        attn_mask: torch.Tensor,
        p: Optional[Union[int, float]] = 2,
    ) -> torch.Tensor:
        """计算重构的损失"""
        masked_loss = (logist - labels) ** p
        masked_loss = masked_loss.mean(dim=-1)

        # Only make losses in the masked areas
        return (masked_loss * (~attn_mask).int()).sum() / (~attn_mask).sum()

    def forward(
        self,
        time_patch: torch.Tensor,
        time_reconstruct: torch.Tensor,
        freq_patch: torch.Tensor,
        freq_reconstruct: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """通过正向传播计算这部分损失的具体数值"""
        # TODO: 这个损失可以同时在时域和频域中进行引入

        # 对时序的重构部分进行傅里叶变换得到频域中的重构部分
        # # TODO: 这里进行傅里叶变换的时候可能需要选择维度
        # freq_reconstruct = fft.fft(time_reconstruct)

        # 计算时域损失
        time_loss = self.reconstruct_loss(
            logist=time_patch, labels=time_reconstruct, attn_mask=attn_mask
        )

        # 计算频域损失
        freq_loss = self.reconstruct_loss(
            logist=freq_patch, labels=freq_reconstruct, attn_mask=attn_mask
        )

        # 根据权重计算最后的损失
        loss = self.weight * time_loss + (1 - self.weight) * freq_loss

        # 返回最后的损失信息
        if self.return_loss_dict:
            return {"time": time_loss, "freq": freq_loss, "sum": loss}
        else:
            return loss
