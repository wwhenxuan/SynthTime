# -*- coding: utf-8 -*-
"""
Created on 2024/9/23 17:00
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
获取使用的损失函数的接口模块
"""
from typing import Tuple, Dict, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

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
