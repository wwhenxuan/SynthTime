# -*- coding: utf-8 -*-
"""
Created on 2025/03/05 00:00:27
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
from accelerate import Accelerator


def print_args(args, accelerator: Accelerator) -> None:
    # Printing the basic config of model and training
    accelerator.print(
        "\033[1m" + f"Basic Config of {args.model} in {args.tasks}" + "\033[0m"
    )
    accelerator.print(
        f'  {"Task:":<20}{args.tasks:<20}{"Model ID:":<20}{args.model:<20}'
    )
    accelerator.print(
        f'  {"Num Epochs:":<20}{args.num_epochs:<20}{"Warmup Epochs:":<20}{args.warmup_epochs:<20}'
    )
    accelerator.print(
        f'  {"Batch Size:":<20}{args.batch_size:<20}{"shuffle:":<20}{args.shuffle:<20}'
    )
    accelerator.print(
        f'  {"num_workers:":<20}{args.num_workers:<20}{"Anneal Strategy::":<20}{args.anneal_strategy:<20}'
    )
    accelerator.print(
        f'  {"optimizer:":<20}{args.optimizer:<20}{"criterion:":<20}{args.criterion:<20}'
    )
    accelerator.print(
        f'  {"scheduler:":<20}{args.scheduler:<20}{"learning_rate:":<20}{args.learning_rate:<20}'
    )
    accelerator.print(
        f'  {"momentum:":<20}{args.momentum:<20}{"weight_decay:":<20}{args.weight_decay:<20}'
    )
    accelerator.print(f'  {"beta1:":<20}{args.beta1:<20}{"beta2:":<20}{args.beta2:<20}')
    accelerator.print(f'  {"eps:":<20}{args.eps:<20}{"amsgrad:":<20}{args.amsgrad:<20}')
    accelerator.print(
        f'  {"Step size:":<20}{args.step_size:<20}{"gamma:":<20}{args.gamma:<20}'
    )
    accelerator.print(
        f'  {"cycle_momentum:":<20}{args.cycle_momentum:<20}{"Base Momentum:":<20}{args.base_momentum:<20}'
    )
    accelerator.print(
        f'  {"Max Momentum:":<20}{args.max_momentum:<20}{"Graph Maker:":<20}{args.graph_generate:<20}'
    )

    if args.tasks == "finetune":
        accelerator.print("\033[1m" + f"{args.model} Model Fine-tuning!" + "\033[0m")
        accelerator.print(
            f'  {"Dataset":<20}{args.dataset:<20}{"train Number (K):":<20}{args.train_num:<20}'
        )
        accelerator.print(
            f'  {"Variate Number:":<20}{args.num_vars:<20}{"Per-trained Params:":<20}{args.pretrained_params_path:<20}'
        )
        accelerator.print(
            f'  {"Training Epochs:":<20}{args.num_epochs:<20}{"Test Epoch:":<20}{args.test_epochs:<20}'
        )

    if args.tasks == "pretrain":
        accelerator.print("\033[1m" + f"{args.model} Model Pre-training!" + "\033[0m")
        accelerator.print(
            f'  {"Variate Number:":<20}{args.num_vars:<20}{"Per-trained Params:":<20}{args.pretrained_params_path:<20}'
        )
        accelerator.print(
            f'  {"Training Epochs:":<20}{args.num_epochs:<20}{"Save Epoch:":<20}{args.save_epochs:<20}'
        )

    if args.model == "TGCL":
        accelerator.print(
            "\033[1m" + "Hyperparameters for Time Series Encoders" + "\033[0m"
        )
        accelerator.print(
            f'  {"Layers:":<20}{args.time_layers:<20}{"d_model:":<20}{args.d_model:<20}'
        )
        accelerator.print(
            f'  {"d_ff:":<20}{args.d_ff:<20}{"n_heads:":<20}{args.n_heads:<20}'
        )
        accelerator.print(
            f'  {"Dropout:":<20}{args.time_dropout:<20}{"Activation:":<20}{args.time_act:<20}'
        )
        accelerator.print(
            f'  {"Patch Len:":<20}{args.patch_len:<20}{"Use Conv:":<20}{args.time_act:<20}'
        )
        accelerator.print(
            f'  {"Dropout:":<20}{args.time_dropout:<20}{"Activation:":<20}{args.time_act:<20}'
        )

        accelerator.print("\033[1m" + "Hyperparameters for Graph Encoders" + "\033[0m")
        accelerator.print(
            f'  {"GNN:":<20}{args.graph_network:<20}{"Readout:":<20}{args.readout:<20}'
        )
        accelerator.print(
            f'  {"Layers:":<20}{args.graph_layers:<20}{"Num Heads:":<20}{args.gnn_heads:<20}'
        )
        accelerator.print(
            f'  {"Dropout:":<20}{args.graph_dropout:<20}{"Activation:":<20}{args.graph_act:<20}'
        )
        accelerator.print(
            f'  {"Norm:":<20}{args.norm:<20}{"Graph Pooling:":<20}{args.pooling_type:<20}'
        )
        accelerator.print(
            f'  {"In Channels:":<20}{args.in_channels:<20}{"Out Channels:":<20}{args.out_channels:<20}'
        )
    accelerator.print()
