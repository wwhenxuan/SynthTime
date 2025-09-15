# -*- coding: utf-8 -*-
"""
Created on 2025/09/12 10:36:58
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan
"""
from .projection import QueryKeyProjection, RotaryProjection
from .projection import (
    get_activation,
    get_talking_heads,
    Normalization,
    TimeFreqConcatenate,
    DepthWiseConv,
)

from .transformer import TransformerEncoder, TransformerLayer

from .block_family import AttentionLayer
from .block_family import TemporalAttention
from .block_family import FrequencyFilter
from .block_family import TimeFreqFusion
from .block_family import SEAttention

from .embedding import ResidualEmbedding, FlattenHeads
