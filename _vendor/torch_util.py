#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch


def packed_attention_mask(
    sample_id: torch.Tensor,
) -> torch.Tensor:
    sample_id = sample_id.unsqueeze(-1)
    attention_mask = sample_id.eq(sample_id.mT)
    return attention_mask


def packed_causal_attention_mask(
    sample_id: torch.Tensor,
    time_id: torch.Tensor,
) -> torch.Tensor:
    attention_mask = packed_attention_mask(sample_id)
    expanded_id1 = time_id.unsqueeze(-2)
    expanded_id2 = time_id.unsqueeze(-1)
    compare_res = expanded_id1 <= expanded_id2
    attention_mask = attention_mask * compare_res
    return attention_mask


def safe_div(
    numer: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    return numer / torch.where(
        denom == 0,
        1.0,
        denom,
    )


def size_to_mask(
    max_size: int,
    sizes: torch.Tensor,
) -> torch.Tensor:
    mask = torch.arange(max_size, device=sizes.device)
    return torch.lt(mask, sizes.unsqueeze(-1))
