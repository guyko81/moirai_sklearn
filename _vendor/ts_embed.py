#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(ResidualBlock, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        self.hidden_layer = nn.Linear(input_dims, hidden_dims)
        self.silu = nn.SiLU()
        self.output_layer = nn.Linear(hidden_dims, output_dims)
        self.residual_layer = nn.Linear(input_dims, output_dims)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        hidden = self.hidden_layer(x)
        hidden = self.silu(hidden)
        output = self.output_layer(hidden)
        residual = self.residual_layer(x)
        return output + residual
