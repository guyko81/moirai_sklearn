#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2.0

import numpy as np


class LastValueImputation:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, x):
        x = x.T
        x[0:1][np.isnan(x[0:1])] = self.value
        mask = np.isnan(x)
        idx = np.arange(len(x))
        if x.ndim == 2:
            idx = np.expand_dims(idx, axis=1)
        idx = np.where(~mask, idx, 0)
        idx = np.maximum.accumulate(idx, axis=0)
        if x.ndim == 2:
            x = x[idx, np.arange(x.shape[1])]
        else:
            x = x[idx]
        return x.T


class CausalMeanImputation:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, x, value=0.0):
        mask = np.isnan(x).T

        last_value_imputation = LastValueImputation(self.value)
        x = last_value_imputation(x)
        mask[0] = False
        x = x.T

        if x.ndim == 1:
            adjusted_values_to_causality = np.concatenate((np.repeat(0.0, 1), x[:-1]))
            cumsum = np.cumsum(adjusted_values_to_causality)
            indices = np.linspace(0, len(x) - 1, len(x))
            indices[0] = 1
            ar_res = cumsum / indices
            x[mask] = ar_res[mask]
        else:
            adjusted_values_to_causality = np.vstack(
                (np.zeros((1, x.shape[1])), x[:-1, :])
            )
            cumsum = np.cumsum(adjusted_values_to_causality, axis=0)
            indices = np.linspace(0, len(x) - 1, len(x)).reshape(-1, 1)
            indices[0] = 1
            ar_res = cumsum / indices
            x[mask] = ar_res[mask]
        return x.T
