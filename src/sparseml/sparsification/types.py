# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Base classes and implementations for types of sparsification algorithms.
"""

from enum import Enum


__all__ = ["SparsificationTypes"]


class SparsificationTypes(Enum):
    """
    SparsificationTypes to give context to what a modifier or other parts of the
    system are and can do when applied to a model for sparsification.
    """

    general = "general"
    epoch = "epoch"
    learning_rate = "learning_rate"
    activation_sparsity = "activation_sparsity"
    pruning = "pruning"
    quantization = "quantization"
    distillation = "distillation"
    per_layer_distillation = "per_layer_distillation"
    regularization = "regularization"
    structured = "structured"
