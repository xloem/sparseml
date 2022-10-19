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
Modifier for performing knowledge distillation via feature imitation.
"""

import logging
from typing import Any, List, Optional

import torch

from sparseml.optim import BaseModifier, ModifierProp
from sparseml.pytorch.sparsification.distillation.modifier_distillation_base import (
    BaseDistillationModifier,
)
from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML
from sparseml.sparsification import SparsificationTypes


__all__ = [
    "PerLayerDistillationModifier",
]

_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class PerLayerDistillationModifier(BaseDistillationModifier):
    """
    Adds a knowledge distillation loss based on the feature imitation loss.
    A distillation_teacher module may be provided as a kwarg to
    the Manager initialization and loss_update(loss) must be called before any
    backwards pass in the integrated training flow.
    If no teacher model is provided, then self-distillation will be used.
    The feature difference between teacher and student can be weighted spatially
    by a weighing function.

    | Sample yaml:
    |   !PerLayerDistillationModifier
    |       start_epoch: 0.0
    |       gain: 2.0
    |       number_of_classes: 80
    |       student_features: [64, 128, 256]
    |       teacher_features: [128, 256, 512]

    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param distill_output_keys: List of keys for the module outputs to use for
        distillation if multiple outputs are present. None or empty list defaults
        to using all available outputs
    :param teacher_input_keys: List of keys to filter the inputs by before
        passing into the teacher. None or empty list defaults to using
        all available inputs
    :param update_frequency:
    :param gain: How much to weight the distillation loss. Default is 1.5
    :param normalize: Whether to normalize the output difference by the
        the magnitude of the teacher's output
    """

    def __init__(
        self,
        gain: float,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        distill_output_keys: Optional[List[Any]] = None,
        teacher_input_keys: Optional[List[Any]] = None,
        update_frequency: float = -1.0,
        normalize: bool = True,
    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            distill_output_keys=distill_output_keys,
            teacher_input_keys=teacher_input_keys,
            update_frequency=update_frequency,
        )
        self.gain = gain
        self.normalize = normalize

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [
            SparsificationTypes.per_layer_distillation,
            SparsificationTypes.distillation,
        ]

    @ModifierProp()
    def gain(self) -> float:
        """
        :return: how much to weight the distillation loss vs the base loss
            (e.g. hardness of 0.6 will return 0.6 * distill_loss + 0.4 * base_loss)
        """
        return self._gain

    @gain.setter
    def gain(self, value: float):
        """
        :params value: how much to weight the distillation loss vs the base loss
            (e.g. hardness of 0.6 will return 0.6 * distill_loss + 0.4 * base_loss)
        """
        self._gain = value

    @ModifierProp()
    def normalize(self) -> bool:
        """
        :return: whether to normalize distillation loss by magnitude of teacher output
        """
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool):
        """
        :params value: whether to normalize distillation loss
            by magnitude of teacher output
        """
        self._normalize = value

    def compute_distillation_loss(self, student_outputs, teacher_outputs, **kwargs):
        distillation_loss = 0.0
        common_modules = set(student_outputs.keys()) & set(teacher_outputs.keys())
        common_modules.remove("output")

        for module in common_modules:
            student_module_output = student_outputs[module]
            teacher_module_output = teacher_outputs[module]

            output_difference = torch.mean(
                (student_module_output - teacher_module_output) ** 2,
            )

            if self.normalize:
                teacher_output_magnitude = torch.mean(teacher_module_output ** 2)
                output_difference /= teacher_output_magnitude

            distillation_loss += output_difference

        return distillation_loss

    def compute_total_loss(self, loss, distillation_loss):
        return loss + self.gain * distillation_loss
