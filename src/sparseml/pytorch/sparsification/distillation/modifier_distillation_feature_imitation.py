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
Modifier for performing model distillation
"""

import torch
import logging
from typing import Any, List

from sparseml.optim import ModifierProp
from sparseml.pytorch.sparsification.distillation.modifier_distillation_base import (
    BaseDistillationModifier,
    kldiv_loss,
)
from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML


__all__ = [
    "FeatureImitationModifier",
]

_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class DistillationModifier(BaseDistillationModifier):
    """
    Adds a knowledge distillation loss based on a teacher model during the
    loss_update phase of the SparseML lifecycle. A distillation_teacher
    module may be provided as a kwarg to the Manager initialization and
    loss_update(loss) must be called before any backwards pass in the integrated
    training flow. If no teacher model is provided, then self distillation
    will be used

    | Sample yaml:
    |   !DistillationModifier
    |       start_epoch: 0.0
    |       hardness: 0.5
    |       temperature: 2.0
    |       distill_output_keys: [0]

    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param distill_output_keys: list of keys for the module outputs to use for
        distillation if multiple outputs are present. None or empty list defaults
        to using all available outputs
    :param teacher_input_keys: list of keys to filter the inputs by before
        passing into the teacher. None or empty list defaults to using
        all available inputs
    :param hardness: how much to weight the distillation loss vs the base loss
        (e.g. hardness of 0.6 will return 0.6 * distill_loss + 0.4 * base_loss).
        Default is 0.5
    :param temperature: temperature applied to teacher and student softmax for
        distillation
    """

    def __init__(
        self,
        number_of_classes: int,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        distill_output_keys: List[Any] = None,
        teacher_input_keys: List[Any] = None,
        update_frequency: float = -1.0,
        gain: float = 1.5,
        output_format: str = "boayx",
        feature_format: str = "boyx",
    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            distill_output_keys=distill_output_keys,
            teacher_input_keys=teacher_input_keys,
            update_frequency=update_frequency,
        )
        self.number_of_classes = number_of_classes
        self.gain = gain

    @ModifierProp()
    def number_of_classes(self) -> int:
        """
        :return: how much to weight the distillation loss vs the base loss
            (e.g. hardness of 0.6 will return 0.6 * distill_loss + 0.4 * base_loss)
        """
        return self._number_of_classes

    @number_of_classes.setter
    def number_of_classes(self, value: int):
        """
        :params value: how much to weight the distillation loss vs the base loss
            (e.g. hardness of 0.6 will return 0.6 * distill_loss + 0.4 * base_loss)
        """
        self._number_of_classes = value

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

    def compute_distillation_loss(self, student_outputs, teacher_outputs, **kwargs):
        number_layers = len(student_outputs["prediction"])
        distillation_loss = 0.0
        for layer in range(number_layers):
            student_class_scores = self._get_scores(student_outputs["prediction"][layer])
            teacher_class_scores = self._get_scores(teacher_outputs["prediction"][layer])
            projection_weight = torch.mean((student_class_scores - teacher_class_scores)**2, dim=(self.output_anchor_dimension, self.output_class_dimension))
            feature_difference = torch.mean((student_outputs["feature"] - teacher_outputs["feature"])**2, dim=self.feature_dimension)

            fi_loss = torch.mean(projection_weight * feature_difference, dim=(self.feature_y_dimension, self.feature_x_dimension))
            distillation_loss += fi_loss

        return fi_loss

    def compute_total_loss(self, loss, distillation_loss):
        return loss + self.gain * distillation_loss

