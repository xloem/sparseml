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
from torch.nn import Module
import logging
from typing import Any, List, Union

from sparseml.optim import BaseModifier, ModifierProp
from sparseml.pytorch.sparsification.distillation.modifier_distillation_base import (
    BaseDistillationModifier,
)
from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML
from sparseml.sparsification import SparsificationTypes


__all__ = [
    "FeatureImitationModifier",
]

_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class FeatureImitationModifier(BaseDistillationModifier):
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

    :param number_of_classes:
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param distill_output_keys: list of keys for the module outputs to use for
        distillation if multiple outputs are present. None or empty list defaults
        to using all available outputs
    :param teacher_input_keys: list of keys to filter the inputs by before
        passing into the teacher. None or empty list defaults to using
        all available inputs
    :param update_frequency:
    :param gain: how much to weight the distillation loss. Default is 1.5
    :param output_format:
    :param feature_format:
    """

    def __init__(
        self,
        number_of_classes: int,
        student_features: List[int],
        teacher_features: List[int],
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        distill_output_keys: List[Any] = None,
        teacher_input_keys: List[Any] = None,
        update_frequency: float = -1.0,
        gain: float = 1.5,
        output_format: str = "bayxo",
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
        self.student_features = student_features
        self.teacher_features = teacher_features
        self.gain = gain
        self.output_format = output_format
        self.feature_format = feature_format
        self._initialize_projection()

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.feature_distillation, SparsificationTypes.distillation]

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
    def student_features(self) -> List[int]:
        return self._student_features

    @student_features.setter
    def student_features(self, value: List[int]):
        self._student_features = value

    @ModifierProp()
    def teacher_features(self) -> List[int]:
        return self._teacher_features

    @teacher_features.setter
    def teacher_features(self, value: List[int]):
        self._teacher_features = value

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
    def output_format(self) -> str:
        return self._output_format

    @output_format.setter
    def output_format(self, value: str):
        self._output_format = value

    @ModifierProp()
    def feature_format(self) -> str:
        return self._feature_format

    @feature_format.setter
    def feature_format(self, value: str):
        self._feature_format = value

    @ModifierProp(serializable=False)
    def output_class_dimension(self) -> int:
        return self.output_format.index("o")

    @ModifierProp(serializable=False)
    def output_anchor_dimension(self) -> int:
        return self.output_format.index("a")

    @ModifierProp(serializable=False)
    def feature_dimension(self) -> int:
        return self.feature_format.index("o")

    @ModifierProp(serializable=False)
    def number_of_layers(self) -> int:
        return len(self.student_features)

    @ModifierProp(serializable=False)
    def projection(self) -> List[Module]:
        return self._projection

    @projection.setter
    def projection(self, value: List[Module]):
        self._projection = value

    def compute_distillation_loss(self, student_outputs, teacher_outputs, **kwargs):
        distillation_loss = 0.0
        for layer in range(self.number_of_layers):
            student_class_scores = self._get_scores(student_outputs["output"][layer])
            teacher_class_scores = self._get_scores(teacher_outputs["output"][layer])
            projection_weight = torch.mean(
                (student_class_scores - teacher_class_scores)**2,
                dim=(self.output_anchor_dimension, self.output_class_dimension)
            )
            teacher_features = teacher_outputs["feature"][layer]
            if self.projection[layer] is not None:
                self.projection[layer] = self.projection[layer].to(teacher_features.device)
                self.projection[layer] = self.projection[layer].to(teacher_features.dtype)
                teacher_features = self.projection[layer](teacher_outputs["feature"][layer])

            feature_difference = torch.mean(
                (student_outputs["feature"][layer] - teacher_features)**2,
                dim=self.feature_dimension,
            )

            fi_loss = torch.mean(projection_weight * feature_difference)

            distillation_loss += fi_loss / self.number_of_layers

        return distillation_loss

    def compute_total_loss(self, loss, distillation_loss):
        return loss + self.gain * distillation_loss

    def _initialize_projection(self):
        projection = []
        for layer in range(self.number_of_layers):
            if self.student_features[layer] == self.teacher_features[layer]:
                projection.append(None)
            else:
                projection.append(
                    torch.nn.Conv2d(
                        self.teacher_features[layer],
                        self.student_features[layer],
                        1,
                        bias=False
                    )
                )
        self.projection = projection

    def _get_scores(self, outputs):
        _, scores = torch.split(outputs, (5, self.number_of_classes), dim=self.output_class_dimension)
        return scores

