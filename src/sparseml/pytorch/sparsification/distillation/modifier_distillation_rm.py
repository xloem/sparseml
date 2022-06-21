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

import logging
from typing import Any, List

from sparseml.optim import ModifierProp
from sparseml.pytorch.utils.positive_box_methods import MatchAnchorIOU
from sparseml.pytorch.sparsification.distillation.modifier_distillation_base import (
    BaseDistillationModifier,
    kl_logsoftmax,
)
from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML


_POSITIVE_BOX_METHODS = {
    "match_anchor_iou": MatchAnchorIOU,
}


__all__ = [
    "RankMimickingModifier",
]


_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class RankMimickingModifier(BaseDistillationModifier):
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
    :param end_epoch: The epoch to start the modifier at
    :param gain:
    :param temperature: temperature applied to teacher and student softmax for
        distillation
    :param positive_box_method:
    :param positive_box_method_args:
    :param scale_with_batch_size:
    :param distill_output_keys: list of keys for the module outputs to use for
        distillation if multiple outputs are present. None or empty list defaults
        to using all available outputs
    :param teacher_input_keys: list of keys to filter the inputs by before
        passing into the teacher. None or empty list defaults to using
        all available inputs
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        distill_output_keys: List[Any] = None,
        teacher_input_keys: List[Any] = None,
        update_frequency: float = -1.0,
        gain: float = 4.0,
        temperature: float = 1.0,
        positive_box_method: str = "match_anchor_iou",
        positive_box_method_args: Any = None,
    ):
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            distill_output_keys=distill_output_keys,
            teacher_input_keys=teacher_input_keys,
            update_frequency=update_frequency,
        )
        self.gain = gain
        self.temperature = temperature
        self._positive_outputs = _POSITIVE_BOX_METHODS[positive_box_method](
            **positive_box_method_args
        )

    @ModifierProp()
    def gain(self) -> float:
        """
        :return: how much to weight the distillation loss
        """
        return self._gain

    @gain.setter
    def gain(self, value: float):
        """
        :params value: how much to weight the distillation loss
        """
        self._gain = value

    @ModifierProp()
    def temperature(self) -> float:
        """
        :return: temperature applied to teacher and student softmax for
            distillation
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        """
        :params value: temperature applied to teacher and student softmax for
            distillation
        """
        self._temperature = value

    def compute_distillation_loss(
        self, student_outputs, teacher_outputs, student_labels, **kwargs
    ):
        distillation_loss = 0.0
        positive_student_outputs, positive_teacher_outputs = self._positive_outputs(
            student_outputs, teacher_outputs, student_labels
        )
        if (
            positive_student_outputs is not None
            and positive_teacher_outputs is not None
        ):
            distillation_loss += kl_logsoftmax(
                positive_student_outputs, positive_teacher_outputs, self.temperature
            )

        return distillation_loss

    def compute_total_loss(self, loss, distillation_loss):
        return loss + self.gain * distillation_loss
