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
from typing import Any, List, Optional, Union

import torch
from torch.nn import Module

from sparseml.optim import ModifierProp
from sparseml.pytorch.sparsification.distillation.modifier_distillation_base import (
    BaseDistillationModifier,
)
from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML
from sparseml.pytorch.utils import BaseLogger


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
        self.cached_student_output = None
        self.cached_teacher_output = None
        self.student_handles = None
        self.teacher_handles = None

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

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        distillation_teacher: Union[Module, str] = "disable",
        **kwargs,
    ):
        """
        Store the teacher model for distillation if provided

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param distillation_teacher: teacher module to perform knowledge distillation
            with. If not provided, self distillation will be used with a teacher
             from a copy of the given module at the start epoch. If given string
             "disable" this modifier will not apply distillation of any kind,
             even in the active epoch range
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().initialize(module, epoch, loggers, distillation_teacher, **kwargs)

        if isinstance(distillation_teacher, Module):
            self.cached_student_output = {}
            self.cached_teacher_output = {}

            def cache_output(name, outputs):
                def forward_hook_fn(layer, inp, out):
                    outputs[name] = out

                return forward_hook_fn

            def find_layers(layer_module, cached_layers, name=""):
                if isinstance(layer_module, torch.nn.Conv2d) or isinstance(
                    layer_module, torch.nn.Linear
                ):
                    cached_layers[name] = layer_module
                for layer_module, child in layer_module.named_children():
                    find_layers(
                        child,
                        cached_layers,
                        name + "." + layer_module if name != "" else layer_module,
                    )

            cached_student_layers = {}
            cached_teacher_layers = {}
            find_layers(module, cached_student_layers)
            find_layers(distillation_teacher, cached_teacher_layers)
            cached_student_layers_ = {}
            cached_teacher_layers_ = {}
            for layer_name in cached_student_layers:
                if layer_name in cached_teacher_layers:
                    cached_student_layers_[layer_name] = cached_student_layers[
                        layer_name
                    ]
                    cached_teacher_layers_[layer_name] = cached_student_layers[
                        layer_name
                    ]
            cached_student_layers = cached_student_layers_
            cached_teacher_layers = cached_teacher_layers_

            self.student_handles = []
            self.teacher_handles = []
            for layer_name in cached_student_layers:
                self.student_handles.append(
                    cached_student_layers[layer_name].register_forward_hook(
                        cache_output(layer_name, self.cached_student_output)
                    )
                )
                self.teacher_handles.append(
                    cached_teacher_layers[layer_name].register_forward_hook(
                        cache_output(layer_name, self.cached_teacher_output)
                    )
                )
            self._teacher = distillation_teacher
        else:
            raise ValueError(
                "unrecognized value for distillation_modifier given of "
                f"{distillation_teacher}. "
                "To disable set to 'disable' and for self attention set to 'self'"
            )

    def finalize(
        self, module: Optional[Module] = None, reset_loggers: bool = True, **kwargs
    ):
        """
        Cleans up any state and hooks

        :param module: The model/module to finalize the modifier for.
            Marked optional so state can still be cleaned up on delete,
            but generally should always be passed in.
        :param reset_loggers: True to remove any currently attached loggers (default),
            False to keep the loggers attached.
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().finalize(module, reset_loggers, **kwargs)
        for handle in self.student_handles:
            handle.remove()
        for handle in self.teacher_handles:
            handle.remove()
        self.student_handles = None
        self.teacher_handles = None
        self.cached_student_output = None
        self.cached_teacher_output = None

    def compute_distillation_loss(self, **kwargs):
        distillation_loss = 0.0

        for layer_name in self.cached_student_output:
            student_module_output = self.cached_student_output[layer_name]
            teacher_module_output = self.cached_teacher_output[layer_name]

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
