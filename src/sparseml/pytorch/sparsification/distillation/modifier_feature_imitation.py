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
from typing import Any, Callable, List, Optional, Union

import torch
from torch.nn import Module

from sparseml.optim import BaseModifier, ModifierProp
from sparseml.pytorch.sparsification.distillation.modifier_distillation_base import (
    BaseDistillationModifier,
)
from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML
from sparseml.pytorch.utils import BaseLogger


__all__ = [
    "FeatureImitationModifier",
]

_LOGGER = logging.getLogger(__name__)

@PyTorchModifierYAML()
class FeatureImitationModifier(BaseDistillationModifier):
    """
    Adds a knowledge distillation loss based on the feature imitation loss.
    A distillation_teacher module may be provided as a kwarg to
    the Manager initialization and loss_update(loss) must be called before any
    backwards pass in the integrated training flow.
    If no teacher model is provided, then self-distillation will be used.
    The feature difference between teacher and student can be weighted spatially
    by a weighing function.

    | Sample yaml:
    |   !FeatureImitationModifier
    |       start_epoch: 0.0
    |       gain: 2.0
    |       number_of_classes: 80
    |       student_features: [64, 128, 256]
    |       teacher_features: [128, 256, 512]

    :param number_of_classes: Number of classes
    :param student_features: List containing the number of features at each layer
        of the student model
    :param teacher_features: List containing the number of features at each layer
        of the teacher model
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
    :param output_format: Format for output tensors following this convention:
        ("b"=batch size, "a"=anchors, "x"=horizontal tiles, "y"=vertical tiles,
         "o"=outputs)
    :param feature_format: Format for feature tensors following this convention:
        ("b"=batch size, "x"=horizontal tiles, "y"=vertical tiles, "o"=outputs)
    :param weight_function: Optional string to identify function to weight the
        difference between teacher and student feature
    """

    def __init__(
        self,
        number_of_classes: int,
        student_features: List[int],
        teacher_features: List[int],
        gain: float,
        student_feature_type: str = None,
        teacher_feature_type: Optional[str] = None,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        distill_output_keys: Optional[List[Any]] = None,
        teacher_input_keys: Optional[List[Any]] = None,
        update_frequency: float = -1.0,
        output_format: str = "bayxo",
        feature_format: str = "boyx",
        weight_function: Optional[str] = None,
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
        self.student_feature_type = student_feature_type
        self.teacher_feature_type = teacher_feature_type
        self.output_format = output_format
        self.feature_format = feature_format
        self.weight_function = weight_function
        self._student_feature_tensors = None
        self._teacher_feature_tensors = None
        self._student_handle = None
        self._teacher_handle = None
        self._set_compute_weight()
        self._initialize_projection()

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
    def student_feature_type(self) -> str:
        return self._student_feature_type

    @student_feature_type.setter
    def student_feature_type(self, value: str):
        self._student_feature_type = value

    @ModifierProp()
    def teacher_feature_type(self) -> str:
        if self._teacher_feature_type is None:
            return self._student_feature_type
        else:
            return self._teacher_feature_type

    @teacher_feature_type.setter
    def teacher_feature_type(self, value: str):
        self._teacher_feature_type = value

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

    @ModifierProp()
    def weight_function(self) -> str:
        return self._weight_function

    @weight_function.setter
    def weight_function(self, value: str):
        self._weight_function = value

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

    @ModifierProp(serializable=False)
    def compute_weight(self) -> Callable:
        weight_methods = {
            "prediction": self._weight_prediction,
        }
        if self.weight_function in weight_methods:
            return weight_methods.get(self.weight_function, None)

    @compute_weight.setter
    def compute_weight(self, value: Callable):
        self._compute_weight = value

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
            def cache_input(features):
                def forward_hook_fn(layer, inp, out):
                    features = inp
                return forward_hook_fn

            def find_layers(layer_module, feature_type):
                if layer_module.__class__.__name__ == feature_type:
                    return layer_module
                else:
                    for child in layer_module.children():
                        detect_layer = find_layers(child, feature_type)
                        if detect_layer is not None:
                            return detect_layer
                    return None

            student_detection_layer = find_layers(module, self.student_feature_type)
            teacher_detection_layer = find_layers(self._teacher, self.teacher_feature_type)

            self._student_handle = student_detection_layer.register_forward_hook(
                        cache_input(self._student_feature_tensors)
                    )
            self._teacher_handle = teacher_detection_layer.register_forward_hook(
                        cache_input(self._teacher_feature_tensors)
                    )
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
        self._student_handle.remove()
        self._teacher_handle.remove()
        self._student_handle = None
        self._teacher_handle = None
        self._student_feature_tensors = None
        self._teacher_feature_tensors = None

    def compute_distillation_loss(self, student_outputs, teacher_outputs, **kwargs):
        distillation_loss = 0.0
        for layer in range(self.number_of_layers):
            student_features = self._student_feature_tensors[layer]
            teacher_features = self._teacher_feature_tensors[layer]
            self.projection[layer] = self.projection[layer].to(student_features.device)
            self.projection[layer] = self.projection[layer].to(student_features.dtype)
            student_projected_features = self.projection[layer](student_features)

            feature_difference = torch.mean(
                (student_projected_features - teacher_features) ** 2,
                dim=self.feature_dimension,
            )

            if self.weight_function is not None:
                weight = self.compute_weight(layer, student_outputs, teacher_outputs)
            else:
                weight = 1.0

            fi_loss = torch.mean(weight * feature_difference)
            distillation_loss += fi_loss

        return distillation_loss / self.number_of_layers

    def compute_total_loss(self, loss, distillation_loss):
        return loss + self.gain * distillation_loss

    def _initialize_projection(self):
        projection = []
        for layer in range(self.number_of_layers):
            projection.append(
                torch.nn.Conv2d(
                    in_channels=self.student_features[layer],
                    out_channels=self.teacher_features[layer],
                    kernel_size=1,
                    bias=False,
                )
            )
        self.projection = projection

    def _set_compute_weight(self):
        weight_methods = {"prediction": self._weight_prediction}
        if self.weight_function is None:
            self.compute_weight = None
        else:
            self.compute_weight = weight_methods[self.weight_function]

    def _get_scores(self, outputs):
        _, scores = torch.split(
            outputs, (5, self.number_of_classes), dim=self.output_class_dimension
        )
        return torch.sigmoid(scores)

    def _weight_prediction(self, layer, student_outputs, teacher_outputs):
        """
        Prediction-guided weight for feature imitation.
        Adapted from the paper "Knowledge Distillation for Object Detection
        via Rank Mimicking and Prediction-guided Feature Imitation"
        (https://arxiv.org/abs/2112.04840)
        """

        student_class_scores = self._get_scores(student_outputs[layer])
        teacher_class_scores = self._get_scores(teacher_outputs[layer])

        weight = torch.mean(
            (student_class_scores - teacher_class_scores) ** 2,
            dim=(self.output_anchor_dimension, self.output_class_dimension),
        )

        return weight
