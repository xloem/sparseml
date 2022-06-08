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

from typing import List

import torch

from sparseml.pytorch.utils.yolo_helpers import compute_iou


__all__ = [
    "MatchAnchorIOU",
]


class MatchAnchorIOU(object):
    def __init__(
        self,
        anchors: List[List[int]],
        layer_resolution: List[int],
        image_resolution: List[int],
        number_of_classes: int,
        target_format: str = "icyxhw",
        anchor_format: str = "hw",
        output_format: str = "bayxo",
        threshold: float = 0.25,
    ):
        self.anchors = anchors
        self.layer_resolution = layer_resolution
        self.image_resolution = image_resolution
        self.number_of_classes = number_of_classes
        self.target_format = target_format
        self.anchor_format = anchor_format
        self.output_format = output_format
        self.threshold = threshold
        self.number_of_layers = len(self.anchors)
        assert len(self.layer_resolution) == 2 * self.number_of_layers

    @property
    def anchors(self) -> List[List[int]]:
        return self._anchors

    @anchors.setter
    def anchors(self, value: List[List[int]]) -> List[List[int]]:
        assert len(value) > 0
        for v in value:
            assert len(v) > 1
        self._anchors = value

    @property
    def layer_resolution(self) -> List[int]:
        return self._layer_resolution

    @layer_resolution.setter
    def layer_resolution(self, value: List[int]) -> List[int]:
        assert len(value) > 1
        self._layer_resolution = value

    @property
    def target_format(self) -> str:
        return self._target_format

    @target_format.setter
    def target_format(self, value: str):
        assert (
            "i" in value
            and "c" in value
            and "y" in value
            and "x" in value
            and "h" in value
            and "w" in value
        )

        self._target_format = value

    @property
    def anchor_format(self) -> str:
        return self._anchor_format

    @anchor_format.setter
    def anchor_format(self, value: str):
        assert len(value) == 2
        assert "h" in value and "w" in value

        self._anchor_format = value

    @property
    def output_format(self) -> str:
        return self._output_format

    @output_format.setter
    def output_format(self, value: str):
        assert len(value) == 5
        assert (
            "b" in value
            and "a" in value
            and "y" in value
            and "x" in value
            and "o" in value
        )

        self._output_format = value

    def __call__(self, student_outputs, teacher_outputs, target):
        device = target.device
        target_image = self._get_target_image_index(target)
        target_class = self._get_target_class(target)
        target_box = self._get_target_box(target)

        positive_student_outputs = []
        positive_teacher_outputs = []

        found_match = False
        for layer in range(self.number_of_layers):
            student_scores = self._get_class_scores(student_outputs[layer])
            student_scores = self._filter_class(student_scores, target_class)
            student_scores = self._get_image_output(student_scores, target_image)
            student_scores = self._align_dimensions_to_anchor_boxes(student_scores)

            teacher_scores = self._get_class_scores(teacher_outputs[layer])
            teacher_scores = self._filter_class(teacher_scores, target_class)
            teacher_scores = self._get_image_output(teacher_scores, target_image)
            teacher_scores = self._align_dimensions_to_anchor_boxes(teacher_scores)

            layer_anchor_boxes = self._get_anchor_box(layer)
            layer_anchor_boxes = [b.to(device) for b in layer_anchor_boxes]
            iou_scores = compute_iou(layer_anchor_boxes, target_box)

            is_anchor_match = iou_scores >= self.threshold

            found_match = found_match or torch.any(is_anchor_match)

            if torch.any(is_anchor_match):
                positive_student_outputs.append(student_scores[is_anchor_match])
                positive_teacher_outputs.append(teacher_scores[is_anchor_match])

        if found_match:
            positive_student_outputs = torch.cat(positive_student_outputs)
            positive_teacher_outputs = torch.cat(positive_teacher_outputs)
            return positive_student_outputs, positive_teacher_outputs
        else:
            return None, None

    def _get_target_box(self, target):
        x = self._get_target_x(target)
        y = self._get_target_y(target)
        width = self._get_target_width(target)
        height = self._get_target_height(target)
        return [y, x, height, width]

    def _get_target_image_index(self, target):
        index = self.target_format.index("i")
        return target[index].to(int)

    def _get_target_class(self, target):
        index = self.target_format.index("c")
        return target[index].to(int)

    def _get_target_x(self, target):
        index = self.target_format.index("x")
        return target[index] * self._get_image_width()

    def _get_target_y(self, target):
        index = self.target_format.index("y")
        return target[index] * self._get_image_height()

    def _get_target_width(self, target):
        index = self.target_format.index("w")
        return target[index] * self._get_image_width()

    def _get_target_height(self, target):
        index = self.target_format.index("h")
        return target[index] * self._get_image_height()

    def _get_image_width(self):
        index = self.anchor_format.index("w")
        return self.image_resolution[index]

    def _get_image_height(self):
        index = self.anchor_format.index("h")
        return self.image_resolution[index]

    def _get_anchor_box(self, layer):
        layer_width = self._get_layer_width(layer)
        layer_height = self._get_layer_height(layer)

        image_width = self._get_image_width()
        image_height = self._get_image_height()

        x = torch.arange(0, layer_width, dtype=torch.float32)
        x = (x + 0.5) * image_width / layer_width

        y = torch.arange(0, layer_height, dtype=torch.float32)
        y = (y + 0.5) * image_height / layer_height

        width = self._get_anchor_width(layer)
        height = self._get_anchor_height(layer)

        number_of_anchors = len(width)

        x = x.view((1, -1, 1)).repeat((layer_height, 1, number_of_anchors))
        y = y.view((-1, 1, 1)).repeat((1, layer_width, number_of_anchors))
        width = (
            torch.tensor(width).view((1, 1, -1)).repeat((layer_height, layer_width, 1))
        )
        height = (
            torch.tensor(height).view((1, 1, -1)).repeat((layer_height, layer_width, 1))
        )

        return [y, x, height, width]

    def _get_anchor_width(self, layer):
        index = self.anchor_format.index("w")
        stride = len(self.anchor_format)
        return self.anchors[layer][index::stride]

    def _get_anchor_height(self, layer):
        index = self.anchor_format.index("h")
        stride = len(self.anchor_format)
        return self.anchors[layer][index::stride]

    def _get_layer_width(self, layer):
        index = self.anchor_format.index("w")
        index += 2 * layer
        return self.layer_resolution[index]

    def _get_layer_height(self, layer):
        index = self.anchor_format.index("h")
        index += 2 * layer
        return self.layer_resolution[index]

    def _get_class_scores(self, output):
        output_dimension = self.output_format.index("o")
        _, class_scores = torch.split(
            output, (5, self.number_of_classes), output_dimension
        )
        return class_scores.softmax(output_dimension)

    def _get_image_output(self, output, image_index):
        batch_dimension = self.output_format.index("b")
        return torch.index_select(output, batch_dimension, image_index)

    def _filter_class(self, class_scores, class_index):
        output_dimension = self.output_format.index("o")
        return torch.index_select(class_scores, output_dimension, class_index)

    def _align_dimensions_to_anchor_boxes(self, scores):
        batch_dimension = self.output_format.index("b")
        anchor_dimension = self.output_format.index("a")
        y_dimension = self.output_format.index("y")
        x_dimension = self.output_format.index("x")
        output_dimension = self.output_format.index("o")

        # Permute dimensions such that it has format 'yxa'
        scores = torch.permute(
            scores,
            (
                y_dimension,
                x_dimension,
                anchor_dimension,
                batch_dimension,
                output_dimension,
            ),
        )
        scores = torch.squeeze(scores, 4)
        return torch.squeeze(scores, 3)
