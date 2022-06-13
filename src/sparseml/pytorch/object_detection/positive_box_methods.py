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
        threshold: float = 0.5,
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

    def __call__(self, student_outputs, teacher_outputs, targets):
        device = targets.device
        target_images = self._get_target_images(targets)
        target_classes = self._get_target_classes(targets)
        target_boxes = self._get_target_box(targets)

        positive_student_outputs = []
        positive_teacher_outputs = []
        for _ in range(target_images.numel()):
            positive_student_outputs.append([])
            positive_teacher_outputs.append([])

        found_match = False
        for layer in range(self.number_of_layers):
            with torch.no_grad():
                layer_anchor_boxes = self._get_anchor_box(layer)
                layer_anchor_boxes = [b.to(device) for b in layer_anchor_boxes]
                iou_scores = compute_iou(layer_anchor_boxes, target_boxes)

                is_anchor_match = iou_scores >= self.threshold

                found_match = found_match or torch.any(is_anchor_match)

            student_scores = self._get_class_scores(student_outputs[layer])
            student_scores = self._align_dimensions_to_anchor_boxes(student_scores)

            teacher_scores = self._get_class_scores(teacher_outputs[layer])
            teacher_scores = self._align_dimensions_to_anchor_boxes(teacher_scores)

            for target_index, (target_image, target_class, target_anchor_match) in enumerate(zip(target_images, target_classes, is_anchor_match)):
                if torch.any(target_anchor_match):
                    positive_student_outputs[target_index].append(student_scores[target_image][target_class][target_anchor_match])
                    positive_teacher_outputs[target_index].append(teacher_scores[target_image][target_class][target_anchor_match])

        if found_match:
            positive_student_outputs = [torch.cat(out) for out in positive_student_outputs if out]
            positive_teacher_outputs = [torch.cat(out) for out in positive_teacher_outputs if out]
            return positive_student_outputs, positive_teacher_outputs
        else:
            return None, None

    def _get_target_box(self, targets):
        y_index = self.target_format.index("y")
        x_index = self.target_format.index("x")
        height_index = self.target_format.index("h")
        width_index = self.target_format.index("w")

        indices = torch.tensor([y_index, x_index, height_index, width_index], dtype=torch.int32, device=targets.device)
        target_boxes = torch.index_select(targets, -1, indices)  # number of objects, 4
        image_height = self._get_image_height()
        image_width = self._get_image_width()
        scale = torch.tensor([image_height, image_width, image_height, image_width], dtype=torch.float32, device=targets.device).view(1, -1)
        target_boxes *= scale
        target_boxes = target_boxes.view((target_boxes.shape[0], 1, 1, -1)) # number of objects, number of vertical cells, number of horizontal cells, number of anchors, 4
        return torch.split(target_boxes, 1, dim=-1)

    def _get_target_images(self, targets):
        index = self.target_format.index("i")
        return torch.select(targets, -1, torch.tensor(index, dtype=torch.int32, device=targets.device)).to(int)

    def _get_target_classes(self, targets):
        index = self.target_format.index("c")
        return torch.select(targets, -1, torch.tensor(index, dtype=torch.int32, device=targets.device)).to(int)

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
        x = (x + 0.5) * image_width / layer_width # number of horizontal cells

        y = torch.arange(0, layer_height, dtype=torch.float32)
        y = (y + 0.5) * image_height / layer_height # number of vertical cells

        width = self._get_anchor_width(layer)   # number of anchors
        height = self._get_anchor_height(layer) # number of anchors

        x = x.view((1, 1, -1, 1))                         # number of objects, number of vertical cells, number of horizontal cells, number of anchors
        y = y.view((1, -1, 1, 1))                         # number of objects, number of vertical cells, number of horizontal cells, number of anchors
        width = torch.tensor(width).view((1, 1, 1, -1))   # number of objects, number of vertical cells, number of horizontal cells, number of anchors
        height = torch.tensor(height).view((1, 1, 1, -1)) # number of objects, number of vertical cells, number of horizontal cells, number of anchors

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

    def _align_dimensions_to_anchor_boxes(self, scores):
        batch_dimension = self.output_format.index("b")
        anchor_dimension = self.output_format.index("a")
        y_dimension = self.output_format.index("y")
        x_dimension = self.output_format.index("x")
        output_dimension = self.output_format.index("o")

        # Permute dimensions such that it has format 'boyxa'
        scores = torch.permute(
            scores,
            (
                batch_dimension,
                output_dimension,
                y_dimension,
                x_dimension,
                anchor_dimension,
            ),
        )
        return scores