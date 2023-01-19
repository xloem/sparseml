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

import click
from sparseml.pytorch.yolov8.detection_trainer import SparseDetectionTrainer
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.v8.detect.train import DetectionTrainer


class SparseYOLO(YOLO):
    def __init__(self, model="yolov8n.yaml", type="v8") -> None:
        super().__init__(model, type)

        if self.TrainerClass == DetectionTrainer:
            self.TrainerClass = SparseDetectionTrainer


@click.command(context_settings=(dict(show_default=True)))
@click.option("--recipe", default=None)
@click.option("--recipe-args", default=None)
@click.option("--checkpoint-path", default=None)
def main(**kwargs):
    # TODO add `recipe`, `recipe-args`, `checkpoint-path`
    # NOTE: adapted from yolo repo
    kwargs["model"] = kwargs["model"] or "yolov8n.yaml"
    kwargs["data"] = kwargs["data"] or "coco128.yaml"
    model = SparseYOLO(kwargs["model"])
    model.train(**kwargs)


if __name__ == "__main__":
    main()
