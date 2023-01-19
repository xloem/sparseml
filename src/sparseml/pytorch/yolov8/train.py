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

import warnings
from copy import deepcopy
from datetime import datetime
from typing import Optional

import torch

import hydra
from omegaconf import OmegaConf
from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.utils.logger import (
    LoggerManager,
    PythonLogger,
    TensorBoardLogger,
    WANDBLogger,
)
from ultralytics import __version__
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.utils import DEFAULT_CONFIG, LOGGER
from ultralytics.yolo.utils.torch_utils import de_parallel
from ultralytics.yolo.v8.detect.train import DetectionTrainer


class SparseDetectionTrainer(DetectionTrainer):
    """
    Adds SparseML support to yolov8 DetectionTrainer. This works in the following way:

    1. Hooks into the checkpoint loading logic by overriding `resume_training()`
    2. Adds in manager loading at the end of `_setup_train()`.
        Note #1 is called at the end of `super()._setup_train()`
    3. Adds in manager to checkpoint saving with `save_model`
    4. Does proper deactivation of EMA & AMP with a callback
        that runs at the start of every epoch
    """

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        super().__init__(config, overrides)

        self.manager: Optional[ScheduledModifierManager] = None
        self.checkpoint_manager: Optional[ScheduledModifierManager] = None
        self.logger_manager: LoggerManager = LoggerManager(log_python=False)

        self.do_emulated_step = False

        self.add_callback("on_train_epoch_start", self.callback_on_train_epoch_start)
        self.add_callback("on_train_batch_start", self.callback_on_train_batch_start)
        self.add_callback("on_train_batch_end", self.callback_on_train_batch_end)
        self.add_callback("teardown", self.callback_teardown)

    def train(self):
        # TODO: REMOVE THIS OVERRIDE!
        # override to not support DDP yet - need to check if ddp is compatible
        # with this override
        self._do_train()

    def resume_training(self, ckpt):
        # NOTE: called at the end of `_setup_train`
        super().resume_training(ckpt)

        if ckpt is not None:
            # resume - set manager from checkpoint
            if "recipe" not in ckpt:
                raise ValueError("resume is set not checkpoint does not have recipe")
            self.manager = ScheduledModifierManager.from_yaml(ckpt["recipe"])
        elif self.args.checkpoint_path is not None:
            # previous checkpoint
            if self.args.recipe is not None:
                self.manager = ScheduledModifierManager.from_yaml(
                    self.args.recipe, recipe_variables=self.args.recipe_args
                )
            # TODO load checkpoint from this path
            self.checkpoint_manager = ...
        elif self.args.recipe is not None:
            # normal training
            self.manager = ScheduledModifierManager.from_yaml(
                self.args.recipe, recipe_variables=self.args.recipe_args
            )

    def _setup_train(self, rank, world_size):
        super()._setup_train(rank, world_size)

        if rank in {0, -1}:
            loggers = [
                PythonLogger(logger=LOGGER),
                TensorBoardLogger(log_path=str(self.save_dir / "sparseml_tb")),
            ]
            try:
                config = OmegaConf.to_object(self.args)
                if self.manager is not None:
                    config["manager"] = str(self.manager)
                loggers.append(WANDBLogger(init_kwargs=dict(config=config)))
            except ImportError:
                warnings.warn("Unable to import wandb for logging")
            self.logger_manager = LoggerManager(loggers)

        if self.manager is not None:
            self.manager.initialize(
                self.model, epoch=self.start_epoch, loggers=self.logger_manager
            )

            # NOTE: we intentionally don't divide number of batches by gradient
            # accumulation.
            # This is because yolov8 changes size of gradient accumulation during
            # warmup epochs, which is incompatible with SparseML managers
            # because they assume a static steps_per_epoch.
            # Instead, the manager will effectively ignore gradient accumulation,
            # and we will call self.scaler.emulated_step() if the batch was
            # accumulated.
            steps_per_epoch = len(self.train_loader)  # / self.accumulate

            self.scaler = self.manager.modify(
                self.model,
                self.optimizer,
                steps_per_epoch=steps_per_epoch,
                epoch=self.start_epoch,
                wrap_optim=self.scaler,
            )

        # TODO override LR schedulers

    def callback_on_train_epoch_start(self):
        # NOTE: this callback is registered in __init__
        if self.manager is not None and self.manager.qat_active(epoch=self.epoch):
            if self.scaler is not None:
                self.scaler._enabled = False
            self.ema = None

    def callback_on_train_batch_start(self):
        self.do_emulated_step = True

    def optimizer_step(self):
        super().optimizer_step()
        self.do_emulated_step = False

    def callback_on_train_batch_end(self):
        if self.do_emulated_step:
            self.scaler.emulated_step()

    def save_model(self):
        epoch = -1 if self.epoch == self.epochs - 1 else self.epoch

        # NOTE: identical to super().save_model() with the addition of recipe key
        if self.checkpoint_manager is not None:
            if epoch >= 0:
                epoch += self.checkpoint_manager.max_epochs
            manager = ScheduledModifierManager.compose_staged(
                self.checkpoint_manager, self.manager
            )
        else:
            manager = self.manager if self.manager is not None else None

        ckpt = {
            "epoch": epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model)).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "train_args": self.args,
            "date": datetime.now().isoformat(),
            "version": __version__,
        }

        if manager is not None:
            ckpt["recipe"] = str(manager)

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        del ckpt

    def callback_teardown(self):
        # NOTE: this callback is registered in __init__
        if self.manager is not None:
            self.manager.finalize()


class SparseYOLO(YOLO):
    def __init__(self, model="yolov8n.yaml", type="v8") -> None:
        super().__init__(model, type)

        if self.TrainerClass == DetectionTrainer:
            self.TrainerClass = SparseDetectionTrainer


@hydra.main(
    version_base=None,
    config_path=str(DEFAULT_CONFIG.parent),
    config_name=DEFAULT_CONFIG.name,
)
def train(cfg):
    # TODO add `recipe`, `recipe-args`, `checkpoint-path`
    # NOTE: adapted from yolo repo
    cfg.model = cfg.model or "yolov8n.yaml"
    cfg.data = cfg.data or "coco128.yaml"
    model = SparseYOLO(cfg.model)
    model.train(**cfg)


if __name__ == "__main__":
    train()
