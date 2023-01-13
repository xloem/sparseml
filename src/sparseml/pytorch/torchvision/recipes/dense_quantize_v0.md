<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

---
version: 1.1.0

# General Variables
num_epochs: 3.0
lr_warmup_epochs: 1.0
init_lr: 3e-5
final_lr: 1e-8
weight_decay: 0.0

quant_start_epoch: 0.0
quant_disable_observer_epoch: 2.0
quant_freeze_bn_epoch: 2.0

# Modifiers

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)

  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: eval(lr_warmup_epochs)
    lr_func: linear
    init_lr: eval(final_lr)
    final_lr: eval(init_lr)

  - !LearningRateFunctionModifier
    start_epoch: eval(lr_warmup_epochs)
    end_epoch: eval(num_epochs)
    lr_func: linear
    init_lr: eval(init_lr)
    final_lr: eval(final_lr)

quantization_modifiers:
  - !QuantizationModifier
    start_epoch: 0.0
    submodules:
      - input
      - sections
    disable_quantization_observer_epoch: 2.0
    freeze_bn_stats_epoch: 2.0

regularization_modifiers:
  - !SetWeightDecayModifier
    start_epoch: 0
    weight_decay: eval(weight_decay)

---

quantization_modifier:
  - !QuantizationModifier
    start_epoch: 0.0
    submodules:
      - input
      - sections
    disable_quantization_observer_epoch: eval(quant_disable_observer_epoch)
    freeze_bn_stats_epoch: eval(quant_freeze_bn_epoch)
