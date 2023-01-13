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
num_epochs: 3
lr_warmup_epochs: 1
init_lr: 0.0
warmup_lr: 1e-5
final_lr: 0.0
weight_decay: 0.00001

# Modifiers

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)

  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: eval(lr_warmup_epochs)
    lr_func: linear
    init_lr: eval(init_lr)
    final_lr: eval(warmup_lr)

  - !LearningRateFunctionModifier
    start_epoch: eval(lr_warmup_epochs)
    end_epoch: eval(num_epochs)
    lr_func: cosine
    init_lr: eval(warmup_lr)
    final_lr: eval(final_lr)

  - !SetWeightDecayModifier
    start_epoch: 0
    weight_decay: eval(weight_decay)

---
