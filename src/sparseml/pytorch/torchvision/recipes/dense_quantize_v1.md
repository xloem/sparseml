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
num_epochs: 5.0

init_lr: 1e-2
final_lr: 1e-8

weight_decay: 0.0

qat_start_epoch: 0.0
qat_observer_epochs: 1.0
qat_disable_observer_epoch: eval(qat_start_epoch + qat_observer_epochs)
qat_freeze_bn_epoch: eval(qat_start_epoch + qat_observer_epochs)

# Modifiers

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)

  - !SetLearningRateModifier
    start_epoch: 0.0
    end_epoch: eval(qat_disable_observer_epoch)
    learning_rate: eval(init_lr)

  - !LearningRateFunctionModifier
      start_epoch: eval(qat_disable_observer_epoch)
      end_epoch: eval(num_epochs)
      lr_func: cosine
      init_lr: eval(init_lr)
      final_lr: eval(final_lr)

quantization_modifiers:
  - !QuantizationModifier
      start_epoch: 0.0
      submodules:
        - input
        - sections
        - classifier
      disable_quantization_observer_epoch: eval(qat_disable_observer_epoch)
      freeze_bn_stats_epoch: eval(qat_freeze_bn_epoch)

regularization_modifiers:
  - !SetWeightDecayModifier
      start_epoch: 0
      weight_decay: eval(weight_decay)

---
