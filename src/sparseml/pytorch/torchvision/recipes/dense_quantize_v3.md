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
num_epochs: 10.0

lr_warmup_epochs: 0.01

warmup_lr: 2e-4
min_lr: 1e-8

weight_decay: 0.00001

qat_start_epoch: eval(lr_warmup_epochs)
qat_observer_epochs: 1.0
qat_disable_observer_epoch: eval(qat_start_epoch + qat_observer_epochs)

qat_bn_stats_epochs: 2.0
qat_freeze_bn_epoch: eval(qat_start_epoch + qat_bn_stats_epochs)

quant_conv_act: 0

# Modifiers

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)

  - !SetLearningRateModifier
    start_epoch: 0.0
    end_epoch: eval(qat_start_epoch)
    learning_rate: eval(warmup_lr)

  - !LearningRateFunctionModifier
    start_epoch: eval(qat_start_epoch)
    end_epoch: eval(num_epochs)
    lr_func: cosine
    init_lr: eval(warmup_lr)
    final_lr: eval(min_lr)

quantization_modifiers:
  - !QuantizationModifier
      start_epoch: eval(qat_start_epoch)
      submodules:
        - input
        - sections
      disable_quantization_observer_epoch: eval(qat_disable_observer_epoch)
      freeze_bn_stats_epoch: eval(qat_freeze_bn_epoch)
      quantize_conv_activations: eval(quant_conv_act)

regularization_modifiers:
  - !SetWeightDecayModifier
      start_epoch: 0
      weight_decay: eval(weight_decay)

---

Eventually need to use these in QuantizationModifier:
      custom_quantizable_module_types: [‘SiLU’, ‘Sigmoid’]
      exclude_module_types: [‘SiLU’, ‘Sigmoid’]
