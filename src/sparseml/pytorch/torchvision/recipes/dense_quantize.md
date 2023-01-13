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
num_epochs: 5

# Quantization Variables
quant_epochs: 5
quant_start_epoch: eval(num_epochs - quant_epochs)
quant_end_epoch: eval(num_epochs)
observe_epochs: 3 
quant_disable_observer_epoch: eval(quant_start_epoch + observe_epochs)
quant_freeze_bn_epoch: eval(quant_start_epoch + observe_epochs)
quant_lr_start_epoch: eval(quant_start_epoch)
quant_lr_end_epoch: eval(quant_end_epoch)
quant_lr_cycle_epochs: eval(quant_lr_end_epoch - quant_lr_start_epoch)
quant_init_lr: 0.01
quant_final_lr: 0.000001


# Modifiers
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)
  
quantization_modifiers:
  - !SetLearningRateModifier
    start_epoch: eval(quant_start_epoch)
    learning_rate: eval(quant_init_lr)

  - !LearningRateModifier
    start_epoch: eval(quant_start_epoch)
    lr_class: CosineAnnealingWarmRestarts
    lr_kwargs:
      lr_min: eval(quant_final_lr)
      cycle_epochs: eval(quant_lr_cycle_epochs)
    init_lr: eval(quant_init_lr)
    end_epoch: eval(quant_lr_end_epoch)
    
  - !QuantizationModifier
    start_epoch: eval(quant_start_epoch)
    submodules:
      - input
      - sections
      - classifier
    disable_quantization_observer_epoch: eval(quant_disable_observer_epoch)
    freeze_bn_stats_epoch: eval(quant_freeze_bn_epoch)


