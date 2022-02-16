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
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: 200
  
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: 5
    lr_func: linear
    init_lr: 0.0512
    final_lr: 0.256

  - !LearningRateFunctionModifier
    start_epoch: 5
    end_epoch: 200
    lr_func: cosine
    init_lr: 0.256
    final_lr: 0.0
  
pruning_modifiers:
  - !ACDCPruningModifier
    compression_sparsity: 0.8
    start_epoch: 5
    end_epoch: 180
    update_frequency: 5
    params: ['re:.*conv*'] 
    global_sparsity: True
  

  - !GMPruningModifier
     init_sparsity: 0.8
     final_sparsity: 0.8
     start_epoch: 185
     end_epoch: 200
     update_frequency: 1
     params: ['re:.*conv*'] 
     global_sparsity: True

---

