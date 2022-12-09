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
import logging

from onnx import ModelProto, numpy_helper
import numpy
from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import get_structural_matches
from sparseml.onnx.utils import ONNXGraph


__all__ = ["PropagateMobilebertEmbeddingQuantization"]

_LOGGER = logging.getLogger(__name__)


class PropagateMobilebertEmbeddingQuantization(OnnxTransform):
    """
    Folds any `Identity` initializer node into the graph.

    | Starting with:
    |   INPUT   Identity (with initializer)
    |      |      |
    |      (SOME OP)
    |         |
    |       OUTPUT
    |
    | We end up converting to:
    |       INPUT
    |         |
    |     (SOME OP)
    |         |
    |      OUTPUT
    """

    def transform(self, model: ModelProto) -> ModelProto:
        count_converted_nodes = 0
        graph = ONNXGraph(model)
        for match in get_structural_matches(
            graph,
            children_ops=[["QuantizeLinear"]],
            op_type="ReLu",
        ):
            quantize_params = [
                get_quantization_params(model, quant_node) for quant_node in relu_children
            ]
            if any(params.zero_point != 0 for params in quantize_params):
                # skip if activation zero point does not match relu threshold of 0
                continue

            # set all child input nodes to the relu node input
            for quant_node in relu_children:
                quant_node.input[0] = relu_node.input[0]
            # delete relu node
            remove_node_and_params_from_graph(model, relu_node)

            _LOGGER.debug(f"Matched Identity node: {match.node.name}")

        if count_converted_nodes > 0:
            _LOGGER.info(f"Folded {count_converted_nodes} identity initializer nodes")
        return model

def _fold_relu_quants(model: ModelProto):
    # delete relu nodes that feed directly into quantize nodes with a zero point of 0

        quantize_params = [
            get_quantization_params(model, quant_node) for quant_node in relu_children
        ]
        if any(params.zero_point != 0 for params in quantize_params):
            # skip if activation zero point does not match relu threshold of 0
            continue

        # set all child input nodes to the relu node input
        for quant_node in relu_children:
            quant_node.input[0] = relu_node.input[0]
        # delete relu node
        remove_node_and_params_from_graph(model, relu_node)