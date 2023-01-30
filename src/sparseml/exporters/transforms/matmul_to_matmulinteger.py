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

from onnx import ModelProto, TensorProto
from sparseml.exporters.transforms import OnnxTransform
from sparseml.exporters.transforms.utils import (
    INITIALIZER_MATCH,
    MatchResult,
    add_quantized_conv_matmul_add_ops_no_bias,
    any_of,
    get_quantization_params,
    get_structural_matches,
)
from sparseml.onnx.utils import ONNXGraph


__all__ = ["MatMulToMatMulInteger"]


class MatMulToMatMulInteger(OnnxTransform):
    """
    A transform for converting a MatMul with kernel and no bias into a
    quantized representation

    ```
    |     weight (initializer)
    |         |
    |         Q
    |         |
    | input   Dq
    |   |     |
    |  Q/Dq   Transpose
    |     |   |
    |     MatMul
    ```
    (where `Q` is QuantizeLinear, and `Dq` is DequantizeLinear)
    into
    ```
    |   input
    |     |
    | MatMulInteger (with constant uint8 kernel)
    |     |
    | Cast (INT32 -> FP32)
    |     |
    | Mul (Rescale from bias scale)
    ```
    """

    def transform(self, model: ModelProto) -> ModelProto:
        graph = ONNXGraph(model)
        matches = get_structural_matches(
            graph,
            op_type="MatMul",
            parent_ops=[
                [any_of("QuantizeLinear", "DequantizeLinear")],
                [
                    # weight should be initializer
                    INITIALIZER_MATCH,
                    "QuantizeLinear",
                    "DequantizeLinear",
                    # "Transpose",
                ],
            ],
        )
        for match in matches:
            self.log_match(match)
            self._transform_match(graph, model, match)
        return model

    def _transform_match(
        self,
        graph: ONNXGraph,
        model: ModelProto,
        match: MatchResult,
    ):
        matmul = match.node
        (input_quant,) = match.parents[0]
        # weight_init, weight_quant, weight_dequant, transpose = match.parents[1]
        weight_init, weight_quant, weight_dequant = match.parents[1]

        input_quantize_params = get_quantization_params(
            model, input_quant, include_target=False
        )
        weight_quantize_params = get_quantization_params(
            model, weight_quant, include_target=True
        )
        # sanity check - matching handles this
        assert weight_quantize_params.target is not None

        add_quantized_conv_matmul_add_ops_no_bias(
            model=model,
            node=matmul,
            input_quantize_node=input_quant,
            weight_quantize_node=weight_quant,
            input_quantize_params=input_quantize_params,
            weight_quantize_params=weight_quantize_params,
            target_output=matmul.output[0],
            transpose_weight=False,
        )

        # Clean up
        self.delete_node_deferred(weight_dequant)
        self.delete_node_deferred(weight_quant)
        # self.delete_node_deferred(transpose)
        if len(graph.get_node_children(input_quant)) == 1:
            self.delete_node_deferred(input_quant)
        self.delete_node_deferred(matmul)
