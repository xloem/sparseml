import onnx

from sparseml.exporters.transforms import PropagateMobilebertEmbeddingQuantization

def _create_test_model(is_zero_point_zero= False):
    if is_zero_point_zero:
    zero_point_val = 0 if is_zero_point_zero
    else:
        zero_point_val = 1

    input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, (1,))
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, (1,))

    scale = onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, (1,), [1])
    zero_point = onnx.helper.make_tensor("zero_point", onnx.TensorProto.UINT8, (1,), [0])

    relu_node = onnx.helper.make_node("Relu", ["input"], ["relu_output"])
    dequantize_node = onnx.helper.make_node("DequantizeLinear", ["relu_output", "scale", "zero_point"], ["output"], name="dequantize")

    graph = onnx.helper.make_graph(
        [relu_node, dequantize_node]
        "test_graph",
        [input],
        [output],
        [scale, zero_point])

    model = onnx.helper.make_model(graph)
    return model

def test_propagate_mobilebert_embedding_quantization():
    model = _create_test_model()
    transform = PropagateMobilebertEmbeddingQuantization()
    model = transform(model)
    assert len(model.graph.node) == 1
    assert model.graph.node[0].name == "dequantize"
    assert model.graph.node[0].input[0] == "gather_data"





