import onnx

model = onnx.load("model_1.onnx")
input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in model.graph.input]
print('input shapes ->', input_shapes)
