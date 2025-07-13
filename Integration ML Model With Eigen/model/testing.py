import onnxruntime as rt 
import numpy as np

input = np.array([[8], [9]])

onnx_model = rt.InferenceSession("model/model.onnx")
input_name = onnx_model.get_inputs()[0].name
label_name = onnx_model.get_outputs()[0].name

predict = onnx_model.run([label_name], {input_name: input.astype(np.float32)})[0]
print(predict)