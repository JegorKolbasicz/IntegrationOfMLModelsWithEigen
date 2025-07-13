import numpy as np
from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType



x = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(x, y)

x_test = np.array([[6], [7]])
y_pred = model.predict(x_test)

print(f'Predykcja dla x = 6, x = 7: {y_pred}')

print("Wspolczynnik (coef_):", model.coef_)
print("Wyraz wolny (intercept_):", model.intercept_)

initial_type = [('float_input', FloatTensorType([None, x.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("model/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
