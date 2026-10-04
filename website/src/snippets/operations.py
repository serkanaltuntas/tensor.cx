import tensorcx as cx

a = cx.tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
b = cx.ones((2, 2), dtype=cx.float32, device="cpu")

product = cx.matmul(a, b)
print(product.numpy())
# [[3. 3.]
#  [7. 7.]]

print(cx.sum(a, axis=1).numpy())
# [3. 7.]

print(cx.softmax(a, axis=-1).shape)
# (2, 2)

print(((a - 1) / 2).reshape((4,)).numpy())
# [0.  0.5 1.  1.5]

print(a.sum(axis=1, keepdims=True).shape)
# (2, 1)

values = cx.tensor([[1, 2, 3], [4, 5, 6]])
bias = cx.tensor([0.25, 0.5, 0.75])
result = values.astype(cx.float32) + bias
print((result - result.mean(axis=-1, keepdims=True)).numpy())
# [[-1.25  0.    1.25]
#  [-1.25  0.    1.25]]

print(values.T.numpy())
# [[1 4]
#  [2 5]
#  [3 6]]

expanded = values.expand_dims((0, -1))
print(expanded.shape)  # (1, 2, 3, 1)
print(expanded.squeeze().shape)  # (2, 3)

print(a.sum().numpy())  # 10.0
print(a.mean(axis=(0, 1), keepdims=True).numpy())  # [[2.5]]

print(values[:, ::-1].numpy())
# [[3 2 1]
#  [6 5 4]]
print(cx.concat([values, values], axis=0).shape)  # (4, 3)
print(cx.stack([values, values], axis=1).shape)  # (2, 2, 3)
parts = values.split([1], axis=1)
print([part.shape for part in parts])  # [(2, 1), (2, 2)]

mask = values > 3
print(mask.dtype)  # bool
print(values[mask].numpy())  # [4 5 6]
print(cx.where(mask, values, 0).numpy())
# [[0 0 0]
#  [4 5 6]]
print(mask.any(axis=1).numpy())  # [False  True]
