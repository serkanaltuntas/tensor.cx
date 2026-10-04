import cortex_runtime as cx

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
