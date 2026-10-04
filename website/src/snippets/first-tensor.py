import tensorcx as cx

device = cx.best_device()
x = cx.ones((1_000_000,), dtype=cx.float32, device=device)
y = cx.ones((1_000_000,), dtype=cx.float32, device=device)

z = x + y
print(z.cpu().numpy()[:5])
# [2. 2. 2. 2. 2.]
