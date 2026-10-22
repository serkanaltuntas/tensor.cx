"""Declared dtype contracts for built-in operations, not hardware promises.

The dtype is that of the data operand (weights for embedding). Shape, index,
scalar and other-operand rules still apply; see docs/TENSOR_API.md.
"""

_ALL = ("float32", "int32", "bool")
_NUMERIC = ("float32", "int32")
_FLOAT = ("float32",)
_BOOL = ("bool",)


def _profile(*, dlpack=False):
    result = {}
    for name in ("tensor", "empty", "copy", "astype", "reshape", "transpose",
                 "squeeze", "expand_dims", "indexing", "concat", "stack", "split",
                 "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
                 "where", "masked_select"):
        result[name] = _ALL
    for name in ("zeros", "ones"):
        result[name] = _ALL
    for name in ("add", "subtract", "multiply", "negative", "sum", "max"):
        result[name] = _NUMERIC
    for name in ("divide", "randn", "matmul", "mean", "exp", "gelu", "silu", "softmax",
                 "rmsnorm", "layernorm", "log", "sqrt", "linear", "embedding",
                 "scaled_dot_product_attention", "attention"):
        result[name] = _FLOAT
    for name in ("abs", "min", "argmax", "clip", "topk"):
        result[name] = _NUMERIC
    for name in ("logical_and", "logical_or", "logical_xor", "logical_not", "any", "all"):
        result[name] = _BOOL
    if dlpack:
        result["from_dlpack"] = _ALL
        result["__dlpack__"] = _ALL
    return result


CPU_OPERATIONS = _profile(dlpack=True)
METAL_OPERATIONS = _profile()
CUDA_OPERATIONS = _profile(dlpack=True)

# Initial synchronous Vulkan slice. No broad CPU/Metal capability inheritance.
VULKAN_OPERATIONS = {
    **{name: _ALL for name in (
        "tensor", "empty", "copy", "zeros", "ones", "reshape", "squeeze", "expand_dims",
        "astype", "transpose", "indexing", "concat", "stack", "split",
        "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
        "where", "masked_select"
    )},
    **{name: _NUMERIC for name in ("add", "subtract", "multiply", "negative", "sum", "max", "min", "argmax")},
    "divide": _FLOAT,
    "mean": _FLOAT,
    "matmul": _FLOAT,
    **{name: _FLOAT for name in ("exp", "gelu", "silu", "softmax", "rmsnorm", "layernorm",
        "linear", "embedding", "scaled_dot_product_attention", "attention")},
    **{name: _FLOAT for name in ("log", "sqrt")},
    **{name: _NUMERIC for name in ("abs", "clip", "topk")},
    "randn": _FLOAT,
    **{name: _BOOL for name in ("logical_and", "logical_or", "logical_xor", "logical_not", "any", "all")},
}
