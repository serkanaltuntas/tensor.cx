"""Inference-only PyTorch bridge. Importing tensorcx itself never imports torch."""
from typing import Optional

import torch
import tensorcx as cx


def _check(tensor: torch.Tensor, *, dtype=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('expected a torch.Tensor')
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise ValueError('torch-tensorcx requires contiguous strided tensors')
    if tensor.device.type not in ('cpu', 'cuda') or tensor.device.index not in (None, 0):
        raise ValueError('torch-tensorcx supports CPU and CUDA device 0')
    if tensor.dtype not in (torch.float32, torch.int32, torch.bool) or (dtype is not None and tensor.dtype != dtype):
        raise ValueError('unsupported torch-tensorcx dtype')
    if tensor.requires_grad:
        raise ValueError('torch-tensorcx is inference-only; explicitly detach inputs requiring gradients')
    if tensor.is_conj() or tensor.is_neg():
        raise ValueError('unresolved conjugate/negative views are unsupported')


def from_torch(tensor: torch.Tensor, *, copy=None) -> cx.Tensor:
    """Share supported PyTorch storage; copy=True explicitly isolates the result."""
    _check(tensor)
    return cx.from_dlpack(tensor, copy=copy)


def to_torch(tensor: cx.Tensor, *, copy=False) -> torch.Tensor:
    """Share a tensor.cx CPU/CUDA buffer, optionally producing an isolated copy."""
    if not isinstance(tensor, cx.Tensor):
        raise TypeError('expected a tensorcx.Tensor')
    if type(copy) is not bool:
        raise TypeError('copy must be a boolean')
    # Consume via the object protocol so CUDA consumer streams are negotiated.
    if copy:
        tensor = cx.from_dlpack(tensor, copy=True)
    return torch.utils.dlpack.from_dlpack(tensor)


def _linear_contract(input, weight, bias):
    for tensor in (input, weight) + (() if bias is None else (bias,)):
        _check(tensor, dtype=torch.float32)
        if tensor.device != input.device:
            raise ValueError('linear tensors must be on the same device')
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[1]:
        raise ValueError('linear requires (..., in_features) and (out_features, in_features)')
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError('linear bias must have shape (out_features,)')


@torch.library.custom_op('tensorcx::linear', mutates_args=())
def _linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    _linear_contract(input, weight, bias)
    result = cx.linear(from_torch(input), from_torch(weight), None if bias is None else from_torch(bias))
    return to_torch(result)


@_linear.register_fake
def _linear_fake(input, weight, bias=None):
    _linear_contract(input, weight, bias)
    return input.new_empty(input.shape[:-1] + (weight.shape[0],))


def linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Call tensor.cx's float32 linear as a registered, non-mutating PyTorch op.

    Inputs requiring gradients are rejected; use explicit detach for inference.
    This does not register a PyTorch device backend or an autograd formula.
    """
    # Validation belongs inside the opaque op and its fake implementation;
    # tracing storage/view queries here would break torch.compile capture.
    # PyTorch 2.4 queries storage before calling a custom op, so reject sparse
    # layouts here to retain the public error contract on that version too.
    for tensor in (input, weight) + (() if bias is None else (bias,)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('expected a torch.Tensor')
        if tensor.layout != torch.strided:
            raise ValueError('torch-tensorcx requires contiguous strided tensors')
    return _linear(input, weight, bias)


__all__ = ['from_torch', 'to_torch', 'linear']
