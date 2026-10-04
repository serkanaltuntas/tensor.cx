"""DLPack exchange for contiguous CPU/CUDA float32, int32 and bool arrays."""
from __future__ import annotations

import inspect
import operator

from . import _core


def _copy_option(copy):
    if copy is not None and type(copy) is not bool:
        raise TypeError('copy must be True, False or None')
    return copy


def _device_tuple(value):
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError('DLPack device must be a (device_type, device_id) tuple')
    result = []
    for item in value:
        if isinstance(item, bool):
            raise TypeError('DLPack device fields must be integers')
        result.append(operator.index(item))
    return tuple(result)


def device_of(tensor):
    if tensor.device == 'cpu':
        return (1, 0)
    if tensor.device == 'cuda':
        return (2, 0)
    raise BufferError('DLPack sharing supports CPU and CUDA device 0; Metal sharing is unavailable')


def export(tensor, *, stream=None, max_version=None, dl_device=None, copy=None):
    _copy_option(copy)
    device = device_of(tensor)
    if dl_device is not None and _device_tuple(dl_device) != device:
        raise BufferError('DLPack export does not transfer devices; use Tensor.to explicitly')
    if device[0] == 1:
        if stream is not None:
            raise BufferError('CPU DLPack export requires stream=None')
    elif stream is not None:
        if isinstance(stream, bool):
            raise TypeError('CUDA stream must be an integer or None')
        stream = operator.index(stream)
        if stream != -1 and stream < 1:
            raise ValueError('CUDA stream must be -1, 1, 2 or a nonzero stream pointer')
        if stream == -1 and copy is True:
            raise BufferError('CUDA copy=True requires synchronization; stream=-1 forbids it')
        if stream != 1 and copy is True:
            raise BufferError('CUDA copy=True supports only the legacy default stream')
    versioned = False
    if max_version is not None:
        major, minor = _device_tuple(max_version)
        if major < 0 or minor < 0:
            raise ValueError('max_version must contain nonnegative integers')
        versioned = (major, minor) >= (1, 0)
    # Every tensor.cx operation completes before returning. No export stream
    # work is needed; in particular stream=-1 never triggers synchronization.
    return _core._to_dlpack(tensor._impl, versioned, copy is True)


def from_dlpack(source, *, copy=None):
    """Share supported contiguous storage, or make an explicit same-device copy.

    Foreign writes are visible to every sharing view. Callers must synchronize
    later foreign GPU writes before tensor.cx uses them. Read-only versioned
    storage requires copy=True. Legacy capsules can be consumed once only.
    """
    from .tensor import Tensor
    _copy_option(copy)
    device = (-1, -1)
    if hasattr(source, '__dlpack__'):
        if not hasattr(source, '__dlpack_device__'):
            raise TypeError('DLPack producer must define __dlpack_device__')
        device = _device_tuple(source.__dlpack_device__())
        if device not in ((1, 0), (2, 0)):
            raise BufferError('DLPack import supports CPU and CUDA device 0 only')
        method = source.__dlpack__
        kwargs = {'stream': 1} if device[0] == 2 else {}
        # Python producers expose a signature, avoiding retries after a producer
        # raised TypeError internally. C builtins negotiate by unexpected keyword.
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            if 'max_version' in signature.parameters or any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
                kwargs['max_version'] = (1, 0)
            capsule = method(**kwargs)
        else:
            try:
                capsule = method(**kwargs, max_version=(1, 0))
            except TypeError as error:
                if 'max_version' not in str(error) or not any(s in str(error) for s in ('unexpected keyword', 'invalid keyword')):
                    raise
                capsule = method(**kwargs)
    else:
        capsule = source
    return Tensor(_core._from_dlpack(capsule, copy is True, copy is False, *device))
