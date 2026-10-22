"""Guarded Vulkan compiler: typed lowering, GPU numerics, ownership and cache."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import os
import struct
import subprocess
import sys

import numpy as np
import pytest
import tensorcx as cx
from tensorcx._compiler import cache, vulkan
from tensorcx.experimental import (IRAssign, IRBinaryOp, IRCall, IRCompare,
    IRConstant, IRIf, IRKernel, IRLoad, IRName, IRStore, KernelCompileError,
    _reference_execute)


@cx.experimental.kernel
def blend(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        delta = a[i] - b[i]
        scaled = delta * 0.5
        out[i] = scaled + b[i]


@cx.experimental.kernel
def cached_add(a, b, out, n):
    i = cx.experimental.program_id(0) * cx.experimental.block_size() + cx.experimental.thread_id()
    if i < n:
        out[i] = a[i] + b[i]


def require_device():
    required = bool(os.environ.get('TENSORCX_REQUIRE_MLIR_VULKAN'))
    try:
        if not cx.is_available('vulkan'): raise RuntimeError('Vulkan GPU unavailable')
        tools = vulkan.toolchain()
        return tools
    except (RuntimeError, OSError) as error:
        if required: pytest.fail(str(error))
        pytest.skip(str(error))


def require_tools():
    """Exercise real lowering on CPU-only CI without claiming GPU acceptance."""
    try:
        return vulkan.toolchain()
    except (RuntimeError, OSError) as error:
        if os.environ.get('TENSORCX_REQUIRE_MLIR_SPIRV'): pytest.fail(str(error))
        pytest.skip(str(error))


@pytest.fixture(autouse=True)
def validation_clean():
    yield
    if hasattr(cx._core, '_vulkan_validation_errors'):
        assert cx._core._vulkan_validation_errors() == 0


def ir_for(operation):
    index = IRName('i')
    first = IRLoad('a', index)
    binary = operation in vulkan.BINARY | vulkan.COMPARE
    params = ('a', 'b', 'out', 'n') if binary else ('a', 'out', 'n')
    if operation in {'add', 'sub', 'mul', 'div'}:
        expression = IRBinaryOp(operation, first, IRLoad('b', index))
    elif operation in vulkan.COMPARE:
        condition = IRCompare(operation, first, IRLoad('b', index))
        expression = IRCall('select', (condition, IRConstant(1.), IRConstant(-1.)))
    else:
        expression = IRCall(operation, (first, IRLoad('b', index)) if binary else (first,))
    return IRKernel('kernel_' + operation, params, (
        IRAssign('i', IRBinaryOp('add', IRBinaryOp('mul', IRCall('program_id', (IRConstant(0),)),
            IRCall('block_size', ())), IRCall('thread_id', ()))),
        IRIf(IRCompare('lt', index, IRName('n')), (IRAssign('value', expression), IRStore('out', index, IRName('value')))),
    ))


def reference(ir, arrays, count, block=256):
    private = {name: value.reshape(-1).copy() for name, value in arrays.items()}
    with np.errstate(all='ignore'):
        _reference_execute(ir, private, {'n': count}, count, block)
    return private['out'].reshape(arrays['out'].shape)


@pytest.mark.parametrize('operation', sorted(vulkan.EXPORTS))
def test_lower_to_spirv_without_gpu(monkeypatch, operation):
    tools = require_tools()
    def forbidden(*args, **kwargs): raise AssertionError('device access during standalone lowering')
    for name in ('_vulkan_kernel_support', '_load_vulkan_kernel'):
        monkeypatch.setattr(cx._core, name, forbidden, raising=False)
    ir = ir_for(operation)
    source, artifact = vulkan.lower(ir, tools=tools)
    header, binary = artifact.split(b'\n', 1)
    assert header.decode() == vulkan.manifest(*vulkan.signature(ir))
    assert 'spirv.module' in source and 'spirv.mlir.selection' in source
    assert [words[1] for opcode, words in vulkan.instructions(binary) if opcode == 17] == [1]
    assert struct.unpack('<I', binary[4:8])[0] == 0x00010300
    # lower() has already invoked the real Vulkan 1.1 SPIR-V validator.


def test_lower_to_spirv_without_clang(monkeypatch, tmp_path):
    tools = require_tools()
    for name in ('mlir-opt', 'mlir-translate'):
        (tmp_path/name).symlink_to(tools[name])
    monkeypatch.setenv('TENSORCX_LLVM_BIN', str(tmp_path))
    selected = vulkan.toolchain()
    assert 'clang' not in selected
    _, artifact = vulkan.lower(blend.parse_ir(), tools=selected)
    assert artifact.startswith(b'tensorcx.vulkan.v1|')


def compare(actual, expected, operation):
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    defined = ~np.isnan(expected)
    if operation in vulkan.COMPARE | {'add', 'sub', 'mul', 'div', 'abs', 'sqrt', 'rsqrt', 'minimum', 'maximum'}:
        np.testing.assert_array_equal(actual.view(np.uint32)[defined], expected.view(np.uint32)[defined])
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-6,
                                   atol=float(np.nextafter(np.float32(0), np.float32(1))), equal_nan=True)


@pytest.fixture(scope='module', params=sorted(vulkan.EXPORTS))
def operation(request):
    require_device()
    ir = ir_for(request.param)
    return request.param, ir, vulkan.compile_kernel(ir, cache=False)


@pytest.mark.parametrize('block', [1, 7, 64, 256])
def test_all_intrinsics_special_values_and_private_output(operation, block):
    name, ir, compiled = operation
    bits = np.array([0, 0x80000000, 1, 3, 0x007fffff, 0x00800000, 0x3f000001,
        0x3f800000, 0x3f800001, 0x42cfffff, 0x43000000, 0x43800000, 0x7f7fffff,
        0x7f800000, 0xff800000, 0x7fc12345, 0xff812345], np.uint32)
    bits = np.r_[bits, bits ^ np.uint32(0x80000000)]
    if 'b' in ir.parameters:
        arrays = {'a': np.repeat(bits.view(np.float32), len(bits)),
                  'b': np.tile(bits.view(np.float32), len(bits))}
    else: arrays = {'a': bits.view(np.float32)}
    count = arrays['a'].size
    arrays['out'] = np.full(count, -7, np.float32)
    # Sequence construction passes through Python floats and can quiet sNaNs.
    # Borrow raw CPU storage before the upload to test exact input preservation.
    args = {name: cx.from_dlpack(value, copy=False).to('vulkan') for name, value in arrays.items()}
    result = compiled.launch(*(count if name == 'n' else args[name] for name in ir.parameters), block_size=block)
    compare(result.numpy(), reference(ir, arrays, count, block), name)
    for parameter, original in arrays.items():
        np.testing.assert_array_equal(args[parameter].numpy().view(np.uint32), original.view(np.uint32))


@pytest.mark.parametrize('shape', [(), (0,), (2, 0, 3), (1,), (17,), (257,), (3, 5)])
@pytest.mark.parametrize('fraction', [0, .5, 1])
def test_public_launch_empty_partial_shape_and_aliases(shape, fraction):
    require_device()
    compiled = blend.compile(target='vulkan:0', compiler='mlir', cache=False)
    count = int(np.prod(shape)); threads = int(count * fraction)
    data = np.arange(count, dtype=np.float32).reshape(shape)
    left, right = cx.tensor(data), cx.ones(shape)
    a, b = left.to('vulkan'), right.to('vulkan')
    # The caller's output aliases an input; every unwritten bit must survive.
    result = compiled.launch(a, b, a, threads, thread_count=threads, block_size=7)
    expected = blend.reference(left, right, left, threads, thread_count=threads, block_size=7)
    np.testing.assert_array_equal(result.numpy(), expected.numpy())
    np.testing.assert_array_equal(a.numpy(), data)
    assert result.device == compiled.target == 'vulkan'


def test_constants_reordered_parameters_and_no_host_fallback(monkeypatch):
    require_device()
    original = ir_for('add')
    reordered = replace(original, parameters=('out', 'n', 'b', 'a'))
    module = vulkan.compile_kernel(reordered, cache=False)
    a = cx.tensor([1., 2., 3.], device='vulkan'); b = a + 1
    out = cx.tensor([11., 12., 13.], device='vulkan')
    def forbidden(*args, **kwargs): raise AssertionError('unexpected host data fallback')
    from tensorcx import backend
    route = backend.require_backend('vulkan')
    with monkeypatch.context() as patch:
        patch.setitem(backend._BACKENDS, 'vulkan', replace(route, copy_to_cpu=forbidden))
        result = module.launch(out, 2, b, a, thread_count=2)
    np.testing.assert_array_equal(result.numpy(), [3., 5., 13.])
    constant = replace(original, parameters=('out', 'n'), body=(original.body[0],
        replace(original.body[1], body=(IRStore('out', IRName('i'), IRConstant(-0.)),))))
    zero = vulkan.compile_kernel(constant, cache=False).launch(out, 3)
    np.testing.assert_array_equal(zero.numpy().view(np.uint32), np.full(3, 0x80000000, np.uint32))


def test_transcendental_composition_preserves_rounded_helper_semantics():
    require_device()
    original = ir_for('exp')
    for expression, separate in (
        (IRBinaryOp('sub', IRCall('exp', (IRLoad('a', IRName('i')),)), IRConstant(1.)),
         lambda data: cx.exp(data) - 1.),
        (IRCall('log', (IRCall('exp', (IRLoad('a', IRName('i')),)),)),
         lambda data: cx.log(cx.exp(data))),
    ):
        guard = replace(original.body[1], body=(IRStore('out', IRName('i'), expression),))
        ir = replace(original, body=(original.body[0], guard))
        values = np.linspace(-.01, .01, 4097, dtype=np.float32)
        data = cx.tensor(values, device='vulkan')
        result = vulkan.compile_kernel(ir, cache=False).launch(data, data, values.size, block_size=7)
        # Cancellation may amplify the intrinsic approximation relative to a
        # CPU libm composition; it must not change operation-by-operation rounding.
        np.testing.assert_array_equal(result.numpy().view(np.uint32), separate(data).numpy().view(np.uint32))
        np.testing.assert_array_equal(data.numpy(), values)


def test_launch_errors_leave_templates_and_device_usable():
    require_device()
    module = cached_add.compile(target='vulkan', compiler='mlir', cache=False)
    a = cx.ones((3,), device='vulkan')
    for args, kwargs in [((a, a, a, 2), {}), ((a, a, a, 4), {}),
        ((a, a, a, 3), {'block_size': 0}), ((a, a, a, 3), {'block_size': 2**31}),
        ((a, a.cpu(), a, 3), {}), ((a.astype(cx.int32), a, a, 3), {}),
        ((a, a[:1], a, 3), {}), ((a, a, a, True), {}), ((a, a, a), {})]:
        with pytest.raises((ValueError, TypeError)):
            module.launch(*args, **kwargs)
    np.testing.assert_array_equal(module.launch(a, a, a, 3).numpy(), [2., 2., 2.])
    np.testing.assert_array_equal(a.numpy(), [1., 1., 1.])


def test_threads_pool_and_device_timing():
    require_device()
    compiled = blend.compile(target='vulkan', compiler='mlir', cache=False)
    def forward(repeat):
        data = np.arange(257, dtype=np.float32) * np.float32(repeat * .25)
        a = cx.tensor(data, device='vulkan'); b = cx.ones((257,), device='vulkan')
        with cx.profile(device_timing=True) as profile:
            with cx.memory_pool('vulkan') as pool:
                for block in (1, 3, 7, 64, 256):
                    result = compiled.launch(a, b, a, 257, block_size=block)
                    np.testing.assert_array_equal(result.numpy(), (data - 1) * .5 + 1)
                assert pool.stats()['frame_hits'] > 0
        events = profile.report()['device_events']
        assert {e['name'] for e in events} >= {'vulkan_device_copy', 'vulkan_generated_kernel'}
        if cx.diagnostics()['backends']['vulkan']['timestamps']['supported']:
            assert profile.report()['device_complete'] and not profile.report()['device_errors']
        np.testing.assert_array_equal(a.numpy(), data)
        return result
    baseline = cx.diagnostics()['backends']['vulkan']['allocations']['physical_blocks']
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(forward, range(8)))
    del results; gc.collect()
    assert cx.diagnostics()['backends']['vulkan']['allocations']['physical_blocks'] == baseline


@pytest.mark.parametrize('change', ['index', 'output_read', 'unguarded', 'double_store', 'local_shadow', 'local_type', 'depth', 'operations', 'bad_constant'])
def test_invalid_ir_rejected_before_tools_or_device(monkeypatch, change):
    ir = ir_for('add'); guard = ir.body[1]; store = guard.body[-1]
    if change == 'index': ir = replace(ir, body=(replace(ir.body[0], value=IRConstant(0)), guard))
    elif change == 'output_read': ir = replace(ir, body=(ir.body[0], replace(guard, body=(replace(store, value=IRLoad('out', IRName('i'))),))))
    elif change == 'unguarded': ir = replace(ir, body=(ir.body[0], replace(guard, condition=IRCompare('lt', IRName('i'), IRConstant(1)))))
    elif change == 'double_store': ir = replace(ir, body=(ir.body[0], replace(guard, body=(store, store))))
    elif change == 'local_shadow': ir = replace(ir, body=(ir.body[0], replace(guard, body=(IRAssign('a', IRLoad('a', IRName('i'))), store))))
    elif change == 'local_type': ir = replace(ir, body=(ir.body[0], replace(guard, body=(IRAssign('value', IRCompare('lt', IRLoad('a', IRName('i')), IRLoad('b', IRName('i')))), store))))
    else:
        value = IRLoad('a', IRName('i'))
        if change == 'bad_constant': value = IRConstant(float('nan'))
        elif change == 'depth':
            for _ in range(18): value = IRBinaryOp('add', value, IRConstant(1.))
        else:
            # A shallow balanced tree exceeds the operation-count limit.
            for _ in range(7): value = IRBinaryOp('add', value, value)
        ir = replace(ir, body=(ir.body[0], replace(guard, body=(replace(store, value=value),))))
    def forbidden(*args, **kwargs): raise AssertionError('invalid IR reached device/toolchain')
    monkeypatch.setattr(vulkan, 'toolchain', forbidden)
    if hasattr(cx._core, '_vulkan_kernel_support'): monkeypatch.setattr(cx._core, '_vulkan_kernel_support', forbidden)
    with pytest.raises(KernelCompileError): vulkan.compile_kernel(ir, cache=False)


@pytest.mark.parametrize('target', ['vulkan:-1', 'vulkan:x', 'vulkan:', 'vulkan:2147483648'])
def test_invalid_target_indices(target):
    with pytest.raises(ValueError): cached_add.compile(target=target, compiler='mlir')


def test_cache_hit_reloads_native_module_and_restart(monkeypatch, tmp_path):
    require_device()
    path = tmp_path/'cache'
    monkeypatch.setenv('TENSORCX_DISABLE_CACHE', '0'); monkeypatch.setenv('TENSORCX_CACHE_DIR', str(path))
    first = cached_add.compile(target='vulkan', compiler='mlir')
    assert not first.cache_hit
    def forbidden(*args, **kwargs): raise AssertionError('warm hit recompiled device code')
    with monkeypatch.context() as patch:
        patch.setattr(vulkan, 'lower', forbidden)
        warm = cached_add.compile(target='vulkan', compiler='mlir')
        assert warm.cache_hit and warm._module is not first._module
    a = cx.ones((3,), device='vulkan')
    np.testing.assert_array_equal(warm.launch(a, a, a, 3).numpy(), [2., 2., 2.])
    script = tmp_path/'restart.py'
    script.write_text('''import tensorcx as cx
@cx.experimental.kernel
def cached_add(a,b,out,n):
    i=cx.experimental.program_id(0)*cx.experimental.block_size()+cx.experimental.thread_id()
    if i<n:
        out[i]=a[i]+b[i]
k=cached_add.compile(target="vulkan",compiler="mlir")
assert k.cache_hit
a=cx.ones((3,),device="vulkan")
assert k.launch(a,a,a,3).numpy().tolist()==[2.,2.,2.]
assert cx._core._vulkan_validation_errors()==0
print("fresh process cache hit and GPU execution passed")
''')
    child = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert 'fresh process cache hit' in child.stdout
    assert path.stat().st_mode & 0o777 == 0o700
    assert all(file.stat().st_mode & 0o777 == 0o600 for file in path.iterdir())
    # Checksum damage must become a real miss, never a loaded corrupt shader.
    artifact = next(path.glob('*.bin')); artifact.write_bytes(b'corrupt')
    assert not cached_add.compile(target='vulkan', compiler='mlir').cache_hit


def test_native_rejects_malformed_binary_before_driver():
    require_device()
    ir = ir_for('add'); _, artifact = vulkan.lower(ir)
    kinds, output, guard = vulkan.signature(ir)
    header, raw = artifact.split(b'\n', 1)
    for payload in (b'', artifact[:-1], b'wrong\n'+raw, header+b'\n'+b'\0'*20):
        with pytest.raises(ValueError): cx._core._load_vulkan_kernel(payload, kinds, output, guard, 0)
    words = list(struct.unpack('<'+'I'*(len(raw)//4), raw))
    mutations = [(0, 0), (1, 0x00010600), (3, 0), (4, 1), (5, 0)]
    offset = 5
    for opcode, instruction in vulkan.instructions(raw):
        if opcode == 17: mutations.append((offset+1, 5))
        if opcode == 16: mutations.append((offset+3, 64))
        if opcode == 71 and instruction[2] == 33: mutations.append((offset+3, 99))
        if opcode == 72 and instruction[3] == 35: mutations.append((offset+4, 999))
        offset += len(instruction)
    for position, replacement in mutations:
        changed = words.copy(); changed[position] = replacement
        bad = header + b'\n' + struct.pack('<'+'I'*len(changed), *changed)
        with pytest.raises(ValueError): cx._core._load_vulkan_kernel(bad, kinds, output, guard, 0)


def test_vulkan_diagnostics_ready_subset_and_opt_in(monkeypatch, tmp_path):
    import importlib
    diag = importlib.import_module('tensorcx.diagnostics')
    llvm, shader = tmp_path/'llvm', tmp_path/'shader'
    llvm.mkdir(); shader.mkdir()
    monkeypatch.setenv('TENSORCX_LLVM_BIN', str(llvm))
    monkeypatch.setenv('TENSORCX_VULKAN_BIN', str(shader))
    for folder, names in ((llvm, ('mlir-opt', 'mlir-translate')), (shader, vulkan.VULKAN_TOOLS)):
        for name in names:
            path = folder/name; path.write_text('#!/bin/sh\nexit 0\n'); path.chmod(0o755)
    monkeypatch.setattr(cx._core, '_vulkan_kernel_support', lambda index: None, raising=False)
    def forbidden(*args, **kwargs): raise AssertionError('unrequested compiler tool execution')
    monkeypatch.setattr(diag.subprocess, 'run', forbidden)
    unchecked = diag._llvm_info(False, vulkan_available=True)
    assert not unchecked['ready_targets']['vulkan']
    calls = []
    def query(command, **kwargs):
        assert kwargs['timeout'] == 3 and command[-1] == '--version'
        name = os.path.basename(command[0]); calls.append(name)
        version = ('LLVM version 21.1.8' if name.startswith('mlir-') else
                   'Glslang Version: 16:16.2.0' if name == 'glslangValidator' else 'SPIRV-Tools v2026.1')
        # Real SPIR-V Tools writes --version to stderr, unlike LLVM/glslang.
        return subprocess.CompletedProcess(command, 0,
            '' if name.startswith('spirv-') else version+'\n/private/secret',
            version+'\n/private/secret' if name.startswith('spirv-') else '')
    monkeypatch.setattr(diag.subprocess, 'run', query)
    report = diag._llvm_info(True, vulkan_available=True)
    assert report['status'] == 'missing'  # clang is intentionally not installed.
    assert report['ready_targets']['vulkan'] and not report['ready_targets']['cpu']
    assert set(calls) == {'mlir-opt', 'mlir-translate', *vulkan.VULKAN_TOOLS}
    assert '/private/secret' not in str(report)
    calls.clear(); report = diag._llvm_info(True, vulkan_available=False)
    assert not report['ready_targets']['vulkan'] and set(calls) == {'mlir-opt', 'mlir-translate'}


def test_cache_identity_includes_helper_implementation_and_concurrent_loads(monkeypatch, tmp_path):
    require_device()
    path = tmp_path/'cache'
    monkeypatch.setenv('TENSORCX_DISABLE_CACHE', '0'); monkeypatch.setenv('TENSORCX_CACHE_DIR', str(path))
    with ThreadPoolExecutor(max_workers=4) as workers:
        modules = list(workers.map(lambda _: cached_add.compile(target='vulkan', compiler='mlir'), range(4)))
    assert sorted(module.cache_hit for module in modules) == [False, True, True, True]
    assert len({id(module._module) for module in modules}) == 4
    with monkeypatch.context() as patch:
        patch.setattr(vulkan, 'SOURCE', vulkan.SOURCE + '\n// identity-only regression probe\n')
        assert not cached_add.compile(target='vulkan', compiler='mlir').cache_hit
    assert len(list(path.glob('*.bin'))) == 2
    # Explicit cache bypass must not create even an empty cache directory.
    bypass = tmp_path/'bypass'; monkeypatch.setenv('TENSORCX_CACHE_DIR', str(bypass))
    assert not cached_add.compile(target='vulkan', compiler='mlir', cache=False).cache_hit
    assert not bypass.exists()


def test_checksum_valid_cache_still_validates_shader(monkeypatch, tmp_path):
    import json
    require_device()
    root = tmp_path/'cache'
    monkeypatch.setenv('TENSORCX_DISABLE_CACHE', '0'); monkeypatch.setenv('TENSORCX_CACHE_DIR', str(root))
    good = cached_add.compile(target='vulkan', compiler='mlir')
    path = next(root.glob('*.bin')); content = path.read_bytes()
    header_line, payload = content[len(cache.MAGIC):].split(b'\n', 1)
    metadata = json.loads(header_line)
    split = payload.index(b'\n')+1
    damaged = payload[:split] + b'\0\0\0\0' + payload[split+4:]
    metadata.update(size=len(damaged), sha256=cache.digest(damaged))
    path.write_bytes(cache.MAGIC + json.dumps(metadata, sort_keys=True).encode() + b'\n' + damaged)
    def forbidden(*args, **kwargs): raise AssertionError('invalid cached shader reached driver or recompile')
    with monkeypatch.context() as patch:
        patch.setattr(vulkan, 'lower', forbidden)
        patch.setattr(cx._core, '_load_vulkan_kernel', forbidden)
        with pytest.raises(RuntimeError, match='MLIR tool failed'):
            cached_add.compile(target='vulkan', compiler='mlir')
    a = cx.ones((3,), device='vulkan')
    np.testing.assert_array_equal(good.launch(a,a,a,3).numpy(), [2.,2.,2.])
