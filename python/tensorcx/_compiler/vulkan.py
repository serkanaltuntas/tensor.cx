"""Optional guarded float32 IR -> typed SPIR-V MLIR -> linked Vulkan shader."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import struct
import tempfile

from .. import _core
from ..experimental import (CompiledVulkanKernel, IRAssign, IRBinaryOp, IRCall,
    IRCompare, IRConstant, IRIf, IRLoad, IRName, IRStore, KernelCompileError)
from .cpu import _signature, _run, toolchain as llvm_toolchain
from .emitter import format_f32_constant
from .cache import artifact_cache, cache_enabled, descriptor, digest
from ._vulkan_helpers import SOURCE

ENTRY = "tensorcx_kernel_v1"
BINARY = {"add", "sub", "mul", "div", "minimum", "maximum"}
UNARY = {"sqrt", "abs", "exp", "log", "rsqrt", "tanh", "sigmoid", "silu"}
COMPARE = {"lt", "lte", "gt", "gte", "eq", "neq"}
EXPORTS = BINARY | UNARY | COMPARE
VULKAN_TOOLS = ("glslangValidator", "spirv-opt", "spirv-link", "spirv-val")
OPT_FLAGS = ["--eliminate-dead-functions", "--eliminate-dead-variables",
             "--eliminate-dead-const", "--remove-duplicates"]


def signature(ir):
    """Prove bounded expressions and canonical accesses before recursive passes."""
    def reject(message):
        raise KernelCompileError("MLIR Vulkan expression: " + message)
    if (len(ir.body) != 2 or not isinstance(ir.body[0], IRAssign)
            or not isinstance(ir.body[1], IRIf)):
        reject("requires one canonical index and guard")
    if not 2 <= len(ir.parameters) <= 4 or len(set(ir.parameters)) != len(ir.parameters):
        reject("requires up to two inputs, one output and one uint32 bound")
    body = ir.body[1].body
    if not 1 <= len(body) <= 33 or not isinstance(body[-1], IRStore):
        reject("requires at most 32 locals and one final store")
    index = ir.body[0].target
    output = body[-1].buffer
    if index in ir.parameters: reject("canonical index cannot shadow a parameter")
    locals_ = set()
    operations = 0
    for statement in body:
        if statement is not body[-1]:
            if (not isinstance(statement, IRAssign)
                    or statement.target in locals_ | set(ir.parameters) | {index}):
                reject("only unique local assignments may precede the store")
        pending = [(statement.value, 1)]
        while pending:
            value, depth = pending.pop()
            if depth > 16: reject("expression depth exceeds 16")
            if isinstance(value, IRLoad):
                if value.buffer == output or value.index != IRName(index):
                    reject("input loads require the canonical index; output reads are unsupported")
            elif isinstance(value, IRName):
                if value.name not in locals_: reject("undefined local")
            elif isinstance(value, IRConstant):
                if type(value.value) is not float: reject("arithmetic literals must be float32")
                try: format_f32_constant(value.value)
                except ValueError as error: reject(str(error))
            else:
                if isinstance(value, IRBinaryOp) and value.op in BINARY:
                    children = (value.lhs, value.rhs)
                elif isinstance(value, IRCompare) and value.op in COMPARE:
                    children = (value.lhs, value.rhs)
                elif (isinstance(value, IRCall) and
                      ((value.name in UNARY and len(value.args) == 1)
                       or (value.name in {"minimum", "maximum"} and len(value.args) == 2)
                       or (value.name == "select" and len(value.args) == 3))):
                    children = value.args
                else: reject("unsupported expression")
                operations += 1
                if operations > 64: reject("expression operation count exceeds 64")
                pending.extend((child, depth + 1) for child in children)
        if isinstance(statement, IRAssign): locals_.add(statement.target)
    if body[-1].index != IRName(index): reject("store requires the canonical index")
    try: kinds, out, guard = _signature(ir)
    except KernelCompileError as error: reject(str(error))
    if kinds.count("u") != 1 or not 1 <= kinds.count("t") <= 3:
        reject("requires up to three tensors and one uint32 bound")
    return kinds, out, guard


def manifest(kinds, output, guard):
    return f"tensorcx.vulkan.v1|spirv13|vulkan11|{kinds}|{output}|{guard}|canonical-prefix-v1"


def tool_version(name, text):
    pattern = r'Glslang Version:\s+\d+:([\d.]+)' if name == 'glslangValidator' else r'SPIRV-Tools v(\d{4}\.\d+(?:\.\d+)?)'
    match = re.search(pattern, text)
    return match[1] if match else None


def toolchain():
    tools = llvm_toolchain(names=("mlir-opt", "mlir-translate"))
    explicit = os.environ.get("TENSORCX_VULKAN_BIN")
    if explicit == "": raise RuntimeError("TENSORCX_VULKAN_BIN must name a tool directory")
    for name in VULKAN_TOOLS:
        path = str(Path(explicit) / name) if explicit is not None else shutil.which(name)
        if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"MLIR Vulkan requires executable {name}; set TENSORCX_VULKAN_BIN")
        tools[name] = str(Path(path).resolve())
        if not tool_version(name, _run([tools[name], "--version"], include_stderr=True)):
            raise RuntimeError(f'MLIR Vulkan executable {name} has an unrecognized version')
    return tools


def instructions(data):
    if len(data) < 20 or len(data) % 4: raise ValueError("invalid SPIR-V byte size")
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    if words[0] != 0x07230203: raise ValueError("invalid SPIR-V magic")
    position = 5
    while position < len(words):
        count = words[position] >> 16
        if not count or position + count > len(words): raise ValueError("invalid SPIR-V instruction")
        yield words[position] & 65535, list(words[position:position + count])
        position += count


def _export_library(data):
    """Add named function exports to the compiler's own validated GLSL module.

    The real kernel body is emitted in MLIR. This library supplies only strict
    numerical helpers; no GLSL kernel body is substituted for the user's IR.
    """
    code = list(instructions(data))
    names = {}
    for opcode, words in code:
        if opcode == 5:
            text = struct.pack('<'+'I'*(len(words)-2), *words[2:]).split(b'\0', 1)[0].decode()
            name = text.split('(', 1)[0]
            if name.startswith('tensorcx_') and name[9:] in EXPORTS:
                if name in names: raise ValueError("duplicate Vulkan helper symbol")
                names[name] = words[1]
    if set(names) != {'tensorcx_' + name for name in EXPORTS}:
        raise ValueError("Vulkan helper export signature mismatch")
    decorations = []
    for name, identifier in sorted(names.items()):
        text = name.encode() + b'\0'; text += b'\0' * (-len(text) % 4)
        text_words = struct.unpack('<'+'I'*(len(text)//4), text)
        decorations += [((len(text_words) + 4) << 16) | 71, identifier, 41, *text_words, 0]
    result = list(struct.unpack('<5I', data[:20])) + [(2 << 16) | 17, 5]  # Linkage capability.
    inserted = False
    for opcode, words in code:
        if opcode in (15, 16): continue  # Remove the library-only dummy entry/mode.
        # Pure helpers never use workgroup geometry. Do not retain the dummy
        # shader's fixed WorkgroupSize decoration in the linked kernel.
        if opcode == 71 and words[2:] == [11, 25]: continue
        if not inserted and 19 <= opcode <= 39:
            result += decorations; inserted = True
        result += words
    if not inserted: raise ValueError("Vulkan helper module has no type section")
    return struct.pack('<'+'I'*len(result), *result)


def _library(root, tools):
    calls = []
    for name in sorted(EXPORTS):
        args = 'data[1]' if name in UNARY else 'data[1],data[2]'
        value = f'tensorcx_{name}({args})'
        if name in COMPARE: value = f'uint({value})'
        calls.append(f'data[0] ^= {value};')
    shader = ('#version 450\nlayout(local_size_x=1) in;\n'
              'layout(set=0,binding=0,std430) buffer Data { uint data[]; };\n'
              + SOURCE + '\nvoid main() {\n' + '\n'.join(calls) + '\n}\n')
    (root/'helpers.comp').write_text(shader)
    _run([tools['glslangValidator'], '-V', '--target-env', 'vulkan1.1', '-S', 'comp',
          str(root/'helpers.comp'), '-o', str(root/'helpers.spv')])
    _run([tools['spirv-val'], '--target-env', 'vulkan1.1', str(root/'helpers.spv')])
    (root/'exports.spv').write_bytes(_export_library((root/'helpers.spv').read_bytes()))
    _run([tools['spirv-opt'], *OPT_FLAGS, str(root/'exports.spv'), '-o', str(root/'library.spv')])
    _run([tools['spirv-val'], '--target-env', 'spv1.3', str(root/'library.spv')])
    return root/'library.spv'


BUFFER = '!spirv.ptr<!spirv.struct<Buffer, (!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>'
PUSH = '!spirv.ptr<!spirv.struct<Config, (i32 [0], i32 [4], i32 [8], i32 [12], i32 [16])>, PushConstant>'
POINTER = '!spirv.ptr<i32, Function>'


def emit_vulkan(ir):
    kinds, output, guard = signature(ir)
    slots = {ir.parameters[i]: sum(kind == 't' for kind in kinds[:i])
             for i, kind in enumerate(kinds) if kind == 't'}
    lines = []; serial = 0; used = set(); locals_ = {}
    def emit(text, indent='      '): lines.append(indent + text)
    def value(text):
        nonlocal serial
        serial += 1; name = f'%v{serial}'; emit(name + ' = ' + text); return name
    def expression(node):
        if isinstance(node, IRName): return locals_[node.name]
        if isinstance(node, IRConstant):
            bits = struct.unpack('<i', struct.pack('<f', node.value))[0]
            return value(f'spirv.Constant {bits} : i32')
        if isinstance(node, IRLoad):
            address = value(f'spirv.mlir.addressof @b{slots[node.buffer]} : {BUFFER}')
            pointer = value(f'spirv.AccessChain {address}[%zero, %index] : {BUFFER}, i32, i32 -> !spirv.ptr<i32, StorageBuffer>')
            return value(f'spirv.Load "StorageBuffer" {pointer} : i32')
        if isinstance(node, IRCall) and node.name == 'select':
            condition, left, right = [expression(child) for child in node.args]
            return value(f'spirv.Select {condition}, {left}, {right} : i1, i32')
        name = node.name if isinstance(node, IRCall) else node.op
        children = node.args if isinstance(node, IRCall) else (node.lhs, node.rhs)
        arguments = [expression(child) for child in children]
        used.add(name)
        for i, argument in enumerate(arguments): emit(f'spirv.Store "Function" %scratch{i}, {argument} : i32')
        types = ', '.join([POINTER] * len(arguments))
        pointers = ', '.join(f'%scratch{i}' for i in range(len(arguments)))
        return value(f'spirv.FunctionCall @tensorcx_{name}({pointers}) : ({types}) -> {"i1" if name in COMPARE else "i32"}')
    for statement in ir.body[1].body:
        result = expression(statement.value)
        if isinstance(statement, IRAssign): locals_[statement.target] = result
        else:
            address = value(f'spirv.mlir.addressof @b{slots[statement.buffer]} : {BUFFER}')
            pointer = value(f'spirv.AccessChain {address}[%zero, %index] : {BUFFER}, i32, i32 -> !spirv.ptr<i32, StorageBuffer>')
            emit(f'spirv.Store "StorageBuffer" {pointer}, {result} : i32')
    header = ['spirv.module Logical GLSL450 requires #spirv.vce<v1.3, ['
              + ('Shader, Linkage' if used else 'Shader') + '], []> {',
              f'  spirv.GlobalVariable @config : {PUSH}',
              '  spirv.GlobalVariable @group built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>',
              '  spirv.GlobalVariable @lane built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>']
    for parameter, slot in slots.items():
        access = '' if parameter == ir.parameters[output] else ' {non_writable}'
        header.append(f'  spirv.GlobalVariable @b{slot} bind(0, {slot}){access} : {BUFFER}')
    for name in sorted(used):
        count = 1 if name in UNARY else 2
        args = ', '.join(f'%a{i}: {POINTER}' for i in range(count))
        header.append(f'  spirv.func @tensorcx_{name}({args}) -> {"i1" if name in COMPARE else "i32"} "None" attributes {{linkage_attributes=#spirv.linkage_attributes<linkage_name="tensorcx_{name}", linkage_type=<Import>>}}')
    begin = f'''  spirv.func @{ENTRY}() "None" {{
    %scratch0 = spirv.Variable : {POINTER}
    %scratch1 = spirv.Variable : {POINTER}
    %zero = spirv.Constant 0 : i32
    %one = spirv.Constant 1 : i32
    %two = spirv.Constant 2 : i32
    %four = spirv.Constant 4 : i32
    %p = spirv.mlir.addressof @config : {PUSH}
    %np = spirv.AccessChain %p[%zero] : {PUSH}, i32 -> !spirv.ptr<i32, PushConstant>
    %bp = spirv.AccessChain %p[%two] : {PUSH}, i32 -> !spirv.ptr<i32, PushConstant>
    %xp = spirv.AccessChain %p[%four] : {PUSH}, i32 -> !spirv.ptr<i32, PushConstant>
    %n = spirv.Load "PushConstant" %np : i32
    %block = spirv.Load "PushConstant" %bp : i32
    %gx = spirv.Load "PushConstant" %xp : i32
    %gp = spirv.mlir.addressof @group : !spirv.ptr<vector<3xi32>, Input>
    %lp = spirv.mlir.addressof @lane : !spirv.ptr<vector<3xi32>, Input>
    %gxp = spirv.AccessChain %gp[%zero] : !spirv.ptr<vector<3xi32>, Input>, i32 -> !spirv.ptr<i32, Input>
    %gyp = spirv.AccessChain %gp[%one] : !spirv.ptr<vector<3xi32>, Input>, i32 -> !spirv.ptr<i32, Input>
    %lxp = spirv.AccessChain %lp[%zero] : !spirv.ptr<vector<3xi32>, Input>, i32 -> !spirv.ptr<i32, Input>
    %x = spirv.Load "Input" %gxp : i32
    %y = spirv.Load "Input" %gyp : i32
    %lane = spirv.Load "Input" %lxp : i32
    %row = spirv.IMul %y, %gx : i32
    %group = spirv.IAdd %row, %x : i32
    %base = spirv.IMul %group, %block : i32
    %index = spirv.IAdd %base, %lane : i32
    %active = spirv.ULessThan %index, %n : i32
    spirv.mlir.selection {{
      spirv.BranchConditional %active, ^run, ^merge
    ^run:
'''
    end = f'''      spirv.Branch ^merge
    ^merge:
      spirv.mlir.merge
    }}
    spirv.Return
  }}
  spirv.EntryPoint "GLCompute" @{ENTRY}, @group, @lane
  spirv.ExecutionMode @{ENTRY} "LocalSize", 1, 1, 1
}}
'''
    return '\n'.join(header) + '\n' + begin + '\n'.join(lines) + '\n' + end


def lower(ir, *, tools=None):
    kinds, output, guard = signature(ir)
    source = emit_vulkan(ir); tools = toolchain() if tools is None else tools
    with tempfile.TemporaryDirectory(prefix='tensorcx-vulkan-') as directory:
        root = Path(directory); (root/'kernel.mlir').write_text(source)
        # MLIR verifies the typed resource/control/function interface before
        # serialization. Numerical helper bodies are linked as a real library.
        _run([tools['mlir-opt'], '--no-implicit-module', str(root/'kernel.mlir'), '-o', str(root/'checked.mlir')])
        _run([tools['mlir-translate'], '--serialize-spirv', '--no-implicit-module', str(root/'checked.mlir'), '-o', str(root/'kernel.spv')])
        library = _library(root, tools)
        _run([tools['spirv-link'], '--target-env', 'spv1.3', str(root/'kernel.spv'), str(library), '-o', str(root/'linked.spv')])
        _run([tools['spirv-opt'], *OPT_FLAGS, str(root/'linked.spv'), '-o', str(root/'final.spv')])
        _run([tools['spirv-val'], '--target-env', 'vulkan1.1', str(root/'final.spv')])
        payload = (manifest(kinds, output, guard) + '\n').encode() + (root/'final.spv').read_bytes()
    return source, payload


def compile_kernel(ir, *, target='vulkan', cache=True):
    kinds, output, guard = signature(ir)
    if not hasattr(_core, '_vulkan_kernel_support'):
        raise RuntimeError('MLIR Vulkan requires a Vulkan-enabled tensor.cx build')
    index = int(target.partition(':')[2] or 0)
    _core._vulkan_kernel_support(index)
    source = emit_vulkan(ir); tools = toolchain()
    options = {'helper_source': digest(SOURCE.encode()), 'opt': OPT_FLAGS,
               'spirv_target': 'spv1.3', 'validation_target': 'vulkan1.1'}
    description = descriptor('vulkan', source, tools=tools, options=options,
                             abi=manifest(kinds, output, guard)) if cache_enabled(cache) else {}
    with artifact_cache(description, enabled=cache) as entry:
        payload = entry.read(); hit = payload is not None
        if not hit: _, payload = lower(ir, tools=tools)
        else:
            expected = (manifest(kinds, output, guard) + '\n').encode()
            if not payload.startswith(expected): raise ValueError('Vulkan cached artifact manifest mismatch')
            with tempfile.TemporaryDirectory(prefix='tensorcx-vulkan-check-') as directory:
                path = Path(directory)/'cached.spv'; path.write_bytes(payload[len(expected):])
                _run([tools['spirv-val'], '--target-env', 'vulkan1.1', str(path)])
        # Every hit still undergoes native ABI/resource validation and a fresh
        # device-owned shader/pipeline load; cached loaded handles are forbidden.
        module = _core._load_vulkan_kernel(payload, kinds, output, guard, index)
        if not hit: entry.store(payload)
    return CompiledVulkanKernel(ir.name, target, ir, source, module, cache_hit=hit)
