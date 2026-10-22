"""Strict local Vulkan acceptance of an exact Git commit, optionally with CUDA."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
try:
    from acceptance_support import junit_result, push_revisions, run
    from publication_report import write_public_report
finally:
    sys.path.pop(0)

TEST_FILES = (
    'test_vulkan.py', 'test_vulkan_registry.py', 'test_vulkan_float32.py',
    'test_vulkan_transform.py', 'test_vulkan_predicate.py', 'test_vulkan_reduction.py',
    'test_vulkan_matmul.py', 'test_vulkan_math.py', 'test_vulkan_nn.py',
    'test_vulkan_performance.py', 'test_vulkan_transfer.py', 'test_mlir_vulkan.py',
    'test_int32_primitives.py', 'test_tensor_api.py', 'test_cast_broadcast.py',
    'test_shape_ops.py', 'test_multi_axis_reductions.py', 'test_indexing_joining.py',
    'test_boolean_masks.py', 'test_batched_matmul.py', 'test_math_ops.py',
    'test_inference_ops.py', 'test_device_transfer.py',
)
PYTHON_ANCHORS = (
    'test_vulkan_copy_fill_and_lifetime', 'test_vulkan_float_arithmetic_broadcast_and_scalars',
    'test_vulkan_float32_random_bits', 'test_vulkan_all_cast_pairs',
    'test_vulkan_mask_pipeline_stays_on_device', 'test_vulkan_reduction_cpu_parity',
    'test_vulkan_matmul_separate_rounding_and_underflow',
    'test_vulkan_math_random_full_exponent_range', 'test_vulkan_nn_pipeline_without_host_export',
    'test_vulkan_pool_frames_and_driver_allocation_reduction',
    'test_vulkan_clone_uses_device_copy_without_upload_or_download',
    'test_vulkan_import_without_loader', 'test_vulkan_incomplete_loader_preserves_cpu',
    'test_software_vulkan_is_not_listed_as_a_gpu',
    'test_all_intrinsics_special_values_and_private_output',
    'test_cache_hit_reloads_native_module_and_restart',
    'test_arithmetic_float32_contract', 'test_int32_binary_exact_wrap',
    'test_astype_values_shape_device_and_storage', 'test_squeeze_numpy_and_shared_storage',
    'test_multi_axis_numpy_cpu_parity', 'test_indexing_numpy_parity',
    'test_comparison_numpy_cpu_parity', 'test_batched_matmul_numpy_cpu_parity',
    'test_unary_math_parity', 'test_attention_parity',
    'test_transfer_matrix_bits_metadata_alias_and_lifetime',
)
CUDA_ANCHORS = (
    'test_foreign_cuda_mutation_visible_via_both_aliases',
    'test_dlpack_known_shared_overrun_and_padding_rejected',
    'test_generated_kernel_shared_inputs_private_output_and_suffix',
    'test_transfer_threads_pools_and_explicit_dlpack_bridge',
    'test_vulkan_profile_mixed_cuda_clocks',
)
NATIVE_ANCHORS = (
    'tensorcx_cpu_kernel_tests', 'tensorcx_backend_contract_tests',
    'tensorcx_vulkan_kernel_contract', 'tensorcx_vulkan_kernel_wait_failure',
    'tensorcx_vulkan_kernel_unproven', 'tensorcx_vulkan_contract',
    'tensorcx_vulkan_wait_failure', 'tensorcx_vulkan_unproven_idle',
    'tensorcx_vulkan_pooled_wait_failure', 'tensorcx_vulkan_pooled_unproven_idle',
    'tensorcx_vulkan_timing', 'tensorcx_vulkan_no_loader', 'tensorcx_vulkan_incomplete_loader',
)
CUDA_NATIVE_ANCHORS = (
    'tensorcx_cuda_kernel_tests', 'tensorcx_cuda_backend_contract_tests', 'tensorcx_cuda_profile_tests',
    'tensorcx_vulkan_cuda_shared_contract',
) + tuple(f'tensorcx_shared_{origin}_{fault}' for origin in ('vulkan', 'cuda')
          for fault in ('initialization', 'record', 'cuda_signal', 'cuda_wait', 'cuda_stream_wait',
                        'cuda_completion', 'vk_wait', 'vk_unproven', 'final_sync', 'final_destroy',
                        'foreign_release', 'fork'))
PYTHON_MINIMUM = {False: 4296, True: 6249}


def clean_environment():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('TENSORCX_', 'VK_', 'SKBUILD_', 'CIBW_'))}
    for key in ('VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME', 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS',
                'CMAKE_ARGS', 'CC', 'CXX', 'CUDACXX', 'CUDAHOSTCXX', 'CFLAGS', 'CXXFLAGS',
                'LDFLAGS', 'LD_LIBRARY_PATH', 'LD_PRELOAD'):
        env.pop(key, None)
    for key in subprocess.check_output(['git', 'rev-parse', '--local-env-vars'], cwd=ROOT, text=True).splitlines():
        env.pop(key, None)
    return env


def selected_tests(snapshot, with_cuda):
    files = [snapshot / 'tests/python' / name for name in TEST_FILES]
    if with_cuda:
        files.append(snapshot / 'tests/python/test_shared_storage.py')
    else:
        # These two explicitly heterogeneous tests are outside Vulkan-only mode.
        # In combined mode they are mandatory, not skipped or deselected.
        for name in ('test_vulkan_performance.py::test_vulkan_profile_mixed_cuda_clocks',
                     'test_vulkan_transfer.py::test_transfer_threads_pools_and_explicit_dlpack_bridge'):
            files.extend(['--deselect', 'tests/python/' + name])
    return files


def software_discovery(summary):
    fields = [line.strip().partition('=') for line in summary.splitlines() if '=' in line]
    kinds = [value.strip() for key, _, value in fields if key.strip() == 'deviceType']
    drivers = [value.strip() for key, _, value in fields if key.strip() == 'driverName']
    if (not kinds or len(kinds) != len(drivers)
            or any(kind != 'PHYSICAL_DEVICE_TYPE_CPU' for kind in kinds)
            or any(driver != 'llvmpipe' for driver in drivers)):
        raise RuntimeError('software ICD did not enumerate the required Mesa CPU device')
    return [key.strip() + '=' + value.strip() for key, _, value in fields
            if key.strip() in {'deviceType', 'driverName', 'driverInfo', 'apiVersion'}]


def validate(revision, args):
    env = clean_environment()
    resolved = subprocess.check_output(
        ['git', 'rev-parse', '--verify', '--end-of-options', revision + '^{commit}'], cwd=ROOT, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = ROOT / 'build/vulkan-gate' / (stamp + '-' + resolved[:12])
    output.mkdir(parents=True, exist_ok=False)
    strict = ROOT / 'tools/cuda_gate_pytest.py'  # Shared backend-independent strict plugin.
    strict_bytes = strict.read_bytes()
    report = {'schema_version': 1, 'revision': resolved, 'started_utc': stamp, 'status': 'running',
              'with_cuda': args.with_cuda,
              'python_minimum': PYTHON_MINIMUM[args.with_cuda],
              'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'support_sha256': hashlib.sha256((ROOT / 'tools/acceptance_support.py').read_bytes()).hexdigest(),
              'pytest_controller_sha256': hashlib.sha256(strict_bytes).hexdigest(),
              'test_files': list(TEST_FILES) + (['test_shared_storage.py'] if args.with_cuda else []),
              'validation_required': True}
    print(f'Vulkan gate: {resolved}; CUDA sharing: {args.with_cuda}; logs: {output}', flush=True)
    try:
        sdk, llvm, shaders = args.sdk.resolve(), args.llvm_bin.resolve(), args.shader_bin.resolve()
        layer, software = args.validation_layer.resolve(), args.software_icd.resolve()
        if not (sdk / 'include/vulkan/vulkan.h').is_file():
            raise RuntimeError('Vulkan SDK header is missing')
        if not software.is_file() or not any(layer.glob('*.json')):
            raise RuntimeError('software ICD and explicit validation-layer manifests are required')
        env.update(CC=args.cc, CXX=args.cxx, CMAKE_BUILD_PARALLEL_LEVEL='2',
                   TENSORCX_LLVM_BIN=str(llvm), TENSORCX_VULKAN_BIN=str(shaders),
                   TENSORCX_VULKAN_VALIDATION='1', VK_LAYER_PATH=str(layer),
                   TENSORCX_TEST_VULKAN_SOFTWARE_ICD=str(software),
                   TENSORCX_REQUIRE_BACKENDS='cpu,vulkan' + (',cuda' if args.with_cuda else ''),
                   TENSORCX_REQUIRE_MLIR_VULKAN='1', TENSORCX_REQUIRE_MLIR_SPIRV='1')
        if args.library_dir:
            env['LD_LIBRARY_PATH'] = str(args.library_dir.resolve())
        if args.gpu_icd:
            env.update(VK_DRIVER_FILES=str(args.gpu_icd.resolve()),
                       VK_ICD_FILENAMES=str(args.gpu_icd.resolve()))
        if args.with_cuda:
            env.update(CUDACXX='nvcc', CUDAHOSTCXX=args.cxx,
                       TENSORCX_REQUIRE_SHARED_CUDA_VULKAN='1')
        options = ['-DTENSORCX_ENABLE_METAL=OFF', '-DTENSORCX_ENABLE_VULKAN=ON',
                   '-DTENSORCX_ENABLE_CUDA=' + ('ON' if args.with_cuda else 'OFF'),
                   f'-DTENSORCX_VULKAN_INCLUDE_DIR={sdk}/include',
                   f'-DTENSORCX_GLSLANG_VALIDATOR={shaders}/glslangValidator',
                   f'-DTENSORCX_SPIRV_VAL={shaders}/spirv-val']
        if args.with_cuda:
            options.append('-DCMAKE_CUDA_ARCHITECTURES=52')
        report['compiler'] = run([args.cxx, '--version'], ROOT, env, output / 'compiler.log')
        software_env = {**env, 'VK_DRIVER_FILES': str(software), 'VK_ICD_FILENAMES': str(software)}
        summary = run([args.vulkaninfo.resolve(), '--summary'], ROOT, software_env, output / 'software-discovery.log')
        report['software_discovery'] = software_discovery(summary)
        with tempfile.TemporaryDirectory(prefix='tensorcx-vulkan-gate-') as temporary:
            work = Path(temporary)
            snapshot = work / 'source'
            snapshot.mkdir()
            archive = work / 'source.tar'
            run(['git', 'archive', '--format=tar', '-o', archive, resolved], ROOT, env, output / 'archive.log')
            report['source_archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
            with tarfile.open(archive) as source:
                source.extractall(snapshot, filter='data')
            python = work / 'env/bin/python'
            run(['uv', 'venv', work / 'env', '--python', sys.executable], work, env, output / 'venv.log')
            wheels = output / 'wheel'
            wheels.mkdir()
            run(['uv', 'build', '--wheel', snapshot, '--python', sys.executable, '--out-dir', wheels],
                work, {**env, 'CMAKE_ARGS': shlex.join(options)}, output / 'wheel-build.log')
            artifacts = list(wheels.glob('*.whl'))
            if len(artifacts) != 1:
                raise RuntimeError('exact-commit build must produce one wheel')
            report['wheel_sha256'] = hashlib.sha256(artifacts[0].read_bytes()).hexdigest()
            run(['uv', 'pip', 'install', '--python', python, artifacts[0], 'pytest>=8', 'nanobind>=2.4',
                 'cmake>=3.21', 'ninja'], work, env, output / 'install.log')
            runner = ['uv', 'run', '--no-project', '--python', python]
            report['dependencies'] = run(['uv', 'pip', 'freeze', '--python', python], work, env, output / 'dependencies.log')
            raw = run([python, '-I', '-c', '''import json, tensorcx as cx
from tensorcx._compiler.vulkan import toolchain
assert cx.is_available('vulkan'), 'real Vulkan GPU required'
assert cx._core._build_info()['vulkan']
cx.ones((3,), device='vulkan').numpy()
assert cx._core._vulkan_validation_errors() == 0
details = cx.diagnostics(check_tools=True)
assert details['llvm']['ready_targets']['vulkan'], 'Vulkan compiler diagnostics not ready'
print(json.dumps({'devices': cx.devices(), 'build': cx._core._build_info(),
                  'tools': toolchain(), 'diagnostics': details}))
'''], work, env, output / 'preflight.log')
            report['preflight'] = json.loads(raw)
            if report['preflight']['build']['cuda'] != args.with_cuda:
                raise RuntimeError('wrong acceptance backend build variant')
            nanobind = run([python, '-I', '-c', 'import nanobind; print(nanobind.cmake_dir())'],
                           work, env, output / 'nanobind.log')
            native = work / 'native'
            run([*runner, 'cmake', '-S', snapshot, '-B', native, '-G', 'Ninja', *options,
                 '-DTENSORCX_BUILD_TESTS=ON', '-Dnanobind_DIR=' + nanobind,
                 '-DPython_EXECUTABLE=' + str(python)], work, env, output / 'configure.log')
            run([*runner, 'cmake', '--build', native, '--parallel', '2'], work, env, output / 'build.log')
            run([*runner, 'ctest', '--test-dir', native, '--no-tests=error', '--output-on-failure',
                 '--output-junit', output / 'ctest.xml'], work, env, output / 'ctest.log')
            anchors = NATIVE_ANCHORS + (CUDA_NATIVE_ANCHORS if args.with_cuda else ())
            report['native'] = junit_result(output / 'ctest.xml', len(anchors), anchors)
            env.update(TENSORCX_TEST_VULKAN_NO_LOADER=str(native / 'libtensorcx_vulkan_no_loader.so'),
                       TENSORCX_TEST_VULKAN_INCOMPLETE_LIBRARY=str(native / 'libtensorcx_vulkan_incomplete_loader.so'))
            pytest_controller = work / 'strict_pytest.py'
            pytest_controller.write_bytes(strict_bytes)
            run([python, '-I', pytest_controller, '-q', '--rootdir', snapshot, '-o', 'xfail_strict=true',
                 *selected_tests(snapshot, args.with_cuda), '--junitxml', output / 'pytest.xml'],
                work, env, output / 'pytest.log')
            report['python'] = junit_result(output / 'pytest.xml', PYTHON_MINIMUM[args.with_cuda],
                PYTHON_ANCHORS + (CUDA_ANCHORS if args.with_cuda else ()))
            # Installed-wheel smoke checks use the exact snapshot's workload.
            variant = 'cuda-vulkan' if args.with_cuda else 'vulkan'
            for mode, flags in (('gpu', []), ('mlir', ['--mlir']), ('hidden-gpu', ['--hidden-gpu'])):
                runtime = env.copy()
                if mode != 'mlir': runtime['TENSORCX_LLVM_BIN'] = str(work / 'missing-llvm')
                if mode == 'hidden-gpu':
                    runtime.update(CUDA_VISIBLE_DEVICES='', VK_DRIVER_FILES=str(work / 'missing-icd.json'),
                                   VK_ICD_FILENAMES=str(work / 'missing-icd.json'))
                raw = run([python, '-I', snapshot / 'tools/distribution_probe.py', '--variant', variant, *flags],
                          work, runtime, output / (mode + '.log'))
                report[mode] = json.loads(raw)
            if args.with_cuda:
                for tool in ('memcheck', 'racecheck'):
                    run(['compute-sanitizer', '--tool', tool, '--error-exitcode', '1',
                         native / 'tensorcx_vulkan_cuda_shared_tests'], work, env, output / (tool + '.log'))
                report['device_sanitizers'] = ['memcheck', 'racecheck']
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_public_report(output / 'result.json', report, root=ROOT)
    print(f'Vulkan gate passed: {resolved}; {output / "result.json"}', flush=True)


def configuration(name, *, boolean=False):
    command = ['git', 'config', '--local', *(('--bool',) if boolean else ()), '--get', 'tensorcx.vulkan' + name]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode == 1:
        return None
    if result.returncode:
        raise RuntimeError('invalid local Vulkan gate configuration: ' + name)
    return result.stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--revision')
    mode.add_argument('--pre-push', action='store_true')
    for option in ('sdk', 'llvm-bin', 'shader-bin', 'validation-layer', 'software-icd',
                   'library-dir', 'gpu-icd', 'vulkaninfo'):
        parser.add_argument('--' + option, type=Path)
    parser.add_argument('--with-cuda', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--cc')
    parser.add_argument('--cxx')
    args = parser.parse_args(argv)
    revisions = push_revisions(sys.stdin.read()) if args.pre_push else [args.revision]
    if not revisions:
        return
    for name, key in (('sdk', 'Sdk'), ('llvm_bin', 'LlvmBin'), ('shader_bin', 'ShaderBin'),
                      ('validation_layer', 'ValidationLayer'), ('software_icd', 'SoftwareIcd'),
                      ('library_dir', 'LibraryDir'), ('gpu_icd', 'GpuIcd'), ('vulkaninfo', 'Info')):
        if getattr(args, name) is None:
            value = configuration(key)
            if value: setattr(args, name, Path(value))
    if args.with_cuda is None:
        args.with_cuda = configuration('WithCuda', boolean=True) == 'true'
    args.cc = args.cc or configuration('Cc') or ('gcc-13' if args.with_cuda else 'cc')
    args.cxx = args.cxx or configuration('Cxx') or ('g++-13' if args.with_cuda else 'c++')
    if args.shader_bin is None and args.sdk:
        args.shader_bin = args.sdk / 'bin'
    if args.vulkaninfo is None and args.sdk:
        args.vulkaninfo = args.sdk / 'bin/vulkaninfo'
    for name in ('sdk', 'llvm_bin', 'shader_bin', 'validation_layer', 'software_icd'):
        if getattr(args, name) is None:
            parser.error(f'provide --{name.replace("_", "-")} or its tensorcx.vulkan Git config')
    for revision in revisions:
        validate(revision, args)


if __name__ == '__main__':
    main()
