import os

import pytest

import cortex_runtime as cx


BACKEND_ORDER = ("cpu", "metal", "cuda")
KNOWN_BACKENDS = frozenset(BACKEND_ORDER)
KNOWN_CAPABILITIES = frozenset(
    {
        "copy",
        "tensor_factories_float32",
        "tensor_factories_int32",
        "binary_ops_float32",
        "binary_ops_int32",
        "binary_ops_dtype_mismatch",
        "unary_float32",
        "reductions_float32",
        "reductions_int32",
        "normalization_float32",
    }
)
CAPABILITY_DEPENDENCIES = {
    "tensor_factories_float32": frozenset({"copy"}),
    "tensor_factories_int32": frozenset({"copy"}),
    "binary_ops_float32": frozenset({"copy", "tensor_factories_float32"}),
    "binary_ops_int32": frozenset({"copy"}),
    "binary_ops_dtype_mismatch": frozenset(
        {"binary_ops_float32", "tensor_factories_float32", "tensor_factories_int32"}
    ),
    "unary_float32": frozenset({"copy"}),
    "reductions_float32": frozenset({"copy"}),
    "reductions_int32": frozenset({"copy"}),
    "normalization_float32": frozenset({"copy"}),
}

BACKEND_CAPABILITIES = {
    "cpu": frozenset(
        {
            "copy",
            "tensor_factories_float32",
            "tensor_factories_int32",
            "binary_ops_float32",
            "binary_ops_int32",
            "binary_ops_dtype_mismatch",
            "unary_float32",
            "reductions_float32",
            "reductions_int32",
            "normalization_float32",
        }
    ),
    "metal": frozenset(
        {
            "copy",
            "tensor_factories_float32",
            "tensor_factories_int32",
            "binary_ops_float32",
            "binary_ops_int32",
            "binary_ops_dtype_mismatch",
            "unary_float32",
            "reductions_float32",
            "reductions_int32",
            "normalization_float32",
        }
    ),
    "cuda": frozenset({"copy", "tensor_factories_float32", "binary_ops_float32"}),
}


def pytest_configure(config):
    _validate_declared_capabilities()

    required = _required_backends()
    required_capabilities = _required_backend_capabilities()
    required_by_capability = frozenset(required_capabilities)
    required_backends = required | required_by_capability

    unknown = sorted(required_backends - KNOWN_BACKENDS)
    if unknown:
        raise pytest.UsageError(
            "required backend configuration contains unknown backend(s): "
            + ", ".join(unknown)
        )

    unavailable = sorted(name for name in required_backends if not cx.is_available(name))
    if unavailable:
        raise pytest.UsageError(
            "required backend(s) unavailable: "
            + ", ".join(unavailable)
            + "; unset CORTEX_REQUIRE_BACKENDS or fix backend registration"
        )

    missing = []
    for name, capabilities in required_capabilities.items():
        declared = BACKEND_CAPABILITIES.get(name, frozenset())
        missing.extend(
            f"{name}:{capability}"
            for capability in sorted(capabilities)
            if capability not in declared
        )
    if missing:
        raise pytest.UsageError(
            "required backend capability not declared: " + ", ".join(missing)
        )


def pytest_generate_tests(metafunc):
    if "backend_name" not in metafunc.fixturenames:
        return

    marker = metafunc.definition.get_closest_marker("backend_capability")
    if marker is None:
        return
    if len(marker.args) != 1:
        raise ValueError("backend_capability marker requires exactly one capability")

    capability = marker.args[0]
    if capability not in KNOWN_CAPABILITIES:
        raise ValueError(f"unknown backend capability: {capability}")
    include_cpu = marker.kwargs.get("include_cpu", True)
    if not isinstance(include_cpu, bool):
        raise ValueError("backend_capability include_cpu must be a boolean")
    backends = (
        BACKEND_ORDER
        if include_cpu
        else tuple(name for name in BACKEND_ORDER if name != "cpu")
    )
    metafunc.parametrize(
        "backend_name",
        [_backend_param(name, capability) for name in backends],
    )


def _backend_param(name: str, capability: str):
    marks = []
    if capability not in BACKEND_CAPABILITIES.get(name, frozenset()):
        marks.append(
            pytest.mark.skip(
                reason=f"{name} backend does not declare {capability} capability"
            )
        )
    elif not cx.is_available(name):
        marks.append(pytest.mark.skip(reason=f"{name} backend is not available"))
    return pytest.param(name, id=name, marks=marks)


def _required_backends() -> frozenset[str]:
    raw = os.environ.get("CORTEX_REQUIRE_BACKENDS", "")
    return frozenset(name.strip() for name in raw.split(",") if name.strip())


def _required_backend_capabilities() -> dict[str, frozenset[str]]:
    raw = os.environ.get("CORTEX_REQUIRE_BACKEND_CAPABILITIES", "")
    required = {}
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        backend, separator, capability = item.partition(":")
        backend = backend.strip()
        capability = capability.strip()
        if not separator or not backend or not capability:
            raise pytest.UsageError(
                "CORTEX_REQUIRE_BACKEND_CAPABILITIES entries must use "
                "backend:capability"
            )
        if capability not in KNOWN_CAPABILITIES:
            raise pytest.UsageError(
                "CORTEX_REQUIRE_BACKEND_CAPABILITIES contains unknown "
                f"capability: {capability}"
            )
        required.setdefault(backend, set()).add(capability)
    return {name: frozenset(capabilities) for name, capabilities in required.items()}


def _validate_declared_capabilities():
    for name, capabilities in BACKEND_CAPABILITIES.items():
        unknown = sorted(capabilities - KNOWN_CAPABILITIES)
        if unknown:
            raise pytest.UsageError(
                f"{name} backend declares unknown capability: " + ", ".join(unknown)
            )

        missing_dependencies = []
        for capability in sorted(capabilities):
            dependencies = CAPABILITY_DEPENDENCIES.get(capability, frozenset())
            for dependency in sorted(dependencies - capabilities):
                missing_dependencies.append(f"{capability}->{dependency}")
        if missing_dependencies:
            raise pytest.UsageError(
                f"{name} backend capability dependencies are missing: "
                + ", ".join(missing_dependencies)
            )
