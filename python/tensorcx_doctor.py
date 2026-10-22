"""Standalone CLI: remains importable when tensorcx's native module is broken."""
from __future__ import annotations

import argparse
import importlib
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    parser.add_argument("--check-tools", action="store_true", help="run bounded compiler tool version checks")
    args = parser.parse_args(argv)
    try:
        runtime = importlib.import_module("tensorcx")
    except (ImportError, OSError) as error:
        report = {"schema_version": 1, "status": "import_error",
                  "error_type": type(error).__name__,
                  "hint": "The runtime could not be imported. Install tensorcx and NumPy in this interpreter; use the CPU wheel without CUDA libraries, or install the matching CUDA runtime/driver for a CUDA build."}
    else:
        try:
            report = runtime.diagnostics(check_tools=args.check_tools)
        except (AttributeError, RuntimeError, OSError) as error:
            report = {"schema_version": 1, "status": "diagnostics_error",
                      "error_type": type(error).__name__,
                      "hint": "Diagnostic collection failed. Rebuild/reinstall a matching Python package and native extension, and check device/driver health."}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["status"] != "ok":
        print("tensor.cx: " + report["status"] + " (" + report["error_type"] + ")")
        print(report["hint"])
    else:
        print(f"tensor.cx {report['version']} — {report['system']} {report['machine']}, Python {report['python']}")
        for name, info in report["backends"].items():
            print(f"{name}: compiled={info['compiled']}, available={info['available']}")
            memory = info["memory"]
            print(f"  memory ({memory['scope']}): available={memory['available_bytes']}, total={memory['total_bytes']} bytes")
            if info.get("architecture"):
                print(f"  architecture={info['architecture']}, runtime={info['runtime_version']}, driver API={info['driver_api_version']}")
            if info.get("hint"):
                print("  " + info["hint"])
            if info["operations"]:
                print("  operations: " + "; ".join(f"{op}({','.join(dtypes)})" for op, dtypes in info["operations"].items()))
        print("LLVM: " + report["llvm"]["status"])
        ready = [name for name, value in report['llvm']['ready_targets'].items() if value]
        print("MLIR ready targets: " + (', '.join(ready) or 'none verified'))
        print(report["llvm"]["hint"])
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
