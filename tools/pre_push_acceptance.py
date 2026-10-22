"""Preserve Git's exact stdin while running the mandatory and opt-in GPU gates."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    payload = sys.stdin.buffer.read()
    result = subprocess.run([sys.executable, str(ROOT / 'tools/cuda_push_gate.py'), '--pre-push'], input=payload)
    if result.returncode:
        return result.returncode
    config = subprocess.run(['git', 'config', '--local', '--bool', '--get', 'tensorcx.vulkanGateEnabled'],
                            cwd=ROOT, text=True, capture_output=True)
    if config.returncode == 1:
        return 0  # An ordinary clone does not opt in to a second local gate.
    if config.returncode or config.stdout.strip() not in {'true', 'false'}:
        raise RuntimeError('invalid tensorcx.vulkanGateEnabled local Git configuration')
    if config.stdout.strip() == 'true':
        return subprocess.run([sys.executable, str(ROOT / 'tools/vulkan_push_gate.py'), '--pre-push'],
                              input=payload).returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
