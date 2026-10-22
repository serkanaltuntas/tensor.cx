"""Package canonical project-owned shader helpers without requiring LLVM."""
from pathlib import Path
import sys


output = Path(sys.argv[1])
inputs = [Path(path) for path in sys.argv[2:]]
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('"""Generated from canonical Vulkan GLSL helpers; do not edit."""\n'
                  + 'SOURCE = ' + repr('\n'.join(path.read_text() for path in inputs)) + '\n')
