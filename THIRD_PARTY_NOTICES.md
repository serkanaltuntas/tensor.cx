# Third-party notices

tensor.cx's own source is licensed under Apache-2.0 (see LICENSE).
These notices cover components compiled into the native extension. They do
not change those components' licenses. Include this file with source and
binary distributions, including CPU-only wheels.

## nanobind — BSD-3-Clause

Source: https://github.com/wjakob/nanobind
Notice verified against nanobind 3.1.0, LICENSE. The build records its actual
resolved version; nanobind is a build dependency with a compiled runtime.

```text
Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## tsl::robin_map — MIT

Source: https://github.com/Tessil/robin-map
Bundled in nanobind; notice verified at revision
4ec1bf19c6a96125ea22062f38c2cf5b958e448e (nanobind 3.1.0).

```text
MIT License

Copyright (c) 2017 Thibaut Goetghebuer-Planchon <tessil@gmx.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## metal-cpp — Apache-2.0 (Metal builds)

Copyright 2024 Apple Inc.

tensor.cx fetches the unmodified headers from
https://github.com/bkaradzic/metal-cpp at
c9727bc9468a90d7ea8fc89d5ee03b8d8992a570, an Apple metal-cpp mirror.
The full Apache-2.0 terms are reproduced in LICENSE. This attribution applies
to metal-cpp, separately from tensor.cx's copyright. The pinned distribution has
no separate NOTICE file.

## DLPack — Apache-2.0

The unmodified DLPack v1.0 header is vendored from
https://github.com/dmlc/dlpack/tree/v1.0 in `third_party/dlpack/` with its license.
Copyright DLPack contributors. The full Apache-2.0 terms are reproduced in
LICENSE. The header defines the exchange ABI used by the binding; DLPack is not
a runtime dependency and its types do not enter the backend-neutral core.

## External dependencies

NumPy is installed separately under its own license. LLVM/MLIR tools, CUDA
SDK/driver/runtime libraries and Apple frameworks remain external requirements;
they are not relicensed by tensor.cx. Current CUDA wheels dynamically depend on
CUDA runtime and driver libraries and do not bundle them. Packaging a future
binary with additional libraries requires checking their redistribution terms
and adding the corresponding notices. No model weights or datasets ship here.
