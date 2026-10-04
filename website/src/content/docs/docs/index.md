---
title: Introduction
description: A small Python-first tensor runtime with a CPU reference and explicit accelerator backends.
---

tensor.cx is an independent, Apache-2.0 licensed project for exploring
tensor execution. A Python API sits above a backend-neutral C++20 core, with
CPU, Apple Metal, and an optional CUDA backend.

The project is **pre-alpha**. It is useful for runtime experiments, studying
backend contracts, and testing small tensor workloads. APIs may change.

## Start here

1. [Build from source](/docs/installation/) with a CPU-only environment.
2. [Run your first tensor operation](/docs/first-tensor/).
3. Read the [backend support matrix](/docs/backends/) before enabling a GPU.
4. Try [matmul and a small inference pipeline](/docs/examples/).

## A deliberately compact runtime

- Contiguous row-major tensors, with `float32` and a smaller `int32` operation set.
- Explicit device transfers and synchronous execution.
- Elementwise add/multiply, 2D matmul, axis reductions, and selected activations
  and normalization operations.
- CPU references for accelerator correctness checks.
- A separate [experimental kernel API](/docs/experimental/).

Autograd, model training, broadcasting, asynchronous streams, distributed
execution, and broad dtype coverage are outside the current runtime. There is
no PyTorch compatibility promise.

## Read the implementation

The [source repository](https://github.com/serkanaltuntas/cortex-runtime) contains
the runtime, tests, and engineering records. These guides are a curated user
entry point. [PROJECT.md](https://github.com/serkanaltuntas/cortex-runtime/blob/main/PROJECT.md)
remains the authoritative phase ledger; architecture and validation records stay
with the implementation.
