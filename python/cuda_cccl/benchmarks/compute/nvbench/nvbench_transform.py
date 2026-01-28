# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import cupy as cp
import numpy as np

import cuda.bench as bench
import cuda.compute
from cuda.compute import CountingIterator, OpKind, gpu_struct


def as_cupy_stream(cs: bench.CudaStream) -> cp.cuda.Stream:
    """Convert nvbench CudaStream to CuPy Stream."""
    return cp.cuda.ExternalStream(cs.addressof())


def unary_transform_pointer(inp, out):
    size = len(inp)

    def op(a):
        return a + 1

    transform = cuda.compute.make_unary_transform(inp, out, op)

    cp.cuda.runtime.deviceSynchronize()

    return transform


def bench_unary_transform_pointer(state: bench.State):
    """Benchmark unary transform with pointer inputs (runtime only)"""

    # Get element size from axis
    size = state.get_int64("Elements")

    # Setup data: np.int32 dtype
    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        inp = cp.random.randint(0, 10, size, dtype=np.int32)
        out = cp.empty_like(inp)

    # Build transform operation
    transform = unary_transform_pointer(inp, out)

    # Report metrics
    state.add_element_count(size)
    state.add_global_memory_reads(size * inp.dtype.itemsize)
    state.add_global_memory_writes(size * out.dtype.itemsize)

    # Execute dispatch
    def launcher(launch: bench.Launch):
        exec_stream = as_cupy_stream(launch.get_stream())
        with exec_stream:
            transform(inp, out, size)

    state.exec(launcher)


if __name__ == "__main__":
    b1 = bench.register(bench_unary_transform_pointer)
    b1.add_int64_power_of_two_axis("Elements", range(12, 29, 4))

    bench.run_all_benchmarks(sys.argv)
