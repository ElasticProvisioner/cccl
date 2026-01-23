// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/dispatch/dispatch_transform.cuh>
#include <thrust/device_vector.h>
#include <nvbench/nvbench.cuh>

#include <cuda/std/tuple>

struct increment_op
{
  template <typename T>
  __device__ T operator()(const T& a) const
  {
    return a + 1;
  }
};

void bench_unary_transform_pointer(nvbench::state& state)
{
  // Get element size from axis
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  // Setup data: match np.int32 dtype
  thrust::device_vector<int32_t> input(elements, int32_t{5});
  thrust::device_vector<int32_t> output(elements);

  // Report metrics
  state.add_element_count(elements);
  state.add_global_memory_reads<int32_t>(elements);
  state.add_global_memory_writes<int32_t>(elements);

  // Execute using CUB dispatch
  state.exec([&](nvbench::launch& launch) {
    cub::detail::transform::dispatch<cub::detail::transform::requires_stable_address::no>(
      ::cuda::std::tuple{input.begin()},
      output.begin(),
      static_cast<ptrdiff_t>(elements),
      cub::detail::transform::always_true_predicate{},
      increment_op{},
      launch.get_stream());
  });
}

NVBENCH_BENCH(bench_unary_transform_pointer)
  .set_name("bench_unary_transform_pointer")
  .add_int64_power_of_two_axis("Elements", {1, 10, 14, 17, 20, 24, 26});
