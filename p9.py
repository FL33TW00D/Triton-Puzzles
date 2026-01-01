import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32
from main import test
import math


r"""
## Puzzle 9: Simple FlashAttention

A scalar version of FlashAttention.

Uses one program block axis. Block size `B0` represent the batches of `q` to process out of `N0`. Sequence length is `T`. Process it `B1 < T` elements (`k`, `v`) at a time for some `B1`.

.. math::
    z_{i} = \sum_{j=1}^{T} \text{softmax}(q_i k_1, \ldots, q_i k_T)_j v_{j} \text{ for } i = 1\ldots N_0

This can be done in 1 loop using a similar trick from the last puzzle.

Hint: Use `tl.where` to mask `q dot k` to -inf to avoid overflow (NaN).
"""


def flashatt_spec(
    q: Float32[Tensor, "200"], k: Float32[Tensor, "200"], v: Float32[Tensor, "200"]
) -> Float32[Tensor, "200"]:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft = x_exp / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)


"""
pid = tl.program_id(0)

q_batch = ...  # shape: (B0,)

# Initialize accumulators for online softmax
# (what do you need to track?)

# Actual loop over ALL keys/values, B1 at a time
for j in range(0, T, B1):
    # Load chunk of keys and values
    k_chunk = ...  # shape: (B1,)
    v_chunk = ...  # shape: (B1,)
    
    Outer product of q and k first
    
# Store final result
"""


@triton.jit
def flashatt_kernel(
    q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr, B1: tl.constexpr
):
    pid = tl.program_id(0)

    q_range = pid * B0 + tl.arange(0, B0)
    q = tl.load(q_ptr + q_range, q_range < N0)

    # Compute maximum of qk products
    qk_max = tl.full((B0,), -math.inf, dtype=tl.float32)
    for j in range(0, T, B1):
        k_range = j + tl.arange(0, B1)
        k = tl.load(k_ptr + k_range, k_range < T)

        qk = q[:, None] * k[None, :]  # [B0, B1] -> [64, 32]
        # partial k product means we need to complete all blocks to get the max
        qk_max = tl.maximum(qk_max, tl.max(qk, axis=1))

    denom = tl.zeros((B0,), dtype=tl.float32)
    # Compute denominator sum(exp((qk-qk_max))
    for j in range(0, T, B1):
        k_range = j + tl.arange(0, B1)
        k = tl.load(k_ptr + k_range, k_range < T)

        v_range = j + tl.arange(0, B1)
        v = tl.load(v_ptr + v_range, v_range < T)

        qk = q[:, None] * k[None, :]  # [B0, B1] -> [64, 32]
        # Triton by default uses 0.0 as the mask value
        qk = tl.where((k_range < T)[None, :], qk, float("-inf"))

        denom += tl.sum(tl.exp(qk - qk_max[:, None]), axis=1)

    # Now do ((exp(qk - qk_max) / denom) * vi)
    z = tl.zeros((B0,), dtype=tl.float32)
    for j in range(0, T, B1):
        k_range = j + tl.arange(0, B1)
        k = tl.load(k_ptr + k_range, k_range < T)

        v_range = j + tl.arange(0, B1)
        v = tl.load(v_ptr + v_range, v_range < T)

        qk = q[:, None] * k[None, :]  # [B0, B1] -> [64, 32]

        weights = tl.exp(qk - qk_max[:, None]) / denom[:, None]  # [B0, B1]

        z += tl.sum(weights * v[None, :], axis=1)

    # now write it out
    tl.store(z_ptr + q_range, z, mask=q_range < N0)

    return


test(
    flashatt_kernel,
    flashatt_spec,
    B={"B0": 64, "B1": 32},
    nelem={"N0": 200, "T": 200},
)
