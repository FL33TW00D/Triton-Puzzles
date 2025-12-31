import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32
from main import test
import math

"""
## Puzzle 8: Long Softmax

Softmax of a batch of logits.

Uses one program block axis. Block size `B0` represents the batch of `x` of length `N0`.
Block logit length `T`.   Process it `B1 < T` elements at a time.  

Note softmax needs to be computed in numerically stable form as in Python. In addition in Triton they recommend not using `exp` but instead using `exp2`. You need the identity

Advanced: there one way to do this with 3 loops. You can also do it with 2 loops if you are clever. Hint: you will find this identity useful:
"""


def softmax_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4 200"]:
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)


@triton.jit
def softmax_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    # [4 rows, 200 columns]
    # single program block, each block is dealing with a single row
    pid_0 = tl.program_id(0)
    log2_e = 1.44269504

    # Compute x_max
    x_max = float("-inf")
    for i in range(tl.cdiv(T, B1)):
        row_off = pid_0 * T
        col_off = i * B1 + tl.arange(0, B1)

        x = tl.load(x_ptr + (row_off + col_off), col_off < T)
        x_max = tl.maximum(tl.max(x), x_max)

    # x_max is now max across all 200 vals

    x_exp_sum = 0.0
    for i in range(tl.cdiv(T, B1)):
        row_off = pid_0 * T
        col_off = i * B1 + tl.arange(0, B1)

        x = tl.load(x_ptr + (row_off + col_off), col_off < T)
        x = tl.exp2(log2_e * (x - x_max))
        x_exp_sum += tl.sum(x)

    # Compute x_exp / x_exp_sum
    for i in range(tl.cdiv(T, B1)):
        row_off = pid_0 * T
        col_off = i * B1 + tl.arange(0, B1)
        x = tl.load(x_ptr + (row_off + col_off), col_off < T)

        x_exp = tl.exp2(log2_e * (x - x_max))
        z = x_exp / x_exp_sum
        tl.store(z_ptr + (row_off + col_off), z, col_off < T)

    return


test(
    softmax_kernel,
    softmax_spec,
    B={"B0": 1, "B1": 32},
    nelem={"N0": 4, "N1": 32, "T": 200},
)
