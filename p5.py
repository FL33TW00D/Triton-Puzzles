import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32
from main import test


"""
Puzzle 5: Fused Outer Multiplication
Multiply a row vector to a column vector and take a relu.

Uses two program block axes. Block size B0 is always less than the vector x length N0. Block size B1 is always less than vector y length N1.

𝑧𝑗,𝑖=relu(𝑥𝑖×𝑦𝑗) for 𝑖=1…𝑁0, 𝑗=1…𝑁1
"""


def mul_relu_block_spec(
    x: Float32[Tensor, "100"], y: Float32[Tensor, "90"]
) -> Float32[Tensor, "90 100"]:
    return torch.relu(x[None, :] * y[:, None])


@triton.jit
def mul_relu_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    # load in x
    # load in y
    # multiply cur block

    x_range = tl.arange(0, B0) + pid_0 * B0
    y_range = tl.arange(0, B1) + pid_1 * B1

    x = tl.load(x_ptr + x_range, x_range < N0)
    y = tl.load(y_ptr + y_range, y_range < N1)

    # (B1, B0) = (1, B0) * (B1, 1)
    z = x[None, :] * y[:, None]
    z = tl.maximum(z, 0)

    z_range = y_range[:, None] * N0 + x_range[None, :]
    z_mask = (y_range[:, None] < N1) & (x_range[None, :] < N0)

    tl.store(z_ptr + z_range, z, z_mask)

    return


test(mul_relu_block_kernel, mul_relu_block_spec, nelem={"N0": 100, "N1": 90})
