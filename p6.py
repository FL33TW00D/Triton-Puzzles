import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32
from main import test


"""
Puzzle 6: Fused Outer Multiplication - Backwards
Backwards of a function that multiplies a matrix with a row vector and take a relu.

Uses two program blocks. Block size B0 is always less than the vector x length N0. Block size B1 is always less than vector y length N1. Chain rule backward dz is of shape N1 by N0
"""


def mul_relu_block_back_spec(
    x: Float32[Tensor, "90 100"],
    y: Float32[Tensor, "90"],
    dz: Float32[Tensor, "90 100"],
) -> Float32[Tensor, "90 100"]:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = torch.relu(x * y[:, None])
    z.backward(dz)
    dx = x.grad
    return dx


@triton.jit
def mul_relu_block_back_kernel(
    x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    # 1a. Load x
    cols = tl.arange(0, B0)[None, :] + pid_0 * B0  # [1, B0]
    rows = tl.arange(0, B1)[:, None] + pid_1 * B1  # [B1, 1]
    range = rows * N0 + cols  # [B1, B0]

    mask = (cols < N0) & (rows < N1)
    x = tl.load(x_ptr + range, mask, 0)  # [B0, B1]

    # 1b. Load y
    y_range = tl.arange(0, B1) + pid_1 * B1
    y = tl.load(y_ptr + y_range, y_range < N1)

    # compute xy
    xy = x * y[:, None]

    # compute dzdx **1**[xy > 0] * y, simplifies to single where
    dzdx = tl.where(xy > 0, y[:, None], 0)

    # load dldz shape [N1, N0]
    dldz = tl.load(dz_ptr + range, mask, 0)
    dldx = dzdx * dldz
    tl.store(dx_ptr + range, dldx, mask)
    return


test(mul_relu_block_back_kernel, mul_relu_block_back_spec, nelem={"N0": 100, "N1": 90})
