import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32
from main import test


def add_vec_block_spec(
    x: Float32[Tensor, "100"], y: Float32[Tensor, "90"]
) -> Float32[Tensor, "90 100"]:
    return x[None, :] + y[:, None]


"""
Puzzle 4: Outer Vector Add Block
Add a row vector to a column vector.

Uses two program block axes. Block size B0 is always less than the vector x length N0. Block size B1 is always less than vector y length N1.
"""


@triton.jit
def add_vec_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    # load x and y
    x_range = tl.arange(0, B0) + pid_0 * B0
    y_range = tl.arange(0, B1) + pid_1 * B1

    x = tl.load(x_ptr + x_range, x_range < N0)
    y = tl.load(y_ptr + y_range, y_range < N1)

    # x [1, B0] + y [B1, 1] -> z [B1, B0]
    z = x[None, :] + y[:, None]

    z_range = y_range[:, None] * N0 + x_range[None, :]
    mask = (x_range[None, :] < N0) & (y_range[:, None] < N1)

    tl.store(z_ptr + z_range, z, mask)


test(add_vec_block_kernel, add_vec_block_spec, nelem={"N0": 100, "N1": 90})
