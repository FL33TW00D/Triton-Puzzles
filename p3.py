import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32
from main import test


def add_vec_spec(
    x: Float32[Tensor, "32"], y: Float32[Tensor, "32"]
) -> Float32[Tensor, "32 32"]:
    return x[None, :] + y[:, None]


"""
Puzzle 3: Outer Vector Add
Add two vectors.

Uses one program block axis. Block size B0 is always the same as vector x length N0. Block size B1 is always the same as vector y length N1.

𝑧𝑗,𝑖=𝑥𝑖+𝑦𝑗 for 𝑖=1…𝐵0, 𝑗=1…𝐵1
"""


@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    x_range = tl.arange(0, B0)
    y_range = tl.arange(0, B1)

    x = tl.load(x_ptr + x_range, x_range < N0)
    y = tl.load(y_ptr + y_range, y_range < N1)

    z = x[None, :] + y[:, None]
    z_range = y_range[:, None] * N0 + x_range[None, :]
    tl.store(z_ptr + z_range, z, z_range < N0 * N1)


test(add_vec_kernel, add_vec_spec, nelem={"N0": 32, "N1": 32})
