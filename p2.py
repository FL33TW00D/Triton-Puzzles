import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32
from main import test


def add2_spec(x: Float32[Tensor, "200"]) -> Float32[Tensor, "200"]:
    return x + 10.0


"""
Puzzle 2: Constant Add Block
Add a constant to a vector. Uses one program block axis (no for loops yet). Block size B0 is now smaller than the shape vector x which is N0.
𝑧𝑖=10+𝑥𝑖 for 𝑖=1…𝑁0
"""


@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    pid = tl.program_id(0)
    x_range = tl.arange(0, B0) + pid * B0
    x = tl.load(x_ptr + x_range, x_range < N0)
    x += 10
    tl.store(z_ptr + x_range, x)


test(add_mask2_kernel, add2_spec, nelem={"N0": 200})
