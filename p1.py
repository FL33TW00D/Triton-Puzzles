import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32
from main import test


def add_spec(x: Float32[Tensor, "32"]) -> Float32[Tensor, "32"]:
    "This is the spec that you should implement. Uses typing to define sizes."
    return x + 10.0


@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    range = tl.arange(0, B0)
    x = tl.load(x_ptr + range)
    x += 10
    tl.store(z_ptr + range, x)


test(add_kernel, add_spec, nelem={"N0": 32})
