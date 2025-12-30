import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32
from main import test


"""
Puzzle 7: Long Sum
Sum of a batch of numbers.

Uses one program blocks. Block size B0 represents a range of batches of x of length N0. Each element is of length T. Process it B1 < T elements at a time.

𝑧𝑖=∑𝑗𝑇𝑥𝑖,𝑗= for 𝑖=1…𝑁0 

Hint: You will need a for loop for this problem. These work and look the same as in Python.
"""


def sum_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4"]:
    return x.sum(1)


@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    return


test(sum_kernel, sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
