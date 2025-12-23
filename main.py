import inspect
import torch
import triton


def test(puzzle, puzzle_spec, nelem={}, B={"B0": 32}):
    B = dict(B)
    if "N1" in nelem and "B1" not in B:
        B["B1"] = 32
    if "N2" in nelem and "B2" not in B:
        B["B2"] = 32

    torch.manual_seed(0)
    signature = inspect.signature(puzzle_spec)
    args = {}
    for n, p in signature.parameters.items():
        print(p)
        args[n + "_ptr"] = ([d.size for d in p.annotation.dims], p)
    args["z_ptr"] = ([d.size for d in signature.return_annotation.dims], None)

    tt_args = []
    for k, (v, t) in args.items():
        tt_args.append(torch.rand(*v, device="cuda") - 0.5)
        if t is not None and t.annotation.dtypes[0] == "int32":
            tt_args[-1] = torch.randint(-100000, 100000, v, device="cuda")

    grid = lambda meta: (
        triton.cdiv(nelem["N0"], meta["B0"]),
        triton.cdiv(nelem.get("N1", 1), meta.get("B1", 1)),
        triton.cdiv(nelem.get("N2", 1), meta.get("B2", 1)),
    )

    puzzle[grid](*tt_args, **B, **nelem)
    z = tt_args[-1]
    tt_args = tt_args[:-1]

    # Run spec on CPU then move to GPU for comparison
    cpu_args = [arg.cpu() for arg in tt_args]
    z_ = puzzle_spec(*cpu_args).cuda()

    match = torch.allclose(z, z_, rtol=1e-3, atol=1e-3)
    print("Results match:", match)

    if not match:
        print("Yours:", z)
        print("Spec:", z_)
        print(torch.isclose(z, z_))
    else:
        print("Correct!")
