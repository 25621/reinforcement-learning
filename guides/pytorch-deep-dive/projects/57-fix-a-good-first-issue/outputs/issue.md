# log_sigmoid's out= variant resizes silently

## Summary

`torch.nn.functional.logsigmoid(x, out=y)` resizes a non-empty `y` without
emitting the documented resize warning. Sibling unary ops (`sigmoid`, `tanh`,
`threshold`) all warn.

## Repro

```python
import torch, warnings
x = torch.randn(20)
out = torch.empty(21)                      # wrong shape, NOT empty
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    torch.nn.functional.logsigmoid(x, out=out)
print(out.shape, [str(m.message)[:40] for m in w])
# torch.Size([20]) []      <- resized, no warning
torch.sigmoid(x, out=torch.empty(21))      # the sibling op DOES warn
```

## Expected

A `UserWarning` starting "An output with one or more elements was resized",
as produced by `torch.sigmoid(x, out=torch.empty(21))`.

## Cause

`log_sigmoid.out` is not a structured kernel, so it resizes its output by hand.
`log_sigmoid_forward_out_cpu` in `aten/src/ATen/native/Activation.cpp` calls
`result.resize_as_(input)`, which is silent, instead of
`at::native::resize_output(result, input.sizes())`, which warns.

## Already tracked

`nn.functional.logsigmoid`'s OpInfo carries an `expectedFailure` for
`TestCommon.test_out_warning`. 27 operators carry that marker.

## Environment

torch 2.10.0+cu128, built from 449b17684101, CPU.

## Same sweep, other operators affected

addbmm, arange, bernoulli, empty, full, log_sigmoid, multinomial, narrow_copy, normal, ones, randn, zeros
