import torch, warnings
x = torch.randn(20)
out = torch.empty(21)                      # wrong shape, NOT empty
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    torch.nn.functional.logsigmoid(x, out=out)
print(out.shape, [str(m.message)[:40] for m in w])
# torch.Size([20]) []      <- resized, no warning
torch.sigmoid(x, out=torch.empty(21))      # the sibling op DOES warn
