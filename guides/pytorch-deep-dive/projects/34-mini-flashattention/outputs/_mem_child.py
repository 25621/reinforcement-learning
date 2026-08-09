
import os, sys, json, resource, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, sys.argv[5])       # project 30, for kernels_lib
import torch, torch.nn.functional as F
import kernels_lib as K

# `import run` would be ambiguous: several projects in this guide have a
# run.py, and kernels_lib itself puts project 24's directory on sys.path to
# reuse perf_lib. Loading THIS project's file by its full path removes the
# guesswork -- an `import` that depends on sys.path order is a bug waiting.
_spec = importlib.util.spec_from_file_location(
    "p34_run", os.path.join(sys.argv[4], "run.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

impl, T = sys.argv[1], int(sys.argv[2])
H, D = P.NH, P.DH
torch.manual_seed(0)
q = torch.randn(H, T, D); k = torch.randn(H, T, D); v = torch.randn(H, T, D)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

if impl == "eager":
    P.eager_attention(q, k, v)
elif impl == "sdpa":
    F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
else:
    mod, _ = K.build("p34_flash", P.CPP, functions=P.FUNCS, extra_cflags=P.AVX2_FLAGS)
    mod.flash_attention_vec(q, k, v, P.BR, P.BC, False)

peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"impl": impl, "T": T, "base_kb": base, "peak_kb": peak}))
