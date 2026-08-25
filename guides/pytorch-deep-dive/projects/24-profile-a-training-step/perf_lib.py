"""Shared pieces for the Phase-5 (performance) projects 24-29.

One small causal transformer, one timing helper, one activation-byte tracker.
Projects 25-29 import this module via sys.path so that every measurement in the
phase is taken on the *same* model — otherwise "compile is 1.9x faster" and
"accumulation costs 8%" would not be comparable numbers.

The GPU in this machine (GTX 1070 Ti, compute capability sm_61) is older than
anything this PyTorch build ships kernels for, so every project here runs on the
CPU. We disable CUDA explicitly rather than let `torch.cuda.is_available()`
return True and then fail on the first kernel launch.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import math  # noqa: E402
import time  # noqa: E402
import weakref  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

THREADS = 6
torch.set_num_threads(THREADS)

VOCAB = 64
D_MODEL = 256
N_HEAD = 4
N_LAYER = 4
SEQ = 128
BATCH = 16


# ---------------------------------------------------------------------------
# the model: a small causal transformer (Linear + GELU + softmax only, no conv)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d=D_MODEL, h=N_HEAD):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).split(d, dim=2)
        q = q.view(b, t, self.h, d // self.h).transpose(1, 2)
        k = k.view(b, t, self.h, d // self.h).transpose(1, 2)
        v = v.view(b, t, self.h, d // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class TinyGPT(nn.Module):
    def __init__(self, vocab=VOCAB, d=D_MODEL, h=N_HEAD, n_layer=N_LAYER, seq=SEQ):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layer)])
        self.nf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.nf(x))


def new_model(seed=0, **kw):
    torch.manual_seed(seed)
    return TinyGPT(**kw)


def loss_fn(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


# ---------------------------------------------------------------------------
# a synthetic but *learnable* task: next token = (previous token + 1) mod vocab,
# corrupted 10% of the time so the loss cannot reach 0
# ---------------------------------------------------------------------------
def make_batch(batch=BATCH, seq=SEQ, vocab=VOCAB, gen=None):
    gen = gen or torch.Generator().manual_seed(0)
    x = torch.randint(0, vocab, (batch, seq), generator=gen)
    y = (x + 1) % vocab
    noise = torch.rand(y.shape, generator=gen) < 0.10
    y = torch.where(noise, torch.randint(0, vocab, y.shape, generator=gen), y)
    return x, y


# ---------------------------------------------------------------------------
# timing: best-of-N, never mean-of-N (CPU timings have a long right tail)
# ---------------------------------------------------------------------------
def best_of(fn, repeats=5, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3, (max(times) - min(times)) * 1e3   # ms, spread


def cpu_time(fn, repeats=3, warmup=1):
    """CPU-seconds per call, summed over this process's threads.

    Wall-clock time on a shared machine measures the neighbours as much as the
    code: when another process takes a core away, the step gets longer without
    doing more work. `time.process_time` counts only cycles spent in *this*
    process, so it stays stable while the wall clock bounces. It is not a
    substitute for wall clock — a user waits in wall-clock seconds — but it
    tells you whether an optimization removed work or just got lucky.
    """
    for _ in range(warmup):
        fn()
    t0 = time.process_time()
    for _ in range(repeats):
        fn()
    return (time.process_time() - t0) / repeats * 1e3          # ms of CPU time


# ---------------------------------------------------------------------------
# activation memory: count the bytes autograd keeps alive between forward and
# backward. `saved_tensors_hooks` sees every tensor a Function stashes for its
# backward; the weakref decrements when autograd finally lets go, so we get a
# live/peak curve rather than a running total.
# ---------------------------------------------------------------------------
class ActivationBytes:
    def __init__(self, model=None):
        self.live = 0
        self.peak = 0
        self.total = 0
        self._seen = set()
        self._param_ptrs = set()
        if model is not None:
            self._param_ptrs = {p.data_ptr() for p in model.parameters()}

    def _pack(self, t):
        if t.data_ptr() in self._param_ptrs or t.data_ptr() in self._seen:
            return t
        self._seen.add(t.data_ptr())
        nbytes = t.numel() * t.element_size()
        self.live += nbytes
        self.total += nbytes
        self.peak = max(self.peak, self.live)
        holder = _Holder(t)
        weakref.finalize(holder, self._release, nbytes, t.data_ptr())
        return holder

    def _release(self, nbytes, ptr):
        self.live -= nbytes
        self._seen.discard(ptr)

    @staticmethod
    def _unpack(h):
        # tensors we chose not to track (parameters, repeats) are passed through
        return h.t if isinstance(h, _Holder) else h

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._ctx.__enter__()
        return self

    def __exit__(self, *a):
        return self._ctx.__exit__(*a)


class _Holder:
    __slots__ = ("t", "__weakref__")

    def __init__(self, t):
        self.t = t


# ---------------------------------------------------------------------------
# exact byte counts for the other three memory buckets
# ---------------------------------------------------------------------------
def param_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def grad_bytes(model):
    return sum(p.grad.numel() * p.grad.element_size()
               for p in model.parameters() if p.grad is not None)


def optimizer_bytes(opt):
    total = 0
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


MB = 1024 * 1024


def mb(x):
    return x / MB


def human(n):
    return f"{n / MB:.2f} MB" if n >= MB else f"{n / 1024:.1f} KB"


def fmt_flops(n):
    for unit in ("", "K", "M", "G", "T"):
        if abs(n) < 1000:
            return f"{n:.2f} {unit}FLOP"
        n /= 1000
    return f"{n:.2f} PFLOP"


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def analytic_flops(batch=BATCH, seq=SEQ, d=D_MODEL, n_layer=N_LAYER, vocab=VOCAB):
    """6*N*D forward+backward FLOPs, the standard transformer estimate."""
    n_params = 12 * n_layer * d * d + vocab * d
    return 6 * n_params * batch * seq + 12 * n_layer * batch * seq * seq * d


def sqrt(x):
    return math.sqrt(x)
