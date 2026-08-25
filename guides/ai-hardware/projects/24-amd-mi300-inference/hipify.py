"""A small hipify: rewrite CUDA source into HIP source by renaming tokens.

AMD's real tool is `hipify-perl` / `hipify-clang` and its substitution table has
thousands of entries. The interesting fact is that the table exists at all:
HIP was designed as a near-copy of the CUDA runtime API, so porting is mostly
a search-and-replace. This file implements the part of the table our sources
actually use, so the claim "it is a rename" can be checked rather than believed.

It also reports what a rename CANNOT fix -- constructs that translate cleanly
and then behave differently on AMD hardware. Those are collected separately by
`landmines()`, and they are where real porting effort goes.
"""

import re

# ------------------------------------------------------------------ renames
# Order matters: longer names first, so `cudaMemcpyHostToDevice` is not
# half-rewritten by the shorter `cudaMemcpy` rule.
RENAMES = [
    (r"\bcuda_runtime\.h\b", "hip/hip_runtime.h"),
    (r"\bcuda_fp16\.h\b", "hip/hip_fp16.h"),
    (r"\bcudaDeviceProp\b", "hipDeviceProp_t"),      # NOT a plain prefix swap
    (r"\bcudaError_t\b", "hipError_t"),
    (r"\bcudaEvent_t\b", "hipEvent_t"),
    (r"\bcudaStream_t\b", "hipStream_t"),
    (r"\bcudaFuncAttributes\b", "hipFuncAttributes"),
    (r"\bcudaMemcpyHostToDevice\b", "hipMemcpyHostToDevice"),
    (r"\bcudaMemcpyDeviceToHost\b", "hipMemcpyDeviceToHost"),
    (r"\bcudaMemcpyDeviceToDevice\b", "hipMemcpyDeviceToDevice"),
    (r"\bcudaSuccess\b", "hipSuccess"),
    (r"\bcuda([A-Z]\w*)", r"hip\1"),                 # the catch-all
    (r"\bcublas([A-Z]\w*)", r"hipblas\1"),
    (r"\bcurand([A-Z]\w*)", r"hiprand\1"),
    (r"\bCUBLAS_(\w+)", r"HIPBLAS_\1"),
    (r"\bCUDA_(\w+)", r"HIP_\1"),
    (r"\bnvtx([A-Z]\w*)", r"roctx\1"),
]

# ---------------------------------------------------------------- landmines
# These all survive the rename and then mean something different on AMD.
LANDMINES = {
    "wavefront-size literal":
        (r"(?:>>\s*5\b|&\s*31\b|%\s*32\b|/\s*32\b|\bWARP_SIZE\b|\b32\s*\*\s*(?:warp|lane))",
         "32 hard-coded as the warp size; an AMD wavefront is 64"),
    "warpSize builtin":
        (r"\bwarpSize\b",
         "compiles on both, but evaluates to 32 on NVIDIA and 64 on AMD"),
    "warp shuffle / vote":
        (r"__(?:shfl|ballot|any|all|activemask)\w*",
         "the mask is 32-bit on NVIDIA and 64-bit on AMD"),
    "inline PTX":
        (r"asm\s+volatile", "PTX is NVIDIA machine code; AMD needs GCN/RDNA ISA"),
    "wmma / tensor cores":
        (r"\bwmma::", "NVIDIA tensor-core API; AMD's equivalent is rocWMMA"),
    "__ldg":
        (r"\b__ldg\b", "read-only cache hint; a no-op on AMD, silently"),
    "cuBLAS":
        (r"\bcublas", "hipBLAS is API-compatible but not ABI-compatible"),
}


def hipify_text(src):
    """Return (ported source, {rule: hits})."""
    counts = {}
    out = src
    for pat, rep in RENAMES:
        out, n = re.subn(pat, rep, out)
        if n:
            counts[pat] = counts.get(pat, 0) + n
    return out, counts


def landmines(src):
    """Things the rename leaves looking correct and behaving differently."""
    found = {}
    for name, (pat, why) in LANDMINES.items():
        hits = re.findall(pat, src)
        if hits:
            found[name] = dict(count=len(hits), why=why)
    return found


def line_stats(before, after):
    a, b = before.splitlines(), after.splitlines()
    changed = sum(1 for x, y in zip(a, b) if x != y)
    return dict(lines=len(a), changed=changed,
                unchanged_pct=round(100.0 * (len(a) - changed) / max(len(a), 1), 1))
