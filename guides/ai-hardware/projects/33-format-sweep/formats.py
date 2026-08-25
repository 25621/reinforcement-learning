"""Floating-point formats, built from their bit layout.

Every format in the guide's table is the same three fields -- sign, exponent,
mantissa -- with different widths. So instead of six special cases we write one
function parameterized by (exponent bits E, mantissa bits M) and check it against
the four formats PyTorch can actually cast to (bf16, fp16, fp8 e4m3, fp8 e5m2).

Vocabulary, decoded:
  * "mantissa" is Latin for "makeshift addition" -- the leftover fractional part.
    It is also called the "significand": the digits that carry the *significance*
    of the number, as opposed to the exponent, which only says where the decimal
    (binary) point goes.
  * "E4M3" is just a spelling of the widths: 4 exponent bits, 3 mantissa bits.
  * "bfloat16" = "brain float", from Google Brain, where it was designed. It is
    fp32 with 16 of the mantissa bits thrown away, so the exponent -- the *range* --
    is untouched.
  * "subnormal" (or "denormal") numbers are the ones below the smallest normal
    value. They give up the implicit leading 1 to reach closer to zero, one last
    gentle slope before the number underflows to 0.0.
"""

import math

import torch

# overflow policy: what happens to a value larger than the format can hold
INF = "inf"            # IEEE style: becomes +/-infinity
NAN = "nan"            # FP8 E4M3 style ("fn" = finite, no inf): becomes NaN
SATURATE = "saturate"  # FP4 / integer style: clamps to the largest value


class FloatFormat:
    def __init__(self, name, exp_bits, mant_bits, overflow=INF,
                 nan_takes_top_mantissa=False, torch_dtype=None):
        self.name = name
        self.E = exp_bits
        self.M = mant_bits
        self.overflow = overflow
        self.bias = 2 ** (exp_bits - 1) - 1
        self.torch_dtype = torch_dtype

        # Which exponent field is the largest one that holds real numbers?
        # IEEE reserves the all-ones field for inf/NaN; the "fn" FP8 variant does
        # not, it only reserves the single all-ones-mantissa code inside it.
        top_field = 2 ** exp_bits - 1 - (1 if overflow == INF else 0)
        top_mant = 2 ** mant_bits - 1 - (1 if nan_takes_top_mantissa else 0)
        self.max_exp = top_field - self.bias
        self.max_value = (1 + top_mant / 2 ** mant_bits) * 2.0 ** self.max_exp
        self.min_normal = 2.0 ** (1 - self.bias)
        self.min_subnormal = 2.0 ** (1 - self.bias - mant_bits)
        self.eps = 2.0 ** -mant_bits           # gap between 1.0 and the next value
        self.bits = 1 + exp_bits + mant_bits

    def __repr__(self):
        return f"{self.name}(1+{self.E}+{self.M})"

    def cast(self, x):
        """Round x to the nearest value this format can represent (result stays fp32)."""
        x = x.float()
        sign = torch.sign(x)
        ax = x.abs()

        # frexp gives an exact base-2 exponent; log2+floor would be off by one
        # at exact powers of two because of rounding in the log.
        _, exp = torch.frexp(ax)
        e = (exp - 1).clamp(min=1 - self.bias, max=self.max_exp).float()

        step = torch.pow(torch.tensor(2.0), e - self.M)   # spacing in this binade
        q = torch.round(ax / step) * step                 # round-half-to-even
        q = torch.where(ax == 0, torch.zeros_like(q), q)

        over = q > self.max_value
        if self.overflow == INF:
            q = torch.where(over, torch.full_like(q, float("inf")), q)
        elif self.overflow == NAN:
            q = torch.where(over, torch.full_like(q, float("nan")), q)
        else:
            q = torch.where(over, torch.full_like(q, self.max_value), q)
        return sign * q

    def describe(self):
        return {
            "name": self.name, "bits": self.bits, "exp": self.E, "mant": self.M,
            "max": self.max_value, "min_normal": self.min_normal,
            "min_subnormal": self.min_subnormal, "eps": self.eps,
            "values_in_[1,2)": 2 ** self.M,
            "decimal_digits": round(-math.log10(self.eps), 1),
        }


FP32 = FloatFormat("FP32", 8, 23, INF, torch_dtype=torch.float32)
TF32 = FloatFormat("TF32", 8, 10, INF)
BF16 = FloatFormat("BF16", 8, 7, INF, torch_dtype=torch.bfloat16)
FP16 = FloatFormat("FP16", 5, 10, INF, torch_dtype=torch.float16)
E4M3 = FloatFormat("FP8 E4M3", 4, 3, NAN, nan_takes_top_mantissa=True,
                   torch_dtype=torch.float8_e4m3fn)
E5M2 = FloatFormat("FP8 E5M2", 5, 2, INF, torch_dtype=torch.float8_e5m2)
E2M1 = FloatFormat("FP4 E2M1", 2, 1, SATURATE)

ALL = [FP32, TF32, BF16, FP16, E4M3, E5M2, E2M1]


def bits_of(x):
    """The 32 raw bits of an fp32 value, split into sign / exponent / mantissa."""
    t = torch.tensor([x], dtype=torch.float32)
    raw = t.view(torch.int32).item() & 0xFFFFFFFF
    s = (raw >> 31) & 1
    e = (raw >> 23) & 0xFF
    m = raw & 0x7FFFFF
    return s, e, m, f"{s:01b} {e:08b} {m:023b}"


def grid(fmt, limit=None):
    """Every positive value the format can represent, in order (small formats only)."""
    out = []
    for field in range(2 ** fmt.E):
        for mant in range(2 ** fmt.M):
            if fmt.overflow == INF and field == 2 ** fmt.E - 1:
                continue
            if field == 0:
                v = mant / 2 ** fmt.M * 2.0 ** (1 - fmt.bias)          # subnormal
            else:
                v = (1 + mant / 2 ** fmt.M) * 2.0 ** (field - fmt.bias)
            if v > fmt.max_value + 1e-9:
                continue
            out.append(v)
    out = sorted(set(out))
    return out if limit is None else [v for v in out if v <= limit]


def scaled_cast(x, fmt, per_tensor=True):
    """Cast through a format after rescaling so the largest value lands at the top.

    This is what FP8 training actually does. Without it, a tensor whose values all
    sit near 1e-4 falls off the bottom of E4M3 (its smallest subnormal is ~2e-3)
    and every element becomes zero. The scale is a single fp32 number kept next to
    the tensor, so it costs nothing per element.
    """
    amax = x.abs().amax() if per_tensor else x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, fmt.max_value / amax, torch.ones_like(amax))
    return fmt.cast(x * scale) / scale


class STECast(torch.autograd.Function):
    """Cast in the forward pass, pass the gradient through untouched in the backward.

    Rounding has a derivative of zero almost everywhere, so a naive autograd
    version would train nothing. The straight-through estimator (STE) pretends
    the rounding was the identity function when computing gradients -- the
    standard trick for every quantization-aware training recipe.
    """

    @staticmethod
    def forward(ctx, x, fmt, scaled):
        return scaled_cast(x, fmt) if scaled else fmt.cast(x)

    @staticmethod
    def backward(ctx, g):
        return g, None, None


def ste_cast(x, fmt, scaled=False):
    return STECast.apply(x, fmt, scaled)
