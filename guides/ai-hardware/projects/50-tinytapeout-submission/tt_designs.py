"""tt_designs.py - two candidate designs for one TinyTapeout tile.

TinyTapeout gives every project the same tiny interface, and that interface is
the whole design constraint:

    ui_in  [7:0]   dedicated inputs
    uo_out [7:0]   dedicated outputs
    uio    [7:0]   bidirectional (used here as control inputs)
    ena, clk, rst_n

Eight input wires. At the shuttle's typical 50 MHz that is 50 MB/s of data,
total, forever. Every architectural decision below follows from that one number
rather than from how many multipliers we could afford.

Design 1 - `MAC8`: the obvious thing. An 8x8 signed multiply-accumulate. Bytes
arrive alternately as A and B, so one MAC costs two clocks.

Design 2 - `BinaryNeuron`: the same tile spent differently. Weights and
activations are single bits, so one input byte carries eight of them at once;
the multiply becomes XNOR and the sum becomes a population count. Eight MACs
per clock instead of one per two clocks.

Both are real, simulated and synthesized in run.py; the comparison between them
is the project's point.
"""

from amaranth import Cat, Elaboratable, Module, Signal, signed


class MAC8(Elaboratable):
    """8x8 signed multiply-accumulate behind an 8-bit port.

    Control (uio_in): bit0 = load A, bit1 = accumulate with the byte as B,
    bit2 = clear the accumulator. Output byte selected by uio_in[4:3].
    """

    def __init__(self):
        self.ui_in = Signal(8)
        self.uio_in = Signal(8)
        self.uo_out = Signal(8)
        self.acc = Signal(signed(24))       # exposed for the testbench

    def elaborate(self, platform):
        m = Module()
        a = Signal(signed(8))
        load_a = self.uio_in[0]
        do_mac = self.uio_in[1]
        clear = self.uio_in[2]
        sel = self.uio_in[4:6]

        b = Signal(signed(8))
        m.d.comb += b.eq(self.ui_in)

        with m.If(clear):
            m.d.sync += self.acc.eq(0)
        with m.Elif(do_mac):
            m.d.sync += self.acc.eq(self.acc + a * b)
        with m.If(load_a):
            m.d.sync += a.eq(self.ui_in)

        with m.Switch(sel):
            with m.Case(0):
                m.d.comb += self.uo_out.eq(self.acc[0:8])
            with m.Case(1):
                m.d.comb += self.uo_out.eq(self.acc[8:16])
            with m.Default():
                m.d.comb += self.uo_out.eq(self.acc[16:24])
        return m


class BinaryNeuron(Elaboratable):
    """Eight 1-bit weights, eight 1-bit activations, one clock.

    The arithmetic identity that makes this work: if you encode -1 as 0 and +1
    as 1, then multiplying two values is exactly XNOR, and summing eight
    products is exactly "count the ones, double it, subtract eight".

        x*w  |  x=+1,w=+1 -> +1   XNOR(1,1)=1
             |  x=+1,w=-1 -> -1   XNOR(1,0)=0
        sum  =  2 * popcount(XNOR(x, w)) - 8

    So a multiply-accumulate over 8 values costs one XNOR byte, a population
    count and a shift - no multiplier at all. This is why binarized networks
    exist: not because 1-bit weights are accurate, but because they turn the
    most expensive cell in the chip into wiring.
    """

    def __init__(self, n=8):
        self.n = n
        self.ui_in = Signal(8)
        self.uio_in = Signal(8)
        self.uo_out = Signal(8)
        self.weights = Signal(8)            # exposed for the testbench
        self.dot = Signal(signed(8))

    def elaborate(self, platform):
        m = Module()
        load_w = self.uio_in[0]
        thresh = Signal(signed(8))
        load_t = self.uio_in[1]

        with m.If(load_w):
            m.d.sync += self.weights.eq(self.ui_in)
        with m.If(load_t):
            m.d.sync += thresh.eq(self.ui_in)

        agree = Signal(8)
        m.d.comb += agree.eq(~(self.ui_in ^ self.weights))   # XNOR, bitwise
        pop = Signal(4)
        m.d.comb += pop.eq(sum(agree[i] for i in range(8)))  # population count
        m.d.comb += self.dot.eq((pop << 1) - 8)              # 2*pop - 8

        # uo_out: the dot product in the low bits, the fired/not-fired bit on top
        m.d.comb += self.uo_out.eq(Cat(self.dot[0:7], self.dot >= thresh))
        return m


def binary_dot(x_bits, w_bits):
    """Python reference for BinaryNeuron: 2 * popcount(xnor) - 8."""
    agree = ~(x_bits ^ w_bits) & 0xFF
    return 2 * bin(agree).count("1") - 8
