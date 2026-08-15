"""conv_accel.py - a 3x3 int8 convolution unit, in hardware.

The design is the smallest thing that is still a real CNN accelerator:

    pixels in (one per clock)
        |
    [ line buffer: two rows of W bytes ]      <- so that a 3x3 window can be
        |                                        formed from a 1-D stream
    [ 3x3 window registers ]
        |
    [ 9 multipliers + adder tree ]            <- weight-stationary: weights are
        |                                        loaded once and stay put
    pixels out (one per clock, after the pipeline fills)

Two things a beginner should notice.

*Why a line buffer?* The image arrives one pixel at a time, in reading order.
A 3x3 window needs three pixels from three *different* rows at once. Storing
the last two rows (2 x W bytes) is enough to reconstruct the window without
ever going back to memory - the entire point of a streaming accelerator. A CPU
or GPU solves the same problem with caches; here the "cache" is exactly two
rows long, by construction, and its cost is known at design time.

*Why weight-stationary?* Loading the 9 weights every cycle would need 9 more
bytes/cycle of input bandwidth than the design has pins for. Keeping them in
registers means the only streaming input is the image. This is the same
trade-off a TPU's systolic array makes (project 23) at a different scale.
"""

from amaranth import Array, Elaboratable, Module, Signal, signed


class Conv3x3(Elaboratable):
    """Streaming 3x3 convolution over a W-wide image of int8 pixels.

    Ports
      pix_in / in_valid   : one int8 pixel per clock, in raster order
      w_addr/w_data/w_en  : write port for the 9 int8 weights (weight-stationary)
      pix_out / out_valid : one int32 output per clock, valid convolution only
                            (no padding: an HxW image gives (H-2)x(W-2) outputs)
    """

    def __init__(self, width=8):
        self.W = width
        self.pix_in = Signal(signed(8))
        self.in_valid = Signal()
        self.w_addr = Signal(range(9))
        self.w_data = Signal(signed(8))
        self.w_en = Signal()
        self.pix_out = Signal(signed(24))
        self.out_valid = Signal()

    def elaborate(self, platform):
        m = Module()
        W = self.W

        weights = Array([Signal(signed(8), name=f"wt{i}") for i in range(9)])
        with m.If(self.w_en):
            m.d.sync += weights[self.w_addr].eq(self.w_data)

        # Two delay lines. row_a delays a pixel by W clocks (one image row),
        # row_b by another W, so at any moment the three taps are the same
        # column of three consecutive rows.
        row_a = Array([Signal(signed(8), name=f"a{i}") for i in range(W)])
        row_b = Array([Signal(signed(8), name=f"b{i}") for i in range(W)])

        # 3x3 window: win[r][c], c=2 is the newest column.
        win = [[Signal(signed(8), name=f"w{r}{c}") for c in range(3)]
               for r in range(3)]

        x = Signal(range(W + 1))
        y = Signal(16)
        stage_valid = Signal()

        with m.If(self.in_valid):
            # shift the delay lines
            for i in range(W - 1, 0, -1):
                m.d.sync += row_a[i].eq(row_a[i - 1])
                m.d.sync += row_b[i].eq(row_b[i - 1])
            m.d.sync += row_a[0].eq(self.pix_in)
            m.d.sync += row_b[0].eq(row_a[W - 1])

            # shift the window one column left, insert the new column
            newest = [row_b[W - 1], row_a[W - 1], self.pix_in]   # top, mid, bot
            for r in range(3):
                m.d.sync += win[r][0].eq(win[r][1])
                m.d.sync += win[r][1].eq(win[r][2])
                m.d.sync += win[r][2].eq(newest[r])

            # raster counters
            with m.If(x == W - 1):
                m.d.sync += x.eq(0)
                m.d.sync += y.eq(y + 1)
            with m.Else():
                m.d.sync += x.eq(x + 1)

            # A window is complete once we are at least 2 pixels into a row and
            # at least 2 rows into the image.
            m.d.sync += stage_valid.eq((x >= 2) & (y >= 2))
        with m.Else():
            m.d.sync += stage_valid.eq(0)

        # 9 multiplies and an adder tree, registered on the output.
        acc = sum(win[r][c] * weights[r * 3 + c]
                  for r in range(3) for c in range(3))
        m.d.sync += self.pix_out.eq(acc)
        m.d.sync += self.out_valid.eq(stage_valid)
        return m


def golden(image, weights):
    """The same convolution in Python, for bit-exact comparison. `valid`
    padding: an HxW image with a 3x3 kernel gives (H-2)x(W-2) outputs."""
    H = len(image)
    W = len(image[0])
    out = []
    for r in range(H - 2):
        row = []
        for c in range(W - 2):
            s = 0
            for i in range(3):
                for j in range(3):
                    s += image[r + i][c + j] * weights[i * 3 + j]
            row.append(s)
        out.append(row)
    return out
