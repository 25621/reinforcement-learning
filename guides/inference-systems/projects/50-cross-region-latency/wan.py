"""A TCP proxy that adds distance.

There is one machine here, so "two regions" has to be built rather than
rented. This proxy sits between the client and a replica and holds every
byte for `owd` seconds (one-way delay) before passing it on, in both
directions. A request therefore pays `owd` on the way out and `owd` on the
way back -- one round trip per exchange, which is exactly what geography
charges you.

Why a byte-level proxy rather than `time.sleep()` in the client: the thing
being measured is not "latency added to a request", it is *how many times*
the protocol crosses the ocean. A TCP connection costs a round trip to open
(SYN, SYN-ACK) before a single byte of the request is sent; TLS costs one or
two more; only then does the request itself travel. A sleep in the client
would charge exactly one delay and hide all of that. A proxy that delays
real packets charges whatever the protocol actually incurs, so the
connection-reuse result in section C is measured rather than assumed.

The delay is applied to whole chunks as they arrive, which models
propagation delay (how long light takes to get there) and ignores
bandwidth-limited serialisation. For a few kilobytes of prompt and one
token per packet that is the right model: at 10 Gb/s, 4 KB takes 3 us to
put on the wire and 40,000 us to cross the Atlantic.

Real one-way delays used by the projects (typical public-cloud figures):

    same region          0.25 ms   (a rack away)
    same continent      15 ms      (us-east <-> us-central)
    trans-atlantic      40 ms      (us-east <-> eu-west)
    trans-pacific       70 ms      (us-east <-> ap-southeast)
"""

from __future__ import annotations

import asyncio
import threading

OWD = {"same_region": 0.00025, "same_continent": 0.015,
       "trans_atlantic": 0.040, "trans_pacific": 0.070}


class WanProxy(threading.Thread):
    """Listens on `port`, forwards to `target_port`, delaying both directions
    by `owd` seconds. Runs its own asyncio loop on its own thread so a test
    can start several 'regions' at once."""

    def __init__(self, port, target_port, owd, host="127.0.0.1"):
        super().__init__(daemon=True)
        self.port = port
        self.target_port = target_port
        self.owd = owd
        self.host = host
        self.loop = None
        self._ready = threading.Event()
        self.conns = 0

    async def _pipe(self, reader, writer):
        """Forward bytes, delaying each chunk by `owd` -- but *pipelined*.

        The obvious implementation (read, sleep, write, repeat) is wrong, and
        wrong in a way that would invert this project's main result. Sleeping
        inside the read loop means chunk 2 does not even start its delay until
        chunk 1 has been delivered, so a 12-token response pays the delay
        twelve times. Real links do not work that way: packet 2 is launched
        while packet 1 is still in flight, and both arrive one delay after
        they were sent.

        So the reader stamps each chunk with the time it should ARRIVE
        (now + owd) and hands it to a writer task, which sleeps until that
        moment and writes. Chunks queue behind each other only to preserve
        order, never to re-pay the delay. This models propagation delay and
        deliberately ignores bandwidth: correct for a few KB of prompt and
        one token per packet, which is this project's traffic.
        """
        queue = asyncio.Queue()

        async def deliver():
            while True:
                item = await queue.get()
                if item is None:
                    break
                arrive_at, data = item
                wait = arrive_at - asyncio.get_running_loop().time()
                if wait > 0:
                    await asyncio.sleep(wait)
                writer.write(data)
                await writer.drain()

        task = asyncio.create_task(deliver())
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                at = asyncio.get_running_loop().time() + self.owd
                queue.put_nowait((at, data))
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            queue.put_nowait(None)
            try:
                await task
            except (ConnectionResetError, BrokenPipeError):
                pass
            try:
                writer.close()
            except Exception:
                pass

    async def _handle(self, c_reader, c_writer):
        self.conns += 1
        # Opening a TCP connection costs a full ROUND TRIP before the client
        # may send a byte: its SYN crosses, the SYN-ACK comes back, and only
        # then does the request travel. So a new connection is charged 2 x owd
        # here, once, and a reused connection is charged none of it. That gap
        # is exactly what section C measures, and it is why connection pooling
        # is the cheapest cross-region win available.
        if self.owd:
            await asyncio.sleep(2 * self.owd)
        try:
            s_reader, s_writer = await asyncio.open_connection(
                self.host, self.target_port)
        except Exception:
            c_writer.close()
            return
        await asyncio.gather(self._pipe(c_reader, s_writer),
                             self._pipe(s_reader, c_writer))

    async def _serve(self):
        server = await asyncio.start_server(self._handle, self.host, self.port)
        self._ready.set()
        async with server:
            await server.serve_forever()

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        except Exception:
            pass

    def wait_ready(self, timeout=10):
        self._ready.wait(timeout)

    def stop(self):
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
