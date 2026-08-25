"""Put this at the top of every long-running job you own."""
import faulthandler
import os
import signal

# 1. If the process dies from a segfault, print C and Python stacks.
faulthandler.enable()

# 2. If you send it SIGUSR1, print every thread's Python stack and KEEP RUNNING.
#    This is what lets you diagnose a hang without killing the job.
_dump = open(f"/tmp/stacks-{os.getpid()}.log", "w")
faulthandler.register(signal.SIGUSR1, file=_dump, all_threads=True)

#    Then, from any shell:   kill -USR1 <pid>   and read the file.

# 3. For distributed jobs, also cap the collective wait so an infinite hang
#    becomes an exception with a message (see project 40):
#
#    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=120))
