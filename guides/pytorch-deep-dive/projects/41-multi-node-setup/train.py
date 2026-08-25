"""The script `torchrun` launches — one copy per process, on every node.

This is a complete, ordinary DDP training script. Nothing in it knows how many
machines there are: it reads the environment variables torchrun sets and joins
the group. Run it with, for example:

    torchrun --nnodes=2 --node-rank=0 --nproc-per-node=2 \
             --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29500 --rdzv-id=job1 \
             train.py --out /tmp/run

and the same command with --node-rank=1 on the second machine.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def env_identity():
    """The five variables torchrun sets, and what each is for."""
    return {
        "RANK": os.environ.get("RANK"),                    # global index, 0..WORLD_SIZE-1
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),        # index within this machine
        "GROUP_RANK": os.environ.get("GROUP_RANK"),        # which machine (node) this is
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),        # total processes
        "LOCAL_WORLD_SIZE": os.environ.get("LOCAL_WORLD_SIZE"),
        "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
        "MASTER_PORT": os.environ.get("MASTER_PORT"),
        "TORCHELASTIC_RESTART_COUNT": os.environ.get("TORCHELASTIC_RESTART_COUNT"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--crash-at", type=int, default=-1,
                    help="exit(1) at this step, to demonstrate torchrun restarts")
    ap.add_argument("--crash-rank", type=int, default=1,
                    help="which global rank plays the process that dies")
    ap.add_argument("--checkpoint-mode", default="rank0",
                    choices=["rank0", "everyone", "local_rank0"])
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    ident = env_identity()

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))
    ddp = DDP(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)

    g = torch.Generator().manual_seed(rank)
    losses = []
    restart = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
    for step in range(args.steps):
        if step == args.crash_at and restart == 0 and rank == args.crash_rank:
            # only crash on the first attempt, so the restart can succeed
            print(f"rank {rank}: simulated failure at step {step}", flush=True)
            os._exit(1)
        x = torch.randn(8, 32, generator=g)
        y = torch.randint(0, 8, (8,), generator=g)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(ddp(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())

    # --- checkpointing: who is allowed to write? ---------------------------
    ckpt = os.path.join(args.out, "checkpoint.pt")
    wrote = False
    if args.checkpoint_mode == "everyone":
        torch.save({"model": model.state_dict(), "by": rank}, ckpt)
        wrote = True
    elif args.checkpoint_mode == "rank0" and rank == 0:
        torch.save({"model": model.state_dict(), "by": rank}, ckpt)
        wrote = True
    elif args.checkpoint_mode == "local_rank0" and ident["LOCAL_RANK"] == "0":
        torch.save({"model": model.state_dict(), "by": rank},
                   os.path.join(args.out, f"checkpoint_node{ident['GROUP_RANK']}.pt"))
        wrote = True
    dist.barrier()          # nobody exits before the write is finished

    wsum = float(sum(p.detach().sum() for p in model.parameters()))
    ident.update({"final_loss": losses[-1], "wsum": wsum, "wrote_checkpoint": wrote,
                  "restart_count": restart, "t": time.time()})
    with open(os.path.join(args.out, f"rank{rank}.json"), "w") as f:
        json.dump(ident, f)
    print(f"rank {rank}/{world} local_rank {ident['LOCAL_RANK']} "
          f"node {ident['GROUP_RANK']} done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
