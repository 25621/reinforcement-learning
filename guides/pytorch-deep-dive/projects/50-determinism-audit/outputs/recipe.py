"""The determinism recipe this project's audit arrived at.

Copy this into your project. Every line is here because removing it broke
reproducibility in the measured ablation (section 3 of the README) — except
where noted.
"""
import os, random
import numpy as np
import torch


def set_determinism(seed: int = 0, threads: int | None = None) -> None:
    # 1. every random-number generator that will be asked for a number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                 # also seeds CUDA, if present

    # 2. refuse to run any kernel that has no deterministic implementation.
    #    On CPU this blocks very little (see section 4) - it is cheap insurance
    #    that pays off the day the same code runs on a GPU.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False  # stop cuDNN picking a different
                                            # algorithm run to run

    # 3. the thread count changes the summation order of every parallel
    #    reduction, so it is part of the seed whether you like it or not.
    if threads is not None:
        torch.set_num_threads(threads)


def seed_worker(worker_id: int) -> None:
    """Pass as DataLoader(worker_init_fn=seed_worker).

    PyTorch seeds each worker's torch generator; it does NOT seed numpy or
    Python's random inside the worker. Without this, any np.random augmentation
    is unseeded and the run is not reproducible.
    """
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, **kw):
    g = torch.Generator()
    g.manual_seed(0)                        # shuffling has its own generator
    return torch.utils.data.DataLoader(
        dataset, generator=g, worker_init_fn=seed_worker, **kw)


# And two rules no function can enforce for you:
#   * never derive anything order-dependent from a set or an unordered dict -
#     sort it (or set PYTHONHASHSEED, which every caller must remember).
#   * record the torch version, the thread count and the device with the
#     checkpoint. Bit-exactness is a promise about a configuration, not a model.
