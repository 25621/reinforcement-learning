"""Mini LAION pipeline: turn a dirty crawl into a training-ready shard.

Five stages, each safe to re-run (everything is cached):

    crawl      download 2,400 real photos and inject the defects   (~4 min once)
    filter     dedup -> size -> text -> CLIP score, with per-filter
               precision/recall against the known truth            (~4 min)
    recaption  rewrite the survivors' captions with BLIP           (~5 min)
    shard      write real WebDataset .tar shards                   (~10 s)
    plot       every figure and table in the README                (~20 s)

Run them in order:

    python3 run.py --stage crawl
    python3 run.py --stage filter
    python3 run.py --stage recaption
    python3 run.py --stage shard
    python3 run.py --stage plot
"""

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(HERE))
import pipeline_lib as P  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
DATA = P.data_dir()
torch.set_num_threads(6)

# The CLIP-score rule. LAION-2B-en kept pairs above a raw cosine of 0.28; we
# report both that fixed threshold and a "keep the best 45%" percentile rule,
# because which one you use changes how much data you end up with.
CLIP_KEEP_FRACTION = 0.45
LAION_THRESHOLD = 0.28


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


def _pr(pred_drop, should_drop):
    """Precision / recall of one filter, treating "drop" as the positive class."""
    pred_drop = np.asarray(pred_drop, dtype=bool)
    should_drop = np.asarray(should_drop, dtype=bool)
    tp = int((pred_drop & should_drop).sum())
    fp = int((pred_drop & ~should_drop).sum())
    fn = int((~pred_drop & should_drop).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"dropped": int(pred_drop.sum()), "true_positives": tp,
            "false_positives": fp, "missed": fn,
            "precision": prec, "recall": rec,
            "f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)}


# ---------------------------------------------------------------------------
# stage: crawl
# ---------------------------------------------------------------------------
def stage_crawl(args):
    t0 = time.time()
    crawl = P.build_crawl(args.n_base)
    counts = {d: int(sum(1 for x in crawl["defect"] if x == d)) for d in P.DEFECTS}
    stats = {
        "records": len(crawl["alt"]),
        "base_photos": int(args.n_base),
        "by_defect": counts,
        "clean_share": counts["ok"] / len(crawl["alt"]),
        "seconds": time.time() - t0,
    }
    _save("crawl.json", stats)
    samples = []
    for d in P.DEFECTS:
        i = next(k for k, x in enumerate(crawl["defect"]) if x == d)
        samples.append({"defect": d, "alt": crawl["alt"][i],
                        "declared_size": [int(crawl["w"][i]), int(crawl["h"][i])]})
    _save("crawl_samples.json", samples)
    print(json.dumps(stats, indent=1))


# ---------------------------------------------------------------------------
# stage: filter
# ---------------------------------------------------------------------------
def stage_filter(args):
    crawl = P.build_crawl(args.n_base)
    defect = np.array(crawl["defect"], dtype=object)
    n = len(defect)
    funnel = [{"stage": "raw crawl", "kept": n, "dropped": 0, "seconds": 0.0}]
    report = {}

    # --- 1. near-duplicate removal -----------------------------------------
    # Scored per *group*, not per record. A duplicate group is the original
    # photo plus its copies; the job of the filter is to leave exactly one of
    # them standing, and it does not matter which one it picks. Grading each
    # record against its own label would punish the filter for keeping the copy
    # and dropping the original, which is not a mistake.
    groups = {}
    for i, s in enumerate(crawl["src"]):
        groups.setdefault(int(s), []).append(i)
    dup_groups = {s: g for s, g in groups.items() if len(g) > 1}
    solo = np.array([len(groups[int(s)]) == 1 for s in crawl["src"]])

    def grade(keep):
        left = {s: sum(1 for i in g if keep[i]) for s, g in dup_groups.items()}
        return {
            "duplicate_groups": len(dup_groups),
            "groups_collapsed_to_one": sum(1 for v in left.values() if v == 1),
            "groups_still_duplicated": sum(1 for v in left.values() if v > 1),
            "groups_wiped_out": sum(1 for v in left.values() if v == 0),
            "unique_images_destroyed": int((solo & ~keep).sum()),
            "unique_images": int(solo.sum()),
        }

    # The baseline anyone reaches for first: hash the raw bytes and drop repeats.
    seen, keep_bytes = set(), np.ones(n, dtype=bool)
    t0 = time.time()
    for i, a in enumerate(crawl["images"]):
        h = hash(a.tobytes())
        keep_bytes[i] = h not in seen
        seen.add(h)
    report["exact_byte_dedup"] = grade(keep_bytes)
    report["exact_byte_dedup"]["seconds"] = time.time() - t0

    hashes = {}
    t_hash = {}
    for name, fn in (("dhash", P.dhash), ("phash", P.phash)):
        t0 = time.time()
        hashes[name] = [fn(a) for a in crawl["images"]]
        t_hash[name] = time.time() - t0

    # Which hash, at which cut-off? Sweep both and read the trade-off directly.
    sweep = []
    for name in ("dhash", "phash"):
        for t in (0, 2, 4, 6, 8, 12, 16):
            k, _, _ = P.dedup(hashes[name], max_dist=t)
            sweep.append({"hash": name, "max_dist": t, **grade(k)})
    report["dedup_threshold_sweep"] = sweep

    t0 = time.time()
    keep_dup, dup_of, comparisons = P.dedup(hashes[args.hash], max_dist=args.max_dist)
    t_dedup = time.time() - t0 + t_hash[args.hash]
    report["dedup"] = grade(keep_dup)
    report["dedup"].update(hash=args.hash, max_dist=args.max_dist,
                           pair_comparisons=comparisons,
                           brute_force_pairs=n * (n - 1) // 2,
                           seconds=t_dedup, hash_seconds=t_hash)
    # How far each kind of copy drifted from its original, in bits.
    dists = {"dup_exact": [], "dup_near": []}
    for i, j in dup_of.items():
        if defect[i] in dists:
            dists[defect[i]].append(P.hamming(hashes[args.hash][i],
                                              hashes[args.hash][j]))
    report["dedup"]["caught_by_kind"] = {
        k: {"caught": len(v), "total": int((defect == k).sum()),
            "mean_hamming": float(np.mean(v)) if v else None} for k, v in dists.items()}
    funnel.append({"stage": "dedup", "kept": int(keep_dup.sum()),
                   "dropped": int((~keep_dup).sum()), "seconds": t_dedup})

    # --- 2. resolution / aspect ratio --------------------------------------
    alive = keep_dup.copy()
    t0 = time.time()
    keep_size = P.size_filter(crawl["w"], crawl["h"])
    t_size = time.time() - t0
    drop = alive & ~keep_size
    report["size"] = _pr(drop, alive & np.isin(defect, ["tiny", "banner"]))
    report["size"]["seconds"] = t_size
    alive = alive & keep_size
    funnel.append({"stage": "size/aspect", "kept": int(alive.sum()),
                   "dropped": int(drop.sum()), "seconds": t_size})

    # --- 3. alt-text quality + blocklist -----------------------------------
    t0 = time.time()
    keep_txt, reasons = P.text_filter(crawl["alt"])
    t_txt = time.time() - t0
    drop = alive & ~keep_txt
    report["text"] = _pr(drop, alive & np.isin(defect, ["boilerplate", "blocked"]))
    report["text"]["seconds"] = t_txt
    report["text"]["by_reason"] = {
        r: int(sum(1 for i in range(n) if alive[i] and reasons[i] == r))
        for r in ("blocked", "boilerplate", "too_short", "not_prose")}
    alive = alive & keep_txt
    funnel.append({"stage": "text quality", "kept": int(alive.sum()),
                   "dropped": int(drop.sum()), "seconds": t_txt})

    # --- 4. CLIP score ------------------------------------------------------
    # Scored for EVERY record, not just the survivors, so we can (a) show what
    # the score looks like per defect kind and (b) price the two possible
    # filter orders honestly.
    model, tok = P.load_clip()
    scores, t_all = P.clip_scores(crawl["images"], crawl["alt"], model, tok)
    np.save(DATA / "clip_scores.npy", scores)
    live = np.flatnonzero(alive)
    t0 = time.time()
    _ = P.clip_scores(crawl["images"][live], [crawl["alt"][i] for i in live],
                      model, tok, verbose=False)
    t_live = time.time() - t0

    cut = float(np.quantile(scores[alive], 1 - CLIP_KEEP_FRACTION))
    keep_clip = scores >= cut
    drop = alive & ~keep_clip

    def clip_report(keep):
        """Recall on the pairs we know are wrong, and the price paid in good
        pairs. "Precision" is the wrong word for this filter: it deliberately
        throws away many genuine-but-weakly-matching pairs, and that is not a
        mistake -- it is the trade the filter exists to make."""
        d = alive & ~keep
        mm = alive & (defect == "mismatch")
        ok = alive & (defect == "ok")
        return {
            "kept": int((alive & keep).sum()),
            "mismatch_caught": int((d & mm).sum()),
            "mismatch_total": int(mm.sum()),
            "mismatch_recall": float((d & mm).sum() / max(mm.sum(), 1)),
            "genuine_kept": int((ok & keep).sum()),
            "genuine_total": int(ok.sum()),
            "genuine_kept_fraction": float((ok & keep).sum() / max(ok.sum(), 1)),
            "purity_after": float((ok & keep).sum() / max((alive & keep).sum(), 1)),
        }

    report["clip"] = clip_report(keep_clip)
    report["clip"].update(threshold=cut, keep_fraction=CLIP_KEEP_FRACTION,
                          seconds_all_records=t_all, seconds_survivors_only=t_live,
                          scored_all=int(n), scored_survivors=int(alive.sum()))
    report["clip"]["laion_threshold_rule"] = {
        "threshold": LAION_THRESHOLD, **clip_report(scores >= LAION_THRESHOLD)}
    report["clip"]["keep_fraction_sweep"] = [
        {"keep": f, **clip_report(scores >= float(np.quantile(scores[alive], 1 - f)))}
        for f in (0.9, 0.7, 0.55, 0.45, 0.3, 0.2, 0.1)]
    by_defect = {d: {"n": int((defect == d).sum()),
                     "mean_clip": float(scores[defect == d].mean()),
                     "std_clip": float(scores[defect == d].std())}
                 for d in P.DEFECTS}
    report["clip"]["by_defect"] = by_defect
    # Separability of genuine vs swapped captions, as an AUC (project 14's number).
    good = scores[(defect == "ok")]
    bad = scores[(defect == "mismatch")]
    auc = float((good[:, None] > bad[None, :]).mean()
                + 0.5 * (good[:, None] == bad[None, :]).mean())
    report["clip"]["auc_ok_vs_mismatch"] = auc
    alive = alive & keep_clip
    funnel.append({"stage": "CLIP score", "kept": int(alive.sum()),
                   "dropped": int(drop.sum()), "seconds": t_all})

    # --- what the order of the filters bought us ---------------------------
    per_record = t_all / n
    cheap_first = t_dedup + t_size + t_txt + t_live
    clip_first = t_all + t_dedup + t_size + t_txt
    report["ordering"] = {
        "records_scored_cheap_first": int(len(live)),
        "records_scored_clip_first": int(n),
        "seconds_cheap_first": cheap_first,
        "seconds_clip_first": clip_first,
        "clip_seconds_per_record": per_record,
        "speedup": clip_first / max(cheap_first, 1e-9),
    }

    survivors = np.flatnonzero(alive)
    kept_defects = {d: int((defect[survivors] == d).sum()) for d in P.DEFECTS}
    report["funnel"] = funnel
    report["final"] = {
        "kept": int(len(survivors)),
        "of_crawl": len(survivors) / n,
        "purity": kept_defects["ok"] / max(len(survivors), 1),
        "kept_by_defect": kept_defects,
        "raw_purity": float((defect == "ok").mean()),
    }
    np.save(DATA / "survivors.npy", survivors)
    np.save(DATA / "alive_after_cheap.npy", live)
    _save("filter.json", report)
    print(json.dumps({k: report[k] for k in ("final", "ordering")}, indent=1))


# ---------------------------------------------------------------------------
# stage: recaption
# ---------------------------------------------------------------------------
def stage_recaption(args):
    crawl = P.build_crawl(args.n_base)
    ids = np.load(DATA / "survivors.npy")
    if args.limit:
        ids = ids[:args.limit]
    cache, secs = P.recaption(crawl["images"], ids)
    per = secs / max(len(ids), 1)
    old = [crawl["alt"][i] for i in ids]
    new = [cache[str(int(i))] for i in ids]
    stats = {
        "n": int(len(ids)),
        "seconds": secs,
        "seconds_per_image": per,
        "alt_mean_words": float(np.mean([len(t.split()) for t in old])),
        "recap_mean_words": float(np.mean([len(t.split()) for t in new])),
        "alt_unique": len(set(t.lower() for t in old)) / len(old),
        "recap_unique": len(set(t.lower() for t in new)) / len(new),
    }
    # Does recaptioning actually raise the image-text agreement? Same frozen
    # CLIP, same images, only the text changed.
    #
    # This is measured on TWO populations, and they disagree, which is the whole
    # point. The CLIP-selected survivors were chosen *because* their alt-text
    # scores high, so comparing a fresh caption against them is a rigged contest
    # (selection bias). The cheap-filter pool -- everything that passed dedup,
    # size and text rules, before CLIP had a say -- is the fair one.
    model, tok = P.load_clip()
    defect = np.array(crawl["defect"], dtype=object)
    pools = {"clip_selected": ids}
    cheap = DATA / "alive_after_cheap.npy"
    if cheap.exists():
        pool = np.load(cheap)
        have = np.array([i for i in pool if str(int(i)) in cache])
        if len(have) > 50:
            pools["cheap_filtered"] = have
    stats["populations"] = {}
    for name, sel in pools.items():
        a = [crawl["alt"][i] for i in sel]
        b = [cache[str(int(i))] for i in sel]
        s_old, _ = P.clip_scores(crawl["images"][sel], a, model, tok, verbose=False)
        s_new, _ = P.clip_scores(crawl["images"][sel], b, model, tok, verbose=False)
        d = defect[sel]
        stats["populations"][name] = {
            "n": int(len(sel)),
            "clip_alt": float(s_old.mean()),
            "clip_recap": float(s_new.mean()),
            "recap_wins": float((s_new > s_old).mean()),
            "genuine_share": float((d == "ok").mean()),
            # Broken out by whether the alt-text was any good to begin with:
            # recaptioning can only help where the original text was wrong.
            "by_defect": {k: {"n": int((d == k).sum()),
                              "clip_alt": float(s_old[d == k].mean()),
                              "clip_recap": float(s_new[d == k].mean())}
                          for k in ("ok", "mismatch") if (d == k).sum() > 5},
        }
    stats.update(clip_alt=stats["populations"]["clip_selected"]["clip_alt"],
                 clip_recap=stats["populations"]["clip_selected"]["clip_recap"],
                 recap_wins=stats["populations"]["clip_selected"]["recap_wins"])
    _save("recaption.json", stats)
    _save("recaption_examples.json",
          [{"id": int(i), "alt": a, "blip": b, "human": crawl["human"][i][0]}
           for i, a, b in list(zip(ids, old, new))[:12]])
    print(json.dumps(stats, indent=1))


# ---------------------------------------------------------------------------
# stage: shard
# ---------------------------------------------------------------------------
def stage_shard(args):
    """Write the survivors as real WebDataset shards.

    A shard is just a .tar file whose members are grouped by filename stem:
    ``000123.jpg`` and ``000123.txt`` belong to the same sample. Training reads
    the tar front to back, so one disk seek streams thousands of examples --
    which is why every large-scale image-text loader uses this format instead of
    millions of loose files.
    """
    crawl = P.build_crawl(args.n_base)
    ids = np.load(DATA / "survivors.npy")
    caps = P.load_recaptions()
    shard_dir = DATA / "shards"
    shard_dir.mkdir(exist_ok=True)
    per_shard = args.per_shard
    written, files = 0, []
    for s in range(0, len(ids), per_shard):
        chunk = ids[s:s + per_shard]
        path = shard_dir / f"mini-laion-{s // per_shard:05d}.tar"
        with tarfile.open(path, "w") as tar:
            for i in chunk:
                key = f"{int(i):06d}"
                buf = io.BytesIO()
                Image.fromarray(crawl["images"][i]).save(buf, format="JPEG", quality=90)
                for ext, payload in (
                        ("jpg", buf.getvalue()),
                        ("txt", caps.get(str(int(i)), crawl["alt"][i]).encode()),
                        ("json", json.dumps({
                            "alt": crawl["alt"][i],
                            "recaption": caps.get(str(int(i))),
                            "width": int(crawl["w"][i]), "height": int(crawl["h"][i]),
                        }).encode())):
                    info = tarfile.TarInfo(f"{key}.{ext}")
                    info.size = len(payload)
                    tar.addfile(info, io.BytesIO(payload))
                written += 1
        files.append({"file": path.name, "samples": len(chunk),
                      "megabytes": path.stat().st_size / 1e6})
    _save("shards.json", {"samples": written, "shards": files,
                          "bytes_per_sample": sum(f["megabytes"] for f in files)
                          * 1e6 / max(written, 1)})
    # A tiny preview shard is committed so the layout is visible in the repo.
    prev = OUT / "shard_preview.tar"
    with tarfile.open(shard_dir / files[0]["file"]) as src, \
            tarfile.open(prev, "w") as dst:
        for m in src.getmembers()[:24]:
            dst.addfile(m, src.extractfile(m))
    print(f"wrote outputs/{prev.name} ({prev.stat().st_size / 1e3:.0f} kB)")


# ---------------------------------------------------------------------------
# stage: plot
# ---------------------------------------------------------------------------
def stage_plot(args):
    crawl = P.build_crawl(args.n_base)
    rep = json.loads((OUT / "filter.json").read_text())
    defect = np.array(crawl["defect"], dtype=object)
    scores = np.load(DATA / "clip_scores.npy")

    # 1. the funnel
    fig, ax = ps.new_axes(7.6, 4.0)
    names = [f["stage"] for f in rep["funnel"]]
    kept = [f["kept"] for f in rep["funnel"]]
    ax.barh(range(len(names))[::-1], kept, color=ps.SERIES[0], height=0.6)
    for y, (k, name) in enumerate(zip(kept, names)):
        ax.text(k + max(kept) * 0.012, len(names) - 1 - y,
                f"{k:,}  ({100 * k / kept[0]:.0f}%)", va="center",
                fontsize=9, color=ps.INK_SECONDARY)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, max(kept) * 1.22)
    ps.finish(fig, ax, "Every filter shrinks the pile", "records surviving", "",
              OUT / "funnel.png")

    # 2. CLIP score by defect kind
    fig, ax = ps.new_axes(7.6, 4.2)
    show = ["ok", "mismatch", "boilerplate", "tiny", "banner"]
    bins = np.linspace(0.05, 0.42, 46)
    for k, d in enumerate(show):
        s = scores[defect == d]
        ax.hist(s, bins=bins, density=True, histtype="step", linewidth=1.8,
                color=ps.SERIES[k], label=f"{d}  (mean {s.mean():.3f})")
    ax.axvline(rep["clip"]["threshold"], color=ps.INK_MUTED, linestyle="--",
               linewidth=1.2)
    ax.text(rep["clip"]["threshold"], ax.get_ylim()[1] * 0.95,
            f"  keep-{int(100 * CLIP_KEEP_FRACTION)}% cut", fontsize=8,
            color=ps.INK_MUTED, va="top")
    ax.legend(frameon=False, fontsize=8.5)
    ps.finish(fig, ax, "What the CLIP score sees",
              "CLIP image-text cosine", "density", OUT / "clip_scores.png")

    # 3. duplicate-threshold sweep, both hashes
    fig, ax = ps.new_axes(7.2, 4.2)
    sw = rep["dedup_threshold_sweep"]
    for k, h in enumerate(("dhash", "phash")):
        rows = [s for s in sw if s["hash"] == h]
        ng = rows[0]["duplicate_groups"]
        nu = rows[0]["unique_images"]
        ax.plot([s["max_dist"] for s in rows],
                [s["groups_collapsed_to_one"] / ng for s in rows], "-o",
                color=ps.SERIES[k], label=f"{h}: duplicate groups resolved")
        ax.plot([s["max_dist"] for s in rows],
                [s["unique_images_destroyed"] / nu for s in rows], "--o",
                color=ps.SERIES[k], markerfacecolor=ps.SURFACE,
                label=f"{h}: unique images destroyed")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8.5)
    ps.finish(fig, ax, "How strict should the duplicate test be?",
              "maximum Hamming distance treated as a duplicate",
              "fraction", OUT / "dedup_sweep.png")

    # 4. a montage: what got dropped and why
    rows, cols = 4, 6
    fig, axes = ps.plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.72), dpi=120)
    fig.patch.set_facecolor(ps.SURFACE)
    picks = []
    for d in ["ok", "mismatch", "boilerplate", "tiny", "banner", "dup_near"]:
        idx = [i for i in range(len(defect)) if defect[i] == d][:rows]
        picks.append((d, idx))
    for c, (d, idx) in enumerate(picks):
        for r in range(rows):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(ps.BASELINE)
            if r < len(idx):
                i = idx[r]
                ax.imshow(crawl["images"][i])
                txt = crawl["alt"][i]
                ax.set_xlabel(txt[:26] + ("..." if len(txt) > 26 else ""),
                              fontsize=5.4, color=ps.INK_SECONDARY, labelpad=2)
            if r == 0:
                ax.set_title(d, fontsize=8, color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "crawl_montage.png", facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {OUT / 'crawl_montage.png'}")

    # 5. alt-text vs recaption, side by side on real pictures
    rec = P.load_recaptions()
    if rec:
        # Two examples of each kind: a pair whose alt-text was already fine, one
        # whose caption belongs to a different photo, and one piece of
        # boilerplate. Recaptioning only *helps* on the last two.
        pool = (np.load(DATA / "alive_after_cheap.npy")
                if (DATA / "alive_after_cheap.npy").exists()
                else np.load(DATA / "survivors.npy"))
        pool = [i for i in pool if str(int(i)) in rec]
        ids = []
        for kind in ("ok", "mismatch", "boilerplate"):
            ids += [i for i in pool if crawl["defect"][i] == kind][:2]
        ids = ids[:6] or pool[:6]
        fig, axes = ps.plt.subplots(1, len(ids), figsize=(len(ids) * 1.9, 3.0),
                                    dpi=120)
        fig.patch.set_facecolor(ps.SURFACE)
        for ax, i in zip(axes, ids):
            ax.imshow(crawl["images"][i])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(ps.BASELINE)
            alt = crawl["alt"][i]
            new = rec[str(int(i))]
            ax.set_title(crawl["defect"][i], fontsize=7, color=ps.INK)
            ax.set_xlabel(f"alt: {alt[:34]}\nBLIP: {new[:34]}", fontsize=5.6,
                          color=ps.INK_SECONDARY, labelpad=3, linespacing=1.5)
        fig.tight_layout()
        fig.savefig(OUT / "recaption_examples.png", facecolor=ps.SURFACE,
                    bbox_inches="tight")
        ps.plt.close(fig)
        print(f"wrote {OUT / 'recaption_examples.png'}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["crawl", "filter", "recaption", "shard", "plot"])
    ap.add_argument("--n-base", type=int, default=P.N_BASE)
    ap.add_argument("--max-dist", type=int, default=8)
    ap.add_argument("--hash", default="phash", choices=["dhash", "phash"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-shard", type=int, default=512)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"crawl": stage_crawl, "filter": stage_filter, "recaption": stage_recaption,
     "shard": stage_shard, "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    main()
