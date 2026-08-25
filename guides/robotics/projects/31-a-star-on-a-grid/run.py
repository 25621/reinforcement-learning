"""Project 31 -- A* on a grid: what a heuristic buys, and what it costs.

Seven experiments:

  1. A* against Dijkstra on one map: same answer, a fraction of the work
  2. the heuristic ladder: zero -> Manhattan -> Euclidean -> octile
  3. the inadmissible heuristic: Manhattan on an 8-connected grid
  4. weighted A*: paying a bounded amount of optimality for speed
  5. tie-breaking, the free speed-up nobody mentions
  6. where the heuristic stops helping: a maze
  7. what the grid itself costs you: digitisation bias and any-angle paths

Runs in about three minutes.  NumPy and Matplotlib only.
"""

import csv
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from grid import (blob_map, maze_map, random_query, search, path_length,   # noqa: E402
                  line_of_sight, shortcut_los, free_cells)
from plot_style import COLORS, use_style, save                             # noqa: E402

import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.colors import ListedColormap                               # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def draw_search(ax, grid, res, start, goal, title):
    """Grey = wall, pale blue = cells the search settled, line = the path."""
    img = np.zeros(grid.shape)
    img[res["closed"]] = 1.0
    img[grid] = 2.0
    ax.imshow(img, cmap=ListedColormap(["white", "#BFDCEF", "#4A4A4A"]),
              origin="lower", interpolation="nearest")
    if res["path"]:
        p = np.asarray(res["path"])
        ax.plot(p[:, 1], p[:, 0], color=COLORS[1], lw=1.6)
    ax.plot([start[1]], [start[0]], "o", color=COLORS[2], ms=5)
    ax.plot([goal[1]], [goal[0]], "*", color=COLORS[3], ms=10)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


# =====================================================================  1
def exp1_astar_vs_dijkstra(rng):
    banner("1. A* against Dijkstra: identical path, a fraction of the work")

    grid = blob_map(160, 160, np.random.default_rng(24), n_blobs=34)
    start, goal = (8, 8), (151, 151)
    grid[start] = grid[goal] = False

    dij = search(grid, start, goal, heuristic="zero")
    ast = search(grid, start, goal, heuristic="octile")
    grd = search(grid, start, goal, heuristic="octile", greedy=True)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, res, nm in zip(axes, (dij, ast, grd),
                           ("Dijkstra (h = 0)", "A* (octile h)",
                            "greedy best-first (g ignored)")):
        draw_search(ax, grid, res, start, goal,
                    f"{nm}\n{res['expanded']} expanded, cost {res['cost']:.1f}")
    save(fig, os.path.join(OUT, "astar_vs_dijkstra.png"))

    free = int((~grid).sum())
    for nm, r in (("dijkstra", dij), ("astar", ast), ("greedy", grd)):
        frac = 100.0 * r["expanded"] / free
        print(f"  {nm:<10s} expanded {r['expanded']:6d} ({frac:5.1f}% of free "
              f"cells)  cost {r['cost']:8.3f}  {r['time']*1e3:6.1f} ms")
        record(1, nm, expanded=r["expanded"], pct_of_free=round(frac, 2),
               cost=round(r["cost"], 4), ms=round(r["time"] * 1e3, 2))
    print(f"  free cells on the map: {free}")
    print(f"  A* expands {dij['expanded']/ast['expanded']:.2f}x fewer cells "
          f"than Dijkstra for the SAME cost "
          f"(delta {abs(dij['cost']-ast['cost']):.2e})")
    print(f"  greedy is {ast['expanded']/grd['expanded']:.2f}x cheaper still "
          f"but its path is {100*(grd['cost']/dij['cost']-1):.1f}% too long")
    record(1, "speedup_astar_over_dijkstra",
           ratio=round(dij["expanded"] / ast["expanded"], 3))
    record(1, "greedy_suboptimality_pct",
           pct=round(100 * (grd["cost"] / dij["cost"] - 1), 3))

    # One map proves nothing about greedy.  Repeat over many random queries.
    r2 = np.random.default_rng(101)
    gaps, sp = [], []
    for _ in range(12):
        gm = blob_map(140, 140, r2, n_blobs=30)
        for _ in range(4):
            s2, g2 = random_query(gm, r2)
            opt = search(gm, s2, g2, heuristic="octile")
            gy = search(gm, s2, g2, heuristic="octile", greedy=True)
            if not (opt["found"] and gy["found"]):
                continue
            gaps.append(100 * (gy["cost"] / opt["cost"] - 1))
            sp.append(opt["expanded"] / gy["expanded"])
    gaps = np.array(gaps)
    print(f"  over {len(gaps)} random queries, greedy expands "
          f"{np.mean(sp):.1f}x fewer than A* and its path is "
          f"{gaps.mean():.1f}% too long on average, {gaps.max():.1f}% worst")
    record(1, "greedy_over_many_queries", queries=len(gaps),
           mean_excess_pct=round(float(gaps.mean()), 3),
           worst_excess_pct=round(float(gaps.max()), 3),
           mean_speedup_vs_astar=round(float(np.mean(sp)), 2))


# =====================================================================  2
def exp2_heuristic_ladder(rng):
    banner("2. The heuristic ladder, on 4-connected and 8-connected grids")

    n_maps, n_q = 12, 6
    rows = []
    for conn in (4, 8):
        for hname in ("zero", "manhattan", "euclidean", "octile", "chebyshev"):
            exp, cost, opt_gap, nfail = [], [], [], 0
            r2 = np.random.default_rng(11)
            for m in range(n_maps):
                grid = blob_map(120, 120, r2, n_blobs=26)
                for _ in range(n_q):
                    s, g = random_query(grid, r2)
                    ref = search(grid, s, g, heuristic="zero", conn=conn)
                    if not ref["found"]:
                        continue
                    res = search(grid, s, g, heuristic=hname, conn=conn)
                    if not res["found"]:
                        nfail += 1
                        continue
                    exp.append(res["expanded"])
                    cost.append(res["cost"])
                    opt_gap.append(res["cost"] / ref["cost"] - 1.0)
            rows.append(dict(conn=conn, h=hname, expanded=np.mean(exp),
                             gap=100 * np.mean(opt_gap),
                             worst=100 * np.max(opt_gap)))
    print(f"  {'conn':>4s} {'heuristic':<11s} {'mean expanded':>13s} "
          f"{'mean excess %':>13s} {'worst excess %':>14s}  admissible?")
    for r in rows:
        adm = "yes" if r["worst"] < 1e-9 else "NO"
        print(f"  {r['conn']:>4d} {r['h']:<11s} {r['expanded']:13.0f} "
              f"{r['gap']:13.3f} {r['worst']:14.3f}  {adm}")
        record(2, f"conn{r['conn']}_{r['h']}", expanded=round(r["expanded"], 1),
               mean_excess_pct=round(r["gap"], 4),
               worst_excess_pct=round(r["worst"], 4), admissible=adm)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = ["zero", "manhattan", "euclidean", "octile", "chebyshev"]
    w = 0.38
    for k, conn in enumerate((4, 8)):
        vals = [r["expanded"] for r in rows if r["conn"] == conn]
        ax.bar(np.arange(len(names)) + (k - 0.5) * w, vals, w,
               label=f"{conn}-connected", color=COLORS[k])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("mean cells expanded")
    ax.set_title("Better estimate, less search (bars that break the rules are\n"
                 "cheap because they are cheating -- see experiment 3)")
    ax.legend()
    save(fig, os.path.join(OUT, "heuristic_ladder.png"))


# =====================================================================  3
def exp3_inadmissible(rng):
    banner("3. Manhattan on an 8-connected grid: fast, and wrong")

    r2 = np.random.default_rng(21)
    gaps, wins, n = [], 0, 0
    for m in range(20):
        grid = blob_map(120, 120, r2, n_blobs=24)
        for _ in range(6):
            s, g = random_query(grid, r2)
            ref = search(grid, s, g, heuristic="octile", conn=8)
            bad = search(grid, s, g, heuristic="manhattan", conn=8)
            if not (ref["found"] and bad["found"]):
                continue
            n += 1
            gap = bad["cost"] / ref["cost"] - 1.0
            gaps.append(100 * gap)
            if gap > 1e-9:
                wins += 1
    gaps = np.array(gaps)
    print(f"  {n} queries")
    print(f"  suboptimal on {wins}/{n} = {100*wins/n:.0f}% of them")
    print(f"  mean excess {gaps.mean():.2f}%   worst {gaps.max():.2f}%")
    print(f"  the bound says: h_manhattan <= sqrt(2) * h_true, so the path can"
          f" be up to 41% too long")
    record(3, "manhattan_on_8conn", queries=n, pct_suboptimal=round(100 * wins / n, 1),
           mean_excess_pct=round(float(gaps.mean()), 3),
           worst_excess_pct=round(float(gaps.max()), 3))

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.hist(gaps, bins=30, color=COLORS[1])
    ax.set_xlabel("path cost above optimal (%)")
    ax.set_ylabel("queries")
    ax.set_title("Manhattan heuristic on an 8-connected grid\n"
                 "(it overestimates, so A* stops looking too early)")
    save(fig, os.path.join(OUT, "inadmissible.png"))


# =====================================================================  4
def exp4_weighted(rng):
    banner("4. Weighted A*: a dial from optimal-and-slow to fast-and-nearly")

    epss = [1.0, 1.05, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0]
    r2 = np.random.default_rng(31)
    maps = []
    for _ in range(10):
        grid = blob_map(140, 140, r2, n_blobs=30)
        s, g = random_query(grid, r2)
        maps.append((grid, s, g))

    print(f"  {'eps':>6s} {'mean expanded':>13s} {'speed-up':>9s} "
          f"{'mean excess %':>13s} {'worst %':>8s} {'bound %':>8s}")
    curve = []
    for eps in epss:
        exp, gap = [], []
        for grid, s, g in maps:
            ref = search(grid, s, g, heuristic="octile")
            res = search(grid, s, g, heuristic="octile", weight=eps)
            if not (ref["found"] and res["found"]):
                continue
            exp.append(res["expanded"])
            gap.append(100 * (res["cost"] / ref["cost"] - 1))
        curve.append((eps, np.mean(exp), np.mean(gap), np.max(gap)))
    base = curve[0][1]
    for eps, e, gmean, gmax in curve:
        print(f"  {eps:6.2f} {e:13.0f} {base/e:8.2f}x {gmean:13.2f} "
              f"{gmax:8.2f} {100*(eps-1):8.0f}")
        record(4, f"eps_{eps}", expanded=round(e, 1), speedup=round(base / e, 3),
               mean_excess_pct=round(gmean, 3), worst_excess_pct=round(gmax, 3),
               bound_pct=round(100 * (eps - 1), 1))

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ee = [c[1] for c in curve]
    gg = [c[2] for c in curve]
    ax.plot(ee, gg, "o-", color=COLORS[0])
    for (eps, e, gmean, _) in curve:
        ax.annotate(f"{eps:g}", (e, gmean), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    ax.set_xlabel("mean cells expanded")
    ax.set_ylabel("path cost above optimal (%)")
    ax.set_xscale("log")
    ax.set_title("Weighted A*: the label on each point is eps\n"
                 "(the real cost of speed is far below the eps bound)")
    save(fig, os.path.join(OUT, "weighted_astar.png"))


# =====================================================================  5
def exp5_tie_breaking(rng):
    banner("5. Tie-breaking: the free speed-up")

    r2 = np.random.default_rng(41)
    for label, mk in (("open field (no obstacles)",
                       lambda: np.zeros((120, 120), dtype=bool)),
                      ("blob map", lambda: blob_map(120, 120, r2, n_blobs=26))):
        none_e, low_e, cost_ok = [], [], True
        for _ in range(8):
            grid = mk()
            grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
            s, g = random_query(grid, r2)
            a = search(grid, s, g, heuristic="octile", tie_break="none")
            b = search(grid, s, g, heuristic="octile", tie_break="low_h")
            if not (a["found"] and b["found"]):
                continue
            none_e.append(a["expanded"])
            low_e.append(b["expanded"])
            cost_ok &= abs(a["cost"] - b["cost"]) < 1e-9
        print(f"  {label:<26s} FIFO ties {np.mean(none_e):8.0f} -> "
              f"prefer-low-h {np.mean(low_e):8.0f}  "
              f"({np.mean(none_e)/np.mean(low_e):.2f}x)  "
              f"same cost: {cost_ok}")
        record(5, label.split(" ")[0], fifo=round(float(np.mean(none_e)), 1),
               low_h=round(float(np.mean(low_e)), 1),
               speedup=round(float(np.mean(none_e) / np.mean(low_e)), 3),
               same_cost=cost_ok)

    grid = np.zeros((100, 100), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    s, g = (10, 10), (89, 89)
    a = search(grid, s, g, heuristic="octile", tie_break="none")
    b = search(grid, s, g, heuristic="octile", tie_break="low_h")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    draw_search(axes[0], grid, a, s, g,
                f"ties broken FIFO\n{a['expanded']} expanded")
    draw_search(axes[1], grid, b, s, g,
                f"ties broken toward the goal\n{b['expanded']} expanded")
    save(fig, os.path.join(OUT, "tie_breaking.png"))


# =====================================================================  6
def exp6_maze(rng):
    banner("6. A maze: where the heuristic stops helping")

    r2 = np.random.default_rng(51)
    grid = maze_map(101, 101, r2)
    cells = free_cells(grid)
    s, g = (1, 1), (99, 99)
    dij = search(grid, s, g, heuristic="zero", conn=4)
    ast = search(grid, s, g, heuristic="manhattan", conn=4)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    draw_search(axes[0], grid, dij, s, g,
                f"Dijkstra\n{dij['expanded']} expanded")
    draw_search(axes[1], grid, ast, s, g,
                f"A* (Manhattan)\n{ast['expanded']} expanded")
    save(fig, os.path.join(OUT, "maze.png"))

    print(f"  maze: Dijkstra {dij['expanded']}, A* {ast['expanded']}  "
          f"-> only {dij['expanded']/ast['expanded']:.2f}x")
    record(6, "maze_101", dijkstra=dij["expanded"], astar=ast["expanded"],
           speedup=round(dij["expanded"] / ast["expanded"], 3),
           free_cells=len(cells))

    # The same comparison on an open map, at the same size, for contrast.
    grid2 = blob_map(101, 101, np.random.default_rng(52), n_blobs=22)
    s2, g2 = (2, 2), (98, 98)
    grid2[s2] = grid2[g2] = False
    d2 = search(grid2, s2, g2, heuristic="zero", conn=4)
    a2 = search(grid2, s2, g2, heuristic="manhattan", conn=4)
    print(f"  open: Dijkstra {d2['expanded']}, A* {a2['expanded']}  "
          f"-> {d2['expanded']/a2['expanded']:.2f}x")
    record(6, "open_101", dijkstra=d2["expanded"], astar=a2["expanded"],
           speedup=round(d2["expanded"] / a2["expanded"], 3))

    # Why: measure how wrong the heuristic is, averaged over the map.
    for nm, gr, goal in (("maze", grid, g), ("open", grid2, g2)):
        ref = search(gr, goal, goal, heuristic="zero", conn=4)  # warm the code
        true = {}
        # single Dijkstra from the goal gives the true cost-to-go everywhere
        import heapq as hq
        dist = np.full(gr.shape, np.inf)
        dist[goal] = 0.0
        pq = [(0.0, goal)]
        while pq:
            d, u = hq.heappop(pq)
            if d > dist[u]:
                continue
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v = (u[0] + dy, u[1] + dx)
                if not (0 <= v[0] < gr.shape[0] and 0 <= v[1] < gr.shape[1]):
                    continue
                if gr[v] or dist[v] <= d + 1:
                    continue
                dist[v] = d + 1
                hq.heappush(pq, (d + 1, v))
        ys, xs = np.nonzero(np.isfinite(dist) & ~gr)
        htrue = dist[ys, xs]
        hest = np.abs(ys - goal[0]) + np.abs(xs - goal[1])
        ratio = np.mean(hest[htrue > 0] / htrue[htrue > 0])
        print(f"  {nm}: heuristic captures {100*ratio:.1f}% of the true "
              f"distance on average")
        record(6, f"{nm}_h_informedness_pct", pct=round(100 * float(ratio), 2))


# =====================================================================  7
def exp7_grid_tax(rng):
    banner("7. What the grid itself costs: digitisation bias")

    r2 = np.random.default_rng(61)
    # (a) empty map: the shortest 8-connected path vs the straight line
    grid = np.zeros((200, 200), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    ratios = []
    for _ in range(60):
        s, g = random_query(grid, r2)
        res = search(grid, s, g, heuristic="octile")
        straight = math.dist(s, g)
        ratios.append(res["cost"] / straight)
    ratios = np.array(ratios)
    print(f"  empty map, 8-connected: grid path is {100*(ratios.mean()-1):.2f}% "
          f"longer than the straight line on average, worst "
          f"{100*(ratios.max()-1):.2f}%")
    worst_theory = 100 * (math.hypot(math.sqrt(2) - 1, 1) - 1)
    print(f"  theory: worst case sqrt((sqrt2-1)^2 + 1) - 1 = "
          f"{worst_theory:.2f}%, hit at 22.5 degrees off an axis")
    record(7, "empty_8conn_excess_pct", mean=round(100 * (ratios.mean() - 1), 3),
           worst=round(100 * (ratios.max() - 1), 3))

    # (b) with obstacles: how much of the excess an any-angle post-process wins
    grid = blob_map(160, 160, r2, n_blobs=28)
    gains, before, after = [], [], []
    for _ in range(25):
        s, g = random_query(grid, r2)
        res = search(grid, s, g, heuristic="octile")
        if not res["found"]:
            continue
        sm = shortcut_los(grid, res["path"], r2, iters=400)
        before.append(path_length(res["path"]))
        after.append(path_length(sm))
        gains.append(100 * (1 - after[-1] / before[-1]))
    print(f"  blob map: any-angle shortcutting removes "
          f"{np.mean(gains):.2f}% of the length (worst case "
          f"{np.max(gains):.2f}%)")
    record(7, "anyangle_gain_pct", mean=round(float(np.mean(gains)), 3),
           max=round(float(np.max(gains)), 3))

    # (c) resolution sweep: expansions and time versus grid size
    print(f"  {'size':>6s} {'free cells':>11s} {'expanded':>9s} {'ms':>8s}")
    sizes, times, exps = [], [], []
    for n in (60, 100, 160, 240, 340):
        gr = blob_map(n, n, np.random.default_rng(7), n_blobs=int(26 * (n / 120) ** 2))
        s, g = (2, 2), (n - 3, n - 3)
        gr[s] = gr[g] = False
        res = search(gr, s, g, heuristic="octile")
        sizes.append(n)
        times.append(res["time"] * 1e3)
        exps.append(res["expanded"])
        print(f"  {n:>6d} {int((~gr).sum()):>11d} {res['expanded']:>9d} "
              f"{res['time']*1e3:>8.1f}")
        record(7, f"res_{n}", free=int((~gr).sum()), expanded=res["expanded"],
               ms=round(res["time"] * 1e3, 2))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    axes[0].hist(100 * (ratios - 1), bins=24, color=COLORS[0])
    axes[0].axvline(worst_theory, color=COLORS[1], ls="--",
                    label=f"theory worst case {worst_theory:.2f}%")
    axes[0].set_xlabel("8-connected path length above straight line (%)")
    axes[0].set_ylabel("queries")
    axes[0].set_title("The grid tax on an empty map")
    axes[0].legend()
    axes[1].plot(sizes, times, "o-", color=COLORS[0], label="time (ms)")
    ax2 = axes[1].twinx()
    ax2.plot(sizes, exps, "s--", color=COLORS[1], label="expanded")
    ax2.grid(False)
    axes[1].set_xlabel("grid side length")
    axes[1].set_ylabel("time (ms)")
    ax2.set_ylabel("cells expanded")
    axes[1].set_title("Cost grows with AREA, not with distance")
    save(fig, os.path.join(OUT, "grid_tax.png"))


def main():
    use_style()
    rng = np.random.default_rng(0)
    exp1_astar_vs_dijkstra(rng)
    exp2_heuristic_ladder(rng)
    exp3_inadmissible(rng)
    exp4_weighted(rng)
    exp5_tie_breaking(rng)
    exp6_maze(rng)
    exp7_grid_tax(rng)

    keys = []
    for r in RESULTS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(RESULTS)
    print(f"\nwrote {os.path.join(OUT, 'results.csv')}  ({len(RESULTS)} rows)")


if __name__ == "__main__":
    main()
