"""
Connect the distinct cycles of a solved hex pattern into fewer (ideally one)
by re-filling neighborhoods.

Input is the plain text grid written by hex_walksat.py (first line n, then n
lines of n 0/1 characters). Output is a new grid file in the same format, so
it can go straight into hex_render.py.

How it works
------------
A solved pattern is a set of disjoint cycles (Rule 1). A single flip, or even
a re-fill of a 7-cell neighborhood, almost never merges two of them: the
rules interlock too tightly. But a LARGER neighborhood often can. For a
chosen center and radius r, every cell within hex distance r is freed while
everything outside stays frozen, and ALL assignments of the freed cells under
which every affected cell still satisfies the rules are enumerated (exact
backtracking with lookahead - see enumerate_valid). Each alternative is
applied tentatively and the whole pattern's cycles are recounted; an
alternative that lowers the count is a merge, and the best one is kept.

Because only rule-satisfying alternatives are ever applied, the grid is
valid at every step - there is no repair phase and no violation to chase.

When no region at the starting radius can merge anything, the radius grows up
to --max-radius. If still stuck, random "sideways" moves (a valid re-fill
that leaves the cycle count unchanged) shuffle the structure so new merges
become possible, up to --plateau-steps of them. Enumerations are cached per
region on the exact cells they depend on, so rescanning after a move only
re-does the regions near it. Whatever is stuck
after that is reported honestly - reaching a single cycle is not guaranteed
(it may not even exist for a given grid/size), so the best result seen is
what gets written.

Cycles are found through the periodic boundary (see find_cycles), so a path
crossing a rotated seam counts as one cycle.

Needs hex_walksat.py alongside it for the rules (rule tables, lookahead
table, neighbor table) - the solver is the single authority on what a valid
pattern is. Branching (--allow-branch) is not supported here.
"""

import argparse
import os
import random
import sys
import time

import hex_walksat as hw


def read_grid(path):
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n = int(lines[0])
    grid = [[ch == "1" for ch in lines[1 + i]] for i in range(n)]
    if len(grid) != n or any(len(r) != n for r in grid):
        raise ValueError(f"{path}: expected an {n}x{n} grid")
    return grid, n


def count_violations(grid, n, nt, wall):
    return sum(1 for i in range(n) for j in range(n)
               if (i, j) not in wall and not hw.cell_ok(grid, n, i, j, nt))


def enumerate_valid(grid, nt, center, radius, locked, wall, node_limit, max_sol):
    """Every assignment of the non-locked cells within `radius` of `center`
    (other than the current one) under which every affected cell satisfies
    the rules, everything outside frozen. Exact backtracking: constraints are
    checked as soon as their last free cell is assigned, and lookahead cuts a
    branch the moment some constraint can no longer be satisfied by any
    completion. `grid` is restored before returning.

    Returns (free_cells, solutions, capped); each solution is a list of
    booleans parallel to free_cells. capped is True if the node or solution
    limit cut the enumeration short (so the list may be incomplete)."""
    layers = hw._region_layers(center, nt, radius)
    free = [c for layer in layers for c in layer if c not in locked]
    nfree = len(free)
    fidx = {c: i for i, c in enumerate(free)}

    check = set()
    for c in free:
        check.add(c)
        check.update(nt[c[0]][c[1]])
    check -= wall

    xpos = []
    dependents = [[] for _ in range(nfree)]
    for x in check:
        pos = [(1 << 6, x[0], x[1], fidx.get(x, -1))]
        for k, (a, b) in enumerate(nt[x[0]][x[1]]):
            pos.append((1 << k, a, b, fidx.get((a, b), -1)))
        deps = {fi for (_, _, _, fi) in pos if fi >= 0}
        if not deps:
            continue
        xid = len(xpos)
        xpos.append(pos)
        for fi in deps:
            dependents[fi].append(xid)

    feas = hw.completion_table(False, False)

    def doomed(xid, idx):
        known = pend = 0
        for bit, a, b, fi in xpos[xid]:
            if fi > idx:
                pend |= bit
            elif grid[a][b]:
                known |= bit
        return not feas[(pend << 7) | known]

    orig = [grid[c[0]][c[1]] for c in free]
    sols = []
    st = {"nodes": 0, "capped": False}

    def rec(idx, diff):
        if st["capped"]:
            return
        if idx == nfree:
            if diff > 0:
                sols.append([grid[c[0]][c[1]] for c in free])
                if len(sols) >= max_sol:
                    st["capped"] = True
            return
        st["nodes"] += 1
        if st["nodes"] > node_limit:
            st["capped"] = True
            return
        ci, cj = free[idx]
        for val in (orig[idx], not orig[idx]):
            grid[ci][cj] = val
            if not any(doomed(x, idx) for x in dependents[idx]):
                rec(idx + 1, diff + (val != orig[idx]))

    rec(0, 0)
    for c, v in zip(free, orig):
        grid[c[0]][c[1]] = v
    return free, sols, st["capped"]


def connect_cycles(grid, n, radius=3, max_radius=5, wall_boundary=False,
                   node_limit=30_000, max_sol=2_000, plateau_steps=200,
                   seed=None, time_limit=None, verbose=True):
    """Returns (best_grid, history) where history is the list of cycle counts
    after each applied move (starting with the input's count).

    Each region's enumeration of valid refills depends only on the grid
    within radius+2 of its center (the freed cells, the cells whose rules
    they affect, and those cells' other neighbors), so results are cached on
    exactly that content. After any move, only regions near it are
    re-enumerated; everything else is a cache hit, which makes rescanning
    after a small change nearly free."""
    rng = random.Random(seed)
    nt = hw.build_neighbor_table(n)
    wall = set()
    if wall_boundary:
        wall = {(i, j) for i in range(n) for j in range(n) if i == n - 1 or j == n - 1}
    locked = set(wall)
    grid = [row[:] for row in grid]
    t0 = time.time()
    centers = [(i, j) for i in range(n) for j in range(n)]

    layer_cache = {}

    def cells_within(ctr, rad):
        key = (ctr, rad)
        v = layer_cache.get(key)
        if v is None:
            v = [c for layer in hw._region_layers(ctr, nt, rad) for c in layer]
            layer_cache[key] = v
        return v

    enum_cache = {}

    def alternatives(ctr, r):
        """(free cells, list of alternatives packed as bitmasks over free)."""
        ctx = cells_within(ctr, r + 2)
        key = (ctr, r, tuple(grid[a][b] for a, b in ctx))
        hit = enum_cache.get(key)
        if hit is None:
            free, sols, _ = enumerate_valid(grid, nt, ctr, r, locked, wall,
                                            node_limit, max_sol)
            packed = [sum(1 << k for k, v in enumerate(a) if v) for a in sols]
            hit = (free, packed)
            enum_cache[key] = hit
        return hit

    def apply_mask(free, m):
        for idx, c in enumerate(free):
            grid[c[0]][c[1]] = bool((m >> idx) & 1)

    def state():
        cid, sizes, _ = hw.find_cycles(grid, n, nt)
        return cid, sizes

    def out_of_time():
        return bool(time_limit) and time.time() - t0 > time_limit

    cid, sizes = state()
    K = len(sizes)
    history = [K]
    best_grid, best_K = [row[:] for row in grid], K
    plateau_used = 0
    if verbose:
        print(f"start: {K} cycles {sizes}")

    while K > 1 and not out_of_time():
        merged = False
        sideways = []
        for r in range(radius, max_radius + 1):
            rng.shuffle(centers)
            for ctr in centers:
                if out_of_time():
                    break
                # a merge needs cells of at least two different cycles
                # within reach of the region
                near = {cid[c] for c in cells_within(ctr, r + 1) if c in cid}
                if len(near) < 2:
                    continue
                free, packed = alternatives(ctr, r)
                if not packed:
                    continue
                saved = [grid[c[0]][c[1]] for c in free]
                best_k, best_m, side = K, None, []
                for m in packed:
                    apply_mask(free, m)
                    k = len(hw.find_cycles(grid, n, nt)[1])
                    if k < best_k:
                        best_k, best_m = k, m
                    elif k == K:
                        side.append(m)
                for c, v in zip(free, saved):
                    grid[c[0]][c[1]] = v
                if best_m is not None:
                    apply_mask(free, best_m)
                    cid, sizes = state()
                    if verbose:
                        print(f"  [{time.time() - t0:5.0f}s] merge at {ctr}, radius {r}: "
                              f"{K} -> {len(sizes)} cycles")
                    K = len(sizes)
                    history.append(K)
                    if K < best_K:
                        best_grid, best_K = [row[:] for row in grid], K
                    merged = True
                    break
                sideways.extend((tuple(free), m) for m in side)
            if merged or out_of_time():
                break
        if merged:
            continue
        if out_of_time():
            if verbose:
                print("time limit reached")
            break
        if plateau_used >= plateau_steps or not sideways:
            if verbose:
                print(f"stuck at {K} cycles (radius up to {max_radius}, "
                      f"{plateau_used} sideways moves used)")
            break
        free, m = rng.choice(sideways)
        apply_mask(free, m)
        cid, sizes = state()
        K = len(sizes)
        plateau_used += 1
        history.append(K)
        if verbose and plateau_used % 20 == 0:
            print(f"  [{time.time() - t0:5.0f}s] {plateau_used} sideways moves, {K} cycles")

    bad = count_violations(best_grid, n, nt, wall)
    assert bad == 0, f"internal error: result has {bad} rule violations"
    return best_grid, history


def unique_path(path):
    base, ext = os.path.splitext(path)
    k = 1
    while os.path.exists(f"{base}_{k}{ext}"):
        k += 1
    return f"{base}_{k}{ext}"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", help="grid text file written by hex_walksat.py")
    p.add_argument("--radius", type=int, default=3,
                   help="start by re-filling neighborhoods of this hex radius (default 3; "
                        "radius 1 = 7 cells almost never merges anything)")
    p.add_argument("--max-radius", type=int, default=5,
                   help="grow the radius up to this if nothing merges (default 5)")
    p.add_argument("--wall-boundary", action="store_true",
                   help="the grid was solved with --wall-boundary (edge cells are hard-dead "
                        "and exempt from the rules)")
    p.add_argument("--plateau-steps", type=int, default=200,
                   help="max random sideways moves when stuck (default 200)")
    p.add_argument("--node-limit", type=int, default=30_000,
                   help="search-node cap per region enumeration (default 30000)")
    p.add_argument("--time-limit", type=float, default=None, help="stop after this many seconds")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-o", "--output", default="hex_grid_connected.txt")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    g, n = read_grid(args.input)
    nt = hw.build_neighbor_table(n)
    wall = ({(i, j) for i in range(n) for j in range(n) if i == n - 1 or j == n - 1}
            if args.wall_boundary else set())
    if count_violations(g, n, nt, wall):
        sys.exit("input grid already violates the rules - pass --wall-boundary if it was "
                 "solved with that option, and note --allow-branch grids aren't supported")
    out, hist = connect_cycles(g, n, radius=args.radius, max_radius=args.max_radius,
                               wall_boundary=args.wall_boundary,
                               node_limit=args.node_limit,
                               plateau_steps=args.plateau_steps, seed=args.seed,
                               time_limit=args.time_limit, verbose=not args.quiet)
    _, sizes, _ = hw.find_cycles(out, n, nt)
    print(hw.describe_cycles(sizes, False))
    path = unique_path(args.output)
    hw.write_grid_txt(out, n, path)
    print("saved grid to", path)
