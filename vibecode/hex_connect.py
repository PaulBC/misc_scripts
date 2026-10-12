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

What counts as "connected"
--------------------------
A cycle in the base tile shows up in the full hexagon in one of two ways: as
a single loop three times as long (it winds around the hexagon's 3-fold
center) or as THREE separate copies (it doesn't). One base cycle is therefore
not enough - the tool minimizes the number of closed loops in the full
hexagon (1 for a connected cycle, 3 for a cycle that appears as copies), and
uses the number of base cycles as the tiebreak. The goal is exactly one loop.
A refill that turns a three-copy cycle into a connected one counts as
progress even though the number of base cycles doesn't change.

With --wall-boundary the cycle girding the whole pattern (the one running
against the dead wall) is treated separately: it is kept fixed, never merged
with the interior, and the goal becomes ONE INTERIOR cycle. That matches
hex_render.py --remove-outer-wall, which deletes that cycle.

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
                   uphill_prob=0.25, seed=None, time_limit=None, verbose=True):
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

    rot = hw.build_rotation_table(n)

    def state():
        cid, sizes, _ = hw.find_cycles(grid, n, nt)
        return cid, sizes

    wall_ids_of = lambda cid_: {cid_[c] for c in wall_cycle if c in cid_}

    def score(cid_, sizes_):
        """(closed loops in the full hexagon, cycles), interior only - the
        frozen outer wall cycle is excluded. Lower is better, loops first:
        one connected cycle is (1, 1), but a single cycle that the hexagon
        shows as three separate copies is (3, 1)."""
        lifts_ = hw.cycle_lifts(grid, n, cid_, sizes_, nt, rot)
        skip = wall_ids_of(cid_)
        ks = [k for k in range(len(sizes_)) if k not in skip]
        return (sum(1 if lifts_[k] else 3 for k in ks), len(ks)), lifts_

    def out_of_time():
        return bool(time_limit) and time.time() - t0 > time_limit

    cid, sizes = state()

    # With a wall boundary, the cycle girding the whole pattern (the one
    # whose cells sit against the dead wall) can't be merged with the
    # interior - and shouldn't be: hex_render.py --remove-outer-wall deletes
    # it, so merging interior cells into it would delete part of the pattern.
    # Freeze it, and leave it out of the goal.
    wall_cycle = set()
    if wall_boundary:
        touch = {}
        for c, k in cid.items():
            if any(d in wall for d in nt[c[0]][c[1]]):
                touch[k] = touch.get(k, 0) + 1
        if touch:
            wid = max(touch, key=touch.get)
            wall_cycle = {c for c, k in cid.items() if k == wid}
            locked |= wall_cycle

    cur, lifts = score(cid, sizes)
    history = [cur]
    best_grid, best = [row[:] for row in grid], cur
    plateau_used = 0

    def fmt(sc):
        return f"{sc[0]} loop(s) / {sc[1]} cycle(s)"

    if verbose:
        print(f"start: {fmt(cur)} in the full hexagon; base cycles {sizes}")
        if wall_cycle:
            print(f"outer wall cycle: {len(wall_cycle)} cells, kept fixed and not counted")

    while cur[0] > 1 and not out_of_time():
        improved = False
        sideways = []
        uphill = []
        for r in range(radius, max_radius + 1):
            rng.shuffle(centers)
            for ctr in centers:
                if out_of_time():
                    break
                # worth trying only where two cycles could merge, or where a
                # cycle that shows up as 3 separate copies could be turned
                # into one connected loop
                near = {cid[c] for c in cells_within(ctr, r + 1)
                        if c in cid and c not in wall_cycle}
                if not near or (len(near) < 2 and all(lifts[k] for k in near)):
                    continue
                free, packed = alternatives(ctr, r)
                if not packed:
                    continue
                saved = [grid[c[0]][c[1]] for c in free]
                best_sc, best_m, side, up = cur, None, [], []
                for m in packed:
                    apply_mask(free, m)
                    c2, s2, _ = hw.find_cycles(grid, n, nt)
                    sc, _ = score(c2, s2)
                    if sc < best_sc:
                        best_sc, best_m = sc, m
                    elif sc == cur:
                        side.append(m)
                    elif sc[0] <= cur[0] + 1 and len(up) < 40:
                        up.append(m)
                for c, v in zip(free, saved):
                    grid[c[0]][c[1]] = v
                if best_m is not None:
                    apply_mask(free, best_m)
                    cid, sizes = state()
                    new, lifts = score(cid, sizes)
                    if verbose:
                        print(f"  [{time.time() - t0:5.0f}s] radius {r} at {ctr}: "
                              f"{fmt(cur)} -> {fmt(new)}")
                    cur = new
                    history.append(cur)
                    if cur < best:
                        best_grid, best = [row[:] for row in grid], cur
                    improved = True
                    break
                sideways.extend((tuple(free), m) for m in side)
                uphill.extend((tuple(free), m) for m in up)
            if improved or out_of_time():
                break
        if improved:
            continue
        if out_of_time():
            if verbose:
                print("time limit reached")
            break
        if plateau_used >= plateau_steps or not (sideways or uphill):
            if verbose:
                print(f"stuck at {fmt(cur)} (radius up to {max_radius}, "
                      f"{plateau_used} sideways moves used)")
            break
        # Stuck: shuffle sideways, and now and then accept a slightly WORSE
        # state (at most one more loop) - some connected loops can only be
        # reached through a detour, e.g. merging two connected cycles gives a
        # single cycle that first shows as three copies.
        pool = uphill if (uphill and (not sideways or rng.random() < uphill_prob)) else sideways
        free, m = rng.choice(pool)
        apply_mask(free, m)
        cid, sizes = state()
        cur, lifts = score(cid, sizes)
        plateau_used += 1
        history.append(cur)
        if verbose and plateau_used % 20 == 0:
            print(f"  [{time.time() - t0:5.0f}s] {plateau_used} sideways moves, {fmt(cur)}")

    bad = count_violations(best_grid, n, nt, wall)
    assert bad == 0, f"internal error: result has {bad} rule violations"
    assert all(best_grid[a][b] for a, b in wall_cycle), "outer wall cycle was altered"
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
                        "and exempt from the rules). The outer wall cycle is then kept "
                        "fixed and the goal is a single interior cycle")
    p.add_argument("--plateau-steps", type=int, default=200,
                   help="max random sideways moves when stuck (default 200)")
    p.add_argument("--uphill-prob", type=float, default=0.25,
                   help="when stuck, chance of accepting a slightly worse state (at most one more "
                        "loop) instead of a sideways move, to get past dead ends (default 0.25)")
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
                               plateau_steps=args.plateau_steps, uphill_prob=args.uphill_prob,
                               seed=args.seed,
                               time_limit=args.time_limit, verbose=not args.quiet)
    cid_o, sizes, _ = hw.find_cycles(out, n, nt)
    print(hw.describe_cycles(sizes, False))
    lifts_o = hw.cycle_lifts(out, n, cid_o, sizes, nt)
    if args.wall_boundary:
        touch = {}
        for c, k in cid_o.items():
            if any(d in wall for d in nt[c[0]][c[1]]):
                touch[k] = touch.get(k, 0) + 1
        skip = {max(touch, key=touch.get)} if touch else set()
        note = " (outer wall cycle not counted)"
    else:
        skip, note = set(), ""
    ks = [k for k in range(len(sizes)) if k not in skip]
    loops = sum(1 if lifts_o[k] else 3 for k in ks)
    print(f"closed loops in the full hexagon: {loops}{note}")
    if loops == 1:
        print("-> a single connected loop")
    else:
        print("-> not a single connected loop: "
              f"{sum(1 for k in ks if lifts_o[k])} connected cycle(s), "
              f"{sum(1 for k in ks if not lifts_o[k])} showing as 3 separate copies")
    path = unique_path(args.output)
    hw.write_grid_txt(out, n, path)
    print("saved grid to", path)
