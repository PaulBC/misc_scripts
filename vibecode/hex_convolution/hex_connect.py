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

How it searches
---------------
Phase 1 merges: while there is more than one cycle, it looks for the region
whose refill gives the fewest cycles. Phase 2 starts only if that leaves ONE
cycle shown as three separate copies. A single cycle can only become
connected by moving its curve across the hexagon's 3-fold center (the
origin corner of the tile), so phase 2 tries only regions that contain it -
few enough that large radii are cheap - shuffling sideways around the
center until a refill turns the cycle into one connected loop. (Seen from
the other side: two cycles that both wind around the center merge into one
that doesn't, so a move across the center is what changes the parity.)

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
import random
import sys
import time
from types import SimpleNamespace

import hex_walksat as hw
from hex_common import (UsageError, args_for_command, canonical_argv, grid_digest, make_record,
                        namespace_from, read_meta, unique_path, write_grid_txt)


def read_grid(path):
    with open(path) as f:
        lines = [ln.strip() for ln in f
                 if ln.strip() and not ln.lstrip().startswith("#")]
    n = int(lines[0])
    grid = [[ch == "1" for ch in lines[1 + i]] for i in range(n)]
    if len(grid) != n or any(len(r) != n for r in grid):
        raise ValueError(f"{path}: expected an {n}x{n} grid")
    return grid, n


def count_violations(grid, n, nt, wall):
    return sum(1 for i in range(n) for j in range(n)
               if (i, j) not in wall and not hw.cell_ok(grid, n, i, j, nt))


def enumerate_valid(grid, nt, center, radius, locked, wall, node_limit, max_sol,
                    inward=True):
    """Every assignment of the non-locked cells within `radius` of `center`
    (other than the current one) under which every affected cell satisfies
    the rules, everything outside frozen. Exact backtracking: constraints are
    checked as soon as their last free cell is assigned, and lookahead cuts a
    branch the moment some constraint can no longer be satisfied by any
    completion. Cells are assigned outermost ring first (inward=True; set
    False for the old center-out order). `grid` is restored before returning.

    Returns (free_cells, solutions, capped); each solution is a list of
    booleans parallel to free_cells. capped is True if the node or solution
    limit cut the enumeration short (so the list may be incomplete)."""
    layers = hw._region_layers(center, nt, radius)
    free = [c for layer in layers for c in layer if c not in locked]
    if inward:
        # outermost ring first: those cells border frozen cells, so they are
        # the most constrained and prune the search hardest; the center,
        # least constrained, is decided last
        free.reverse()
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
                   uphill_prob=0.25, center_max_radius=6, center_node_limit=100_000,
                   center_max_sol=4_000, seed=None, time_limit=None, verbose=True):
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

    def alternatives(ctr, r, nl, ms):
        """(free cells, list of alternatives packed as bitmasks over free)."""
        ctx = cells_within(ctr, r + 2)
        key = (ctr, r, nl, ms, tuple(grid[a][b] for a, b in ctx))
        hit = enum_cache.get(key)
        if hit is None:
            free, sols, _ = enumerate_valid(grid, nt, ctr, r, locked, wall, nl, ms)
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

    ORIGIN = (0, 0)     # the hexagon's center: where the three rhombi meet

    while cur[0] > 1 and not out_of_time():
        # Phase 1 (more than one cycle): merge. Fewest cycles first, then
        #   fewest loops, anywhere on the tile.
        # Phase 2 (exactly one cycle, but shown as 3 separate copies): the
        #   only way to make a single cycle connected is to move its curve
        #   across the hexagon's center, so try ONLY regions that contain
        #   the center - there are few of them, so large radii are affordable.
        phase = 1 if cur[1] > 1 else 2
        if phase == 1:
            radii = range(radius, max_radius + 1)
            better = lambda sc: (sc[1], sc[0]) < (cur[1], cur[0])
            rank = lambda sc: (sc[1], sc[0])
        else:
            radii = range(radius, center_max_radius + 1)
            better = lambda sc: sc[1] == cur[1] and sc[0] < cur[0]
            rank = lambda sc: sc[0]
        improved = False
        sideways = []
        uphill = []
        for r in radii:
            if phase == 1:
                cand = list(centers)
            else:
                cand = sorted(set(cells_within(ORIGIN, r)))
            rng.shuffle(cand)
            for ctr in cand:
                if out_of_time():
                    break
                if phase == 1:
                    # a merge needs cells of two different cycles in reach
                    near = {cid[c] for c in cells_within(ctr, r + 1)
                            if c in cid and c not in wall_cycle}
                    if len(near) < 2:
                        continue
                if phase == 1:
                    free, packed = alternatives(ctr, r, node_limit, max_sol)
                else:
                    free, packed = alternatives(ctr, r, center_node_limit, center_max_sol)
                if not packed:
                    continue
                saved = [grid[c[0]][c[1]] for c in free]
                best_sc, best_m, side, up = None, None, [], []
                for m in packed:
                    apply_mask(free, m)
                    c2, s2, _ = hw.find_cycles(grid, n, nt)
                    sc, _ = score(c2, s2)
                    if better(sc):
                        if best_sc is None or rank(sc) < rank(best_sc):
                            best_sc, best_m = sc, m
                    elif sc == cur:
                        side.append(m)
                    elif (phase == 1 and sc[1] <= cur[1] + 1 and len(up) < 40
                          and rank(sc) > rank(cur)):
                        up.append(m)
                for c, v in zip(free, saved):
                    grid[c[0]][c[1]] = v
                if best_m is not None:
                    apply_mask(free, best_m)
                    cid, sizes = state()
                    new, lifts = score(cid, sizes)
                    if verbose:
                        print(f"  [{time.time() - t0:5.0f}s] phase {phase}, radius {r} at "
                              f"{ctr}: {fmt(cur)} -> {fmt(new)}")
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
                print(f"stuck at {fmt(cur)} in phase {phase} "
                      f"({plateau_used} sideways moves used)")
            break
        # Stuck: shuffle sideways (in phase 2, around the center), and in
        # phase 1 now and then accept a slightly worse state - some merges
        # need a detour.
        pool = uphill if (uphill and (not sideways or rng.random() < uphill_prob)) else sideways
        free, m = rng.choice(pool)
        apply_mask(free, m)
        cid, sizes = state()
        cur, lifts = score(cid, sizes)
        plateau_used += 1
        history.append(cur)
        if verbose and plateau_used % 20 == 0:
            print(f"  [{time.time() - t0:5.0f}s] {plateau_used} sideways moves, "
                  f"phase {phase}, {fmt(cur)}")

    bad = count_violations(best_grid, n, nt, wall)
    assert bad == 0, f"internal error: result has {bad} rule violations"
    assert all(best_grid[a][b] for a, b in wall_cycle), "outer wall cycle was altered"
    return best_grid, history


def build_parser():
    """The command-line options. Their names (with - turned into _) are also
    the keyword options of connect_file()."""
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
    p.add_argument("--center-max-radius", type=int, default=6,
                   help="phase 2 (one cycle shown as 3 copies): only regions containing the "
                        "hexagon's center are tried, up to this radius (default 6; few regions, "
                        "so large radii are affordable)")
    p.add_argument("--center-node-limit", type=int, default=100_000,
                   help="phase 2: search-node cap per region (default 100000; far fewer regions "
                        "than phase 1, so a bigger cap is affordable)")
    p.add_argument("--center-max-sol", type=int, default=4_000,
                   help="phase 2: max refills examined per region (default 4000)")
    p.add_argument("--uphill-prob", type=float, default=0.25,
                   help="when stuck, chance of accepting a slightly worse state (at most one more "
                        "loop) instead of a sideways move, to get past dead ends (default 0.25)")
    p.add_argument("--node-limit", type=int, default=30_000,
                   help="search-node cap per region enumeration (default 30000)")
    p.add_argument("--time-limit", type=float, default=None, help="stop after this many seconds")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-o", "--output", default="hex_grid_connected.txt")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def run(args, argv=None, echo=print):
    """Run the connector for parsed arguments exactly as the command line
    does and save the new grid file (carrying the input's provenance chain
    plus this run's record). Returns a namespace with grid, n, grid_path,
    record, loops, single_connected_loop and history. Raises UsageError if
    the input grid isn't a valid pattern for the given options."""
    if argv is None:
        argv = list(sys.argv)

    g, n = read_grid(args.input)
    upstream = read_meta(args.input)
    in_digest = grid_digest(g, n)
    matches = None if not upstream else upstream[-1].get("grid_sha256") == in_digest
    if matches is False:
        echo("note: the input grid differs from the one its metadata describes "
              "(it has been edited since it was written); the history is kept but flagged")
    seed_generated = args.seed is None
    if seed_generated:
        args.seed = random.SystemRandom().randrange(2 ** 32)
        echo("seed:", args.seed, "(generated; pass --seed to repeat this run)")
    nt = hw.build_neighbor_table(n)
    wall = ({(i, j) for i in range(n) for j in range(n) if i == n - 1 or j == n - 1}
            if args.wall_boundary else set())
    if count_violations(g, n, nt, wall):
        raise UsageError("input grid already violates the rules - pass --wall-boundary if it was "
                         "solved with that option, and note --allow-branch grids aren't supported")
    t_conn = time.time()
    out, hist = connect_cycles(g, n, radius=args.radius, max_radius=args.max_radius,
                               wall_boundary=args.wall_boundary,
                               node_limit=args.node_limit,
                               plateau_steps=args.plateau_steps, uphill_prob=args.uphill_prob,
                               center_max_radius=args.center_max_radius,
                               center_node_limit=args.center_node_limit,
                               center_max_sol=args.center_max_sol,
                               seed=args.seed,
                               time_limit=args.time_limit, verbose=not args.quiet)
    cid_o, sizes, _ = hw.find_cycles(out, n, nt)
    echo(hw.describe_cycles(sizes, False))
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
    echo(f"closed loops in the full hexagon: {loops}{note}")
    if loops == 1:
        echo("-> a single connected loop")
    else:
        echo("-> not a single connected loop: "
              f"{sum(1 for k in ks if lifts_o[k])} connected cycle(s), "
              f"{sum(1 for k in ks if not lifts_o[k])} showing as 3 separate copies")
    path = unique_path(args.output)
    record = make_record(
        "hex_connect.py", args, argv, seed_generated=seed_generated,
        result={"loops_before": hist[0][0], "cycles_before": hist[0][1],
                "loops_after": loops, "single_connected_loop": loops == 1,
                "outer_wall_cycle_excluded": bool(args.wall_boundary),
                "elapsed_s": round(time.time() - t_conn, 1)},
        extra={"input_file": args.input, "input_grid_sha256": in_digest,
               "input_matches_upstream": matches,
               "grid_sha256": grid_digest(out, n), "output_grid": path})
    write_grid_txt(out, n, path, meta=upstream + [record])
    echo("saved grid to", path)
    return SimpleNamespace(grid=out, n=n, grid_path=path, record=record, loops=loops,
                           single_connected_loop=loops == 1, history=hist)


def connect_file(input, echo=print, **options):
    """Run the connector from Python, e.g.

        r = connect_file("hex_grid_87.txt", wall_boundary=True)
        r.grid_path          # the connected grid file, ready for hex_render

    Keyword options are the command-line options with - written as _ ; any
    not given take the command line's defaults."""
    parser = build_parser()
    argv = canonical_argv(parser, namespace_from(parser, dict(options, input=input)),
                          "hex_connect.py")
    return run(args_for_command(parser, argv), argv=argv, echo=echo)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args, argv=(list(sys.argv) if argv is None else ["hex_connect.py"] + list(argv)))
    except UsageError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
