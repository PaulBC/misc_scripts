"""
WalkSAT-style local search for the hex non-branching-path pattern.

No SAT solver, no CNF encoding. Direct greedy-plus-noise local search,
same family as Paul Callahan's Stabilizer Algorithm / WalkSAT
(Selman/Kautz/Cohen): pick a violated cell, look at flipping it or one of
its neighbors, take whichever flip reduces total violations the most,
occasionally flip something at random instead to escape local minima.

Boundary: rotated-rhombus periodic identification (equivalence()), which
gives 3-fold rotational symmetry at the seams and is required because the
two rules are unsatisfiable on an open/unwrapped grid.
"""

import colorsys
import math
import os
import random
import time

from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Periodic boundary: rotated rhombus identification
# ---------------------------------------------------------------------------

def equivalence(i, j, n):
    """Map any integer (i, j) - including points that wandered off the base
    n x n tile - back onto the base tile via the 120-degree rotation
    identification (3 copies of the rhombus meet, rotated, at each seam)."""
    while True:
        if ((i // n) + (j // n)) % 3 == 0:
            return i % n, j % n
        i, j = -j - 1, i - j


# Ring-neighbor offsets in cyclic order. Derived from, and verified
# consistent with, the linear part of equivalence()'s rotation map
# R(di, dj) = (-dj, di - dj): R(ring[k]) == ring[(k + 2) % 6] for all k,
# i.e. a 120-degree turn is exactly two steps around this ring. See
# derive_ring.py for the check.
RING = [(1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1)]


def neighbors(i, j, n):
    """The 6 ring neighbors of (i, j), in cyclic order, each folded back
    onto the base tile."""
    return [equivalence(i + di, j + dj, n) for di, dj in RING]


def build_neighbor_table(n):
    """Precompute neighbors(i, j, n) for every cell once. equivalence()
    involves loops/divisions, and it's called constantly during search, so
    this table lookup is a large speedup over calling it fresh every time."""
    return [[neighbors(i, j, n) for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# The two rules
# ---------------------------------------------------------------------------

def rule_ok(state_self, nbr_states, allow_branch=False, branch_over_budget=False):
    """True if this cell currently satisfies its rule, given its own state
    and its 6 neighbors' states (in cyclic ring order).

    allow_branch=False: strict path rule - exactly 2 live neighbors, not
    ring-adjacent (ortho). Every live cell is a pure pass-through, so no
    branching is possible.

    allow_branch=True: relaxed rule - the live-neighbor set may have size
    2 or 3, with the only requirement being that no two of them are
    ring-adjacent. At size 2 this is identical to the strict rule (meta or
    para spacing). At size 3, "no two adjacent" on a 6-cycle can only be
    satisfied by the two alternating triangles {0,2,4} or {1,3,5} - so it
    forces equal spacing automatically and produces a genuine 3-way branch
    point, permitting the path network to fork.

    branch_over_budget: this is the one piece of GLOBAL state in an
    otherwise purely local rule set. It's True when the count of branch
    cells (live, 3 non-adjacent neighbors) across the whole grid currently
    exceeds the allowed budget (see solve()'s branch_limit). When True,
    every count-3 cell is rejected, regardless of its own individual
    spacing being fine - the cap is enforced as a whole, not by singling
    out particular "excess" cells. When False, count-3 cells are accepted
    exactly as in the plain allow_branch rule. So branch cells are free
    until the global count crosses the limit, then all of them come under
    pressure at once until enough convert back to count-2 (or die) to
    bring the total back under budget.
    """
    if state_self:
        live_idx = [k for k, s in enumerate(nbr_states) if s]
        cnt = len(live_idx)
        allowed_counts = (2, 3) if allow_branch else (2,)
        if cnt not in allowed_counts:
            return False
        # no two chosen neighbors may be ring-adjacent (ortho, dist 1)
        for a_pos in range(len(live_idx)):
            for b_pos in range(a_pos + 1, len(live_idx)):
                a, b = live_idx[a_pos], live_idx[b_pos]
                d = (b - a) % 6
                d = min(d, 6 - d)
                if d == 1:
                    return False
        if cnt == 3 and branch_over_budget:
            return False
        return True
    else:
        # Rule 2 (dead cells): no 3 consecutive (ring-adjacent) dead
        # neighbors anywhere in the cyclic order.
        for k in range(6):
            if not nbr_states[k] and not nbr_states[(k + 1) % 6] and not nbr_states[(k + 2) % 6]:
                return False
        return True


def cell_ok(grid, n, i, j, nbr_table=None, allow_branch=False, branch_over_budget=False):
    nbrs = nbr_table[i][j] if nbr_table is not None else neighbors(i, j, n)
    nbr_states = [grid[a][b] for a, b in nbrs]
    return rule_ok(grid[i][j], nbr_states, allow_branch=allow_branch,
                    branch_over_budget=branch_over_budget)


def is_branch_cell(grid, i, j, nbrs):
    """A live cell with exactly 3 live neighbors (regardless of spacing -
    used only for counting/budget purposes; the spacing check itself is
    rule_ok's job)."""
    return grid[i][j] and sum(1 for a, b in nbrs if grid[a][b]) == 3


# ---------------------------------------------------------------------------
# Rule lookup table + exact local repair (large-neighborhood search)
# ---------------------------------------------------------------------------

_RULE_TABLES = {}


def rule_table(allow_branch, branch_over_budget):
    """Precomputed rule_ok for every (own state, 6-bit neighbor mask)
    combination, indexed by (state << 6) | mask where bit k of mask is the
    state of ring neighbor k. rule_ok only depends on those 7 bits (plus
    the two flags), so 128 entries replace all of its per-call work."""
    key = (bool(allow_branch), bool(branch_over_budget))
    table = _RULE_TABLES.get(key)
    if table is None:
        table = [False] * 128
        for own in (0, 1):
            for mask in range(64):
                nbr = [bool((mask >> k) & 1) for k in range(6)]
                table[(own << 6) | mask] = rule_ok(
                    bool(own), nbr, allow_branch=allow_branch,
                    branch_over_budget=branch_over_budget)
        _RULE_TABLES[key] = table
    return table


_COMPLETION_TABLES = {}


def completion_table(allow_branch, branch_over_budget):
    """Lookahead table. Index (pend << 7) | known, where both are 7-bit
    masks over a cell's closed neighborhood (bit 6 = the cell itself, bit k
    = ring neighbor k, same layout as rule_table). `pend` marks positions
    whose state is still undecided, `known` gives the decided positions
    (must not overlap pend). Entry is True iff SOME assignment of the
    pending positions makes the cell satisfy the rule; False means the cell
    is already doomed to be violated whatever happens next."""
    key = (bool(allow_branch), bool(branch_over_budget))
    t = _COMPLETION_TABLES.get(key)
    if t is None:
        rt = rule_table(allow_branch, branch_over_budget)
        t = [False] * (128 * 128)
        for pend in range(128):
            subs = []
            sub = pend
            while True:
                subs.append(sub)
                if sub == 0:
                    break
                sub = (sub - 1) & pend
            for known in range(128):
                if known & pend:
                    continue
                t[(pend << 7) | known] = any(rt[known | sb] for sb in subs)
        _COMPLETION_TABLES[key] = t
    return t


def _region_layers(center, nbr_table, radius):
    """BFS layers around center: layers[k] = cells at hex distance k."""
    seen = {center}
    layers = [[center]]
    for _ in range(radius):
        layer = []
        for c in layers[-1]:
            for d in nbr_table[c[0]][c[1]]:
                if d not in seen:
                    seen.add(d)
                    layer.append(d)
        layers.append(layer)
    return layers


def lns_repair(grid, nbr_table, center, radius, locked, wall_cells,
               allow_branch, branch_over_budget, rng, node_limit,
               accept_equal=True, current_first=False, lookahead=False):
    """Exact local repair around `center`.

    Frees every non-locked cell within hex distance `radius` of `center`
    (distance measured through nbr_table, so it wraps across the rotated
    seams correctly) and runs a branch-and-bound over their 2^k
    assignments, minimizing the number of violated cells among the freed
    cells and their immediate neighbors. Everything outside is frozen, so
    that count is exactly the change in total violations.

    Cells are assigned ring by ring outward from `center`.

    Search options (neither changes what is found when the search
    completes, only how fast):
      current_first try each cell's existing value first (so the search
                    explores minimal changes first); otherwise a random
                    value first.
      lookahead     after each assignment, test every affected constraint
                    for whether ANY completion of its still-unassigned
                    cells can satisfy it; if not it is counted as violated
                    now, so doomed branches are cut early instead of when
                    their last cell is assigned.

    The bound starts at (current violations + 1) if accept_equal (so
    equal-quality *different* assignments are accepted as sideways moves)
    or at current violations (strict improvements only).

    `grid` is restored before returning; nothing is modified. Returns
    (assignment, free_cells, cur, best, nodes) where assignment is a list
    of booleans parallel to free_cells (in search order), or None if
    nothing acceptable was found within node_limit search nodes.
    """
    layers = _region_layers(center, nbr_table, radius)
    free = [c for layer in layers for c in layer if c not in locked]
    if not free:
        return None, free, 0, 0, 0
    nfree = len(free)
    fidx = {c: i for i, c in enumerate(free)}

    # constraints touched by the freed cells: each free cell and its nbrs
    check = set()
    for c in free:
        check.add(c)
        check.update(nbr_table[c[0]][c[1]])
    check -= wall_cells

    xpos = []                       # per constraint: (bit, a, b, free index or -1)
    dependents = [[] for _ in range(nfree)]
    for x in check:
        pos = [(1 << 6, x[0], x[1], fidx.get(x, -1))]
        for k, (a, b) in enumerate(nbr_table[x[0]][x[1]]):
            pos.append((1 << k, a, b, fidx.get((a, b), -1)))
        deps = {fi for (_, _, _, fi) in pos if fi >= 0}
        if not deps:
            continue
        xid = len(xpos)
        xpos.append(pos)
        for fi in deps:
            dependents[fi].append(xid)
    nx = len(xpos)

    table = rule_table(allow_branch, branch_over_budget)
    if lookahead:
        feas = completion_table(allow_branch, branch_over_budget)

        def bad_now(xid, idx):
            known = 0
            pend = 0
            for bit, a, b, fi in xpos[xid]:
                if fi > idx:
                    pend |= bit
                elif grid[a][b]:
                    known |= bit
            return not feas[(pend << 7) | known]
    else:
        def bad_now(xid, idx):
            known = 0
            for bit, a, b, fi in xpos[xid]:
                if fi > idx:
                    return False        # not decidable yet
                if grid[a][b]:
                    known |= bit
            return not table[known]

    cur = sum(1 for xid in range(nx) if bad_now(xid, nfree))
    orig = [grid[c[0]][c[1]] for c in free]
    bad = [False] * nx
    st = {"best": cur + 1 if accept_equal else cur, "assign": None, "nodes": 0}

    def rec(idx, partial, diff):
        if idx == nfree:
            if diff > 0:
                st["best"] = partial
                st["assign"] = [grid[c[0]][c[1]] for c in free]
            return
        if st["nodes"] >= node_limit:
            return
        st["nodes"] += 1
        ci, cj = free[idx]
        deps = dependents[idx]
        o = orig[idx]
        if current_first:
            vals = (o, not o)
        else:
            vals = (True, False) if rng.random() < 0.5 else (False, True)
        for val in vals:
            grid[ci][cj] = val
            newly = []
            for xid in deps:
                if not bad[xid] and bad_now(xid, idx):
                    bad[xid] = True
                    newly.append(xid)
            p = partial + len(newly)
            if p < st["best"]:
                rec(idx + 1, p, diff + (val != o))
            for xid in newly:
                bad[xid] = False

    rec(0, 0, 0)
    for c, v in zip(free, orig):
        grid[c[0]][c[1]] = v
    return st["assign"], free, cur, st["best"], st["nodes"]


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def solve(n, fixed_live=None, p_noise=0.25, max_iters=300_000, seed=None,
          verbose=False, stall_window=3000, kick_window=15_000, kick_size=None,
          allow_branch=False, branch_limit=0.05, wall_boundary=False,
          lns_radius=0, lns_max_radius=None, lns_stall=1500, lns_every=250,
          lns_node_limit=20_000, lns_sideways=True,
          lns_current_first=False, lns_lookahead=True, log_repairs=False,
          stats=None):
    """
    n           : base tile is n x n
    fixed_live  : optional set of (i, j) cells hard-fixed live (never
                  flipped). Off by default - this is a tool for kicking the
                  search out of repetitive patterns when you want it, not
                  something that should run by default.
    p_noise     : when the greedy move offers no improvement (best_delta >=
                  0), this is the probability of taking a random flip
                  instead of the (non-improving) greedy one. If a genuinely
                  improving move exists, it's always taken - no point
                  randomizing away a free improvement.
    stall_window: if the best-seen unsat count hasn't improved in this many
                  iterations, do a moderate kick: randomize a handful of
                  cells around the current violated frontier.
    kick_window : if still stuck after this many iterations (much longer
                  than stall_window), the initial state itself is probably
                  the problem - do a bigger kick, randomizing a larger
                  batch of cells across the grid rather than just one
                  cell's neighborhood.
    kick_size   : how many cells the big kick randomizes. Either an int
                  (absolute cell count) or a float in (0, 1] (fraction of
                  the n x n grid, e.g. 0.05 for 5%). Defaults to 5%.
    allow_branch: if True, live cells may have 2 or 3 live neighbors (with
                  no two ring-adjacent), permitting 3-way branch points.
                  If False (default), the strict no-branching path rule -
                  exactly 2, non-adjacent.
    branch_limit: only relevant when allow_branch=True. Caps the number of
                  branch cells (live, 3 neighbors) - a global budget, not a
                  per-cell rule. Either an int (an absolute cap - usually
                  small, e.g. 5 or 10) or a float in (0, 1] (a fraction of
                  the current live-cell count, e.g. 0.05 for 5%). Below the
                  cap, branch cells are free (this is what makes
                  allow_branch converge so much faster than the strict
                  rule); once the count exceeds the cap, every branch cell
                  comes under pressure until enough resolve back to
                  pass-through to bring the total back down. Default 0.05
                  (5% of live cells).
    wall_boundary: if True, cells on the OUTER hexagon boundary - i=n-1 or
                  j=n-1 - are permanently fixed dead and exempt from rule
                  checking entirely (a hard wall, not a soft preference).
                  The i=0/j=0 edges are deliberately left alone: those
                  fold into the hexagon's interior (where R0 meets R1 and
                  R2 through the periodic wrap), not its outer boundary -
                  verified by checking the hexagon's actual vertex
                  geometry before implementing this. The intent is to get
                  a genuinely walled-in hexagon directly from the solve -
                  paths naturally curve away from a dead wall rather than
                  needing to be pruned after the fact in the renderer.
                  Render with hex_render.py's --shape hexagon (radius 1)
                  to see the walled result; no --prune-boundary needed.
    lns_radius  : 0 = off (default). Otherwise, once the search has gone
                  lns_stall iterations without a new best, pick a violated
                  cell every lns_every iterations and exactly re-solve all
                  cells within this hex distance of it (branch-and-bound,
                  see lns_repair) instead of flipping one cell. Radius 1 =
                  7 free cells, 2 = 19, 3 = 37, 4 = 61.
    lns_max_radius: if set above lns_radius, a repair that can't strictly
                  improve at lns_radius is retried at larger radii up to
                  this one (default: same as lns_radius, no escalation).
    lns_stall   : iterations without a new best before repairs start.
    lns_every   : while stalled, one repair attempt per this many iterations.
    lns_node_limit: search-node cap per repair attempt (bounds the cost of
                  large radii).
    lns_sideways: accept a repair that finds an equally-good but different
                  assignment (a plateau move). If False, strict
                  improvements only.
    lns_current_first: try each cell's existing value before the flipped one
                  (default False = random value first). Looks better on
                  isolated repairs but made full runs worse in testing,
                  because repairs become too deterministic.
    lns_lookahead: prune a repair branch as soon as some constraint can no
                  longer be satisfied by any completion (default True).
                  These three options change only how fast the repair
                  search runs, not what an unbounded search would find.
    log_repairs : print one line per repair attempt (radius, region size,
                  local violations before/after, nodes used, whether the
                  node cap was hit, outcome, global unsat before/after).
                  Off by default: long runs make hundreds of these.
    stats       : optional dict, filled with lns_calls / lns_improved /
                  lns_sideways / lns_nodes counters.
    """
    rng = random.Random(seed)
    if stats is None:
        stats = {}
    for k in ("lns_calls", "lns_improved", "lns_sideways", "lns_nodes", "lns_capped"):
        stats.setdefault(k, 0)
    if lns_max_radius is None or lns_max_radius < lns_radius:
        lns_max_radius = lns_radius
    fixed_live = set(fixed_live or ())
    wall_cells = set()
    if wall_boundary:
        wall_cells = {(i, j) for i in range(n) for j in range(n)
                      if i == n - 1 or j == n - 1}
    locked = fixed_live | wall_cells
    nbr_table = build_neighbor_table(n)
    if kick_size is None:
        kick_size = max(4, (n * n) // 20)
    elif isinstance(kick_size, float):
        kick_size = max(4, round(kick_size * n * n))

    # branch_limit as given may be a fixed absolute count (int) or a
    # fraction of the live-cell count (float) that needs recomputing as
    # live_count changes - branch_limit_frac holds the float if so.
    branch_limit_frac = branch_limit if isinstance(branch_limit, float) else None

    grid = [[rng.random() < 0.5 for _ in range(n)] for _ in range(n)]
    for (i, j) in fixed_live:
        grid[i][j] = True
    for (i, j) in wall_cells:
        grid[i][j] = False

    live_count = sum(1 for i in range(n) for j in range(n) if grid[i][j])
    branch_set = set()
    if allow_branch:
        for i in range(n):
            for j in range(n):
                if is_branch_cell(grid, i, j, nbr_table[i][j]):
                    branch_set.add((i, j))
    if branch_limit_frac is not None:
        branch_limit = max(0, int(branch_limit_frac * live_count))
    branch_over_budget = allow_branch and len(branch_set) > branch_limit

    def ok(i, j):
        if (i, j) in wall_cells:
            return True
        return cell_ok(grid, n, i, j, nbr_table, allow_branch=allow_branch,
                        branch_over_budget=branch_over_budget)

    unsat = {(i, j) for i in range(n) for j in range(n) if not ok(i, j)}

    def refresh(cells):
        for (i, j) in cells:
            if ok(i, j):
                unsat.discard((i, j))
            else:
                unsat.add((i, j))

    def update_branch_bookkeeping(affected_cells, delta_live):
        """After a flip (or kick) has already been applied to `grid`,
        update live_count, branch_set membership for the affected cells,
        and recheck whether the global branch budget has just been
        crossed in either direction. Returns True if it crossed (meaning
        every branch cell's validity needs re-checking, not just the
        locally affected ones)."""
        nonlocal live_count, branch_limit, branch_over_budget
        live_count += delta_live
        if not allow_branch:
            return False
        for (i, j) in affected_cells:
            if is_branch_cell(grid, i, j, nbr_table[i][j]):
                branch_set.add((i, j))
            else:
                branch_set.discard((i, j))
        if branch_limit_frac is not None:
            branch_limit = max(0, int(branch_limit_frac * live_count))
        new_over = len(branch_set) > branch_limit
        crossed = new_over != branch_over_budget
        branch_over_budget = new_over
        return crossed

    def random_kick(size):
        """Randomize `size` random non-fixed cells - a bigger perturbation
        than a single-cell flip, meant to shake the search out of a basin
        that the initial random state fell into."""
        pool = [c for c in [(i, j) for i in range(n) for j in range(n)]
                if c not in locked]
        batch = rng.sample(pool, min(size, len(pool)))
        affected = set()
        delta_live = 0
        for (bi, bj) in batch:
            old = grid[bi][bj]
            new_state = rng.random() < 0.5
            grid[bi][bj] = new_state
            delta_live += int(new_state) - int(old)
            affected.add((bi, bj))
            affected.update(nbr_table[bi][bj])
        crossed = update_branch_bookkeeping(affected, delta_live)
        refresh(affected)
        if crossed:
            refresh(branch_set)

    def try_lns(v, it):
        """Exact repair around violated cell v, escalating radius up to
        lns_max_radius. Commits the result; returns True if grid changed.
        With log_repairs, prints one line per call."""
        choice = None   # (assign, free) to commit
        kind = None
        unsat_before = len(unsat)
        parts = []
        for r in range(lns_radius, lns_max_radius + 1):
            assign, free, cur, best, nodes = lns_repair(
                grid, nbr_table, v, r, locked, wall_cells, allow_branch,
                branch_over_budget, rng, lns_node_limit,
                accept_equal=lns_sideways,
                current_first=lns_current_first, lookahead=lns_lookahead)
            stats["lns_calls"] += 1
            stats["lns_nodes"] += nodes
            capped = nodes >= lns_node_limit
            if capped:
                stats["lns_capped"] += 1
            after = best if assign is not None else cur
            parts.append(f"r{r}[free {len(free)}, local {cur}->{after}, "
                         f"{nodes} nodes{' CAPPED' if capped else ''}]")
            if assign is None:
                continue
            if best < cur:
                choice, kind = (assign, free), "improved"
                break
            if choice is None:
                choice, kind = (assign, free), "sideways"
        if choice is None:
            if log_repairs:
                print(f"iter {it}: repair at {v} " + " ".join(parts)
                      + f" -> no change, unsat {unsat_before}")
            return False
        assign, free = choice
        stats["lns_" + kind] += 1
        changed = set()
        delta_live = 0
        for c, val in zip(free, assign):
            old = grid[c[0]][c[1]]
            if old != val:
                grid[c[0]][c[1]] = val
                delta_live += int(val) - int(old)
                changed.add(c)
        affected = set(changed)
        for c in changed:
            affected.update(nbr_table[c[0]][c[1]])
        crossed = update_branch_bookkeeping(affected, delta_live)
        refresh(affected)
        if crossed:
            refresh(branch_set)
        if log_repairs:
            print(f"iter {it}: repair at {v} " + " ".join(parts)
                  + f" -> {kind}, unsat {unsat_before}->{len(unsat)}")
        return True

    best_unsat = len(unsat)
    since_improved = 0

    for it in range(max_iters):
        if not unsat:
            if verbose:
                print(f"solved in {it} flips")
            return grid, it

        if len(unsat) < best_unsat:
            best_unsat = len(unsat)
            since_improved = 0
        else:
            since_improved += 1

        if since_improved > 0 and since_improved % kick_window == 0:
            if verbose:
                print(f"iter {it}: stuck {since_improved} iters at {len(unsat)} unsat, big kick ({kick_size} cells)")
            random_kick(kick_size)
            continue
        elif since_improved > 0 and since_improved % stall_window == 0:
            if verbose:
                print(f"iter {it}: stalled at {len(unsat)} unsat, small kick")
            random_kick(max(4, kick_size // 6))
            continue

        if (lns_radius > 0 and since_improved >= lns_stall
                and (since_improved - lns_stall) % lns_every == 0):
            if try_lns(rng.choice(tuple(unsat)), it):
                continue

        v = rng.choice(tuple(unsat))
        vi, vj = v
        candidates = [c for c in [v] + nbr_table[vi][vj] if c not in locked]
        if not candidates:
            continue

        # Trial deltas use the CURRENT global branch_over_budget snapshot
        # (not re-simulated per candidate) - re-deriving the true global
        # effect of every candidate flip would mean rechecking the whole
        # branch_set 7 times per step, which is far more expensive than
        # it's worth. The real bookkeeping (and any resulting full
        # branch_set refresh) happens once, after the flip is committed.
        best_ties, best_delta = [], None
        for c in candidates:
            ci, cj = c
            affected = [c] + nbr_table[ci][cj]
            pre = sum(0 if ok(*a) else 1 for a in affected)
            grid[ci][cj] = not grid[ci][cj]
            post = sum(0 if ok(*a) else 1 for a in affected)
            grid[ci][cj] = not grid[ci][cj]
            delta = post - pre
            if best_delta is None or delta < best_delta:
                best_delta, best_ties = delta, [c]
            elif delta == best_delta:
                best_ties.append(c)

        flip = rng.choice(best_ties)
        if best_delta >= 0 and rng.random() < p_noise:
            flip = rng.choice(candidates)

        fi, fj = flip
        old = grid[fi][fj]
        grid[fi][fj] = not old
        delta_live = int(grid[fi][fj]) - int(old)
        affected = set([flip] + nbr_table[fi][fj])
        crossed = update_branch_bookkeeping(affected, delta_live)
        refresh(affected)
        if crossed:
            refresh(branch_set)

        if verbose and it % 20000 == 0:
            extra = f", branches {len(branch_set)}/{branch_limit}" if allow_branch else ""
            print(f"iter {it}: {len(unsat)} unsat{extra}")

    if verbose:
        print(f"gave up after {max_iters} iters, {len(unsat)} unsat remaining")
    return grid, None


def solve_with_restarts(n, fixed_live=None, p_noise=0.25, attempt_iters=150_000,
                          max_attempts=50, base_seed=0, verbose=False,
                          allow_branch=False, branch_limit=0.05, wall_boundary=False,
                          **lns_kwargs):
    """Run solve() repeatedly with fresh seeds, each capped at attempt_iters,
    until one hits zero violations. Some random starts plateau in a local
    minimum for a given n; rather than fight one stuck run with noise
    tricks, it's often cheaper to just try a new random start - the
    per-attempt cost is small and success rate per attempt is decent for
    moderate n."""
    for attempt in range(max_attempts):
        seed = base_seed + attempt
        grid, flips = solve(n, fixed_live=fixed_live, p_noise=p_noise,
                             max_iters=attempt_iters, seed=seed,
                             allow_branch=allow_branch, branch_limit=branch_limit,
                             wall_boundary=wall_boundary, **lns_kwargs)
        if flips is not None:
            if verbose:
                print(f"attempt {attempt} (seed {seed}): solved in {flips} flips")
            return grid, attempt, flips
        if verbose:
            print(f"attempt {attempt} (seed {seed}): no solution within {attempt_iters} iters")
    return None, max_attempts, None


# ---------------------------------------------------------------------------
# Rendering (PIL)
# ---------------------------------------------------------------------------

def is_extra_cell(i, j, n):
    """True if (i, j), within the bounding square [-n, n-1] x [-n, n-1] of
    the three rotated rhombus copies, falls in one of the two leftover
    corner triangles rather than in the hexagon itself. See the derivation:
    inside R0, j-i <= n-1; inside R1, j-i <= n; inside R2, i-j <= n-1. So
    j-i > n or i-j >= n means the cell can't be in any of the three
    rhombi. (The two conditions are mutually exclusive and together
    account for exactly the n^2 leftover cells.)"""
    return (j - i > n) or (i - j >= n)


def find_cycles(grid, n, nbr_table=None):
    """Distinct cycles of the solved pattern: connected components of live
    cells, where adjacency goes through the periodic identification
    (nbr_table / equivalence), so a path that leaves one edge of the tile
    and re-enters through a rotated seam is ONE cycle, not two broken
    pieces. Under Rule 1 (exactly 2 live neighbors per live cell) every
    component is a simple closed loop; with --allow-branch a component can
    be a branching network instead (reported via `branching`).

    Returns (cycle_id, sizes, branching):
      cycle_id  dict {(i, j): id} for every live cell. Ids are ordered by
                size, largest first (ties broken by position), so they're
                stable for a given grid.
      sizes     list of cell counts, indexed by id.
      branching True if any live cell has more than 2 live neighbors.
    """
    if nbr_table is None:
        nbr_table = build_neighbor_table(n)
    seen = set()
    comps = []
    branching = False
    for i in range(n):
        for j in range(n):
            if not grid[i][j] or (i, j) in seen:
                continue
            seen.add((i, j))
            stack = [(i, j)]
            comp = []
            while stack:
                ci, cj = stack.pop()
                comp.append((ci, cj))
                live_nbrs = [(a, b) for (a, b) in nbr_table[ci][cj] if grid[a][b]]
                if len(live_nbrs) > 2:
                    branching = True
                for d in live_nbrs:
                    if d not in seen:
                        seen.add(d)
                        stack.append(d)
            comps.append(comp)
    comps.sort(key=lambda c: (-len(c), min(c)))
    cycle_id = {}
    for k, comp in enumerate(comps):
        for cell in comp:
            cycle_id[cell] = k
    return cycle_id, [len(c) for c in comps], branching


def build_rotation_table(n):
    """rot[i][j][k] = how many 120-degree rotations (0, 1 or 2) equivalence()
    applies to fold cell (i, j)'s k-th ring neighbor back onto the base tile.
    Walking around a cycle and adding these up tells you whether the cycle,
    developed in the plane, comes back to where it started (see cycle_lifts)."""
    table = [[[0] * 6 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k, (di, dj) in enumerate(RING):
                a, b = i + di, j + dj
                r = 0
                while ((a // n) + (b // n)) % 3 != 0:
                    a, b = -b - 1, a - b
                    r += 1
                table[i][j][k] = r % 3
    return table


def cycle_lifts(grid, n, cycle_id, sizes, nbr_table=None, rot=None):
    """For each cycle (same ids/order as find_cycles), whether it is
    CONNECTED in the full hexagon: True if the cycle only closes after three
    trips around the base tile (one loop three times as long, winding around
    the hexagon's 3-fold center), False if it closes after one trip (so the
    hexagon holds three separate copies of it). Returns None for a component
    that isn't a simple cycle (branching).

    Exact and cheap: a cycle's development in the plane closes iff the 120-degree
    rotations picked up folding each step back onto the base tile sum to 0
    mod 3 (a simple closed curve here can only have rotation or identity
    holonomy, never a translation)."""
    if nbr_table is None:
        nbr_table = build_neighbor_table(n)
    if rot is None:
        rot = build_rotation_table(n)
    start = {}
    for c, k in cycle_id.items():
        if k not in start or c < start[k]:
            start[k] = c
    out = []
    for k in range(len(sizes)):
        s0 = start[k]
        c, prev, total, steps = s0, None, 0, 0
        ok = True
        while True:
            nb = nbr_table[c[0]][c[1]]
            live = [x for x, (a, b) in enumerate(nb) if grid[a][b]]
            if len(live) != 2:
                ok = False
                break
            x = live[0] if (prev is None or nb[live[0]] != prev) else live[1]
            total += rot[c[0]][c[1]][x]
            prev, c = c, nb[x]
            steps += 1
            if c == s0 or steps > sizes[k] + 1:
                break
        out.append(None if not ok else (total % 3 != 0))
    return out


def count_loops(sizes, lifts):
    """Closed loops in the full hexagon: 1 for a connected cycle, 3 for a
    cycle that appears as three separate copies."""
    return sum(1 if (l is True) else 3 for l in lifts)


def cycle_palette(k):
    """k visually distinct RGB colors (golden-ratio hue steps, with
    saturation/brightness cycling so neighboring ids differ in more than
    hue alone). Bright enough to read against the dark dead cells."""
    out = []
    for idx in range(k):
        h = (idx * 0.6180339887) % 1.0
        sat, val = ((0.70, 0.95), (0.90, 0.78), (0.55, 0.90))[idx % 3]
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def describe_cycles(sizes, branching, show=12):
    kind = "components (branching network)" if branching else "cycles"
    head = ", ".join(str(sz) for sz in sizes[:show])
    more = f", ... (+{len(sizes) - show} more)" if len(sizes) > show else ""
    return f"{len(sizes)} {kind}; cell counts, largest first: {head}{more}"


def render_png(grid, n, path, cell_px=32, live_color=(222, 184, 135),
               dead_color=(15, 15, 15), edge_color=(106, 90, 205),
               background=(255, 255, 255), margin_px=24, full_hex=False,
               color_cycles=False):
    """
    Render the solved grid to a PNG with PIL.

    color_cycles: color each live cell by which distinct cycle it belongs
    to (see find_cycles) instead of the single live_color. Cycles are
    found on the base tile through the periodic identification, so every
    rotated copy of a cycle in a full_hex render gets the same color, and
    a cycle that crosses a seam is one color, not two.

    Cell (i, j) is placed at x = 0.5*j - i, y = 0.866*j (unit spacing) -
    this is the axis geometry actually required by RING (120 degrees
    between the i- and j-axes), not the more common 60-degree axial
    convention, and it's oriented to lean the same direction as the
    reference image. See the earlier debugging notes if you change RING
    and need to re-derive this.

    full_hex: if True, render the full hexagon made of the 3 rotated
    copies of the base rhombus instead of just the base rhombus itself.
    Each cell's color is looked up via equivalence() rather than by
    physically rotating anything, so the three copies are guaranteed to
    align exactly - no manual alignment needed. The leftover corner cells
    outside the hexagon (see is_extra_cell) are simply left unpainted.
    """
    # circumradius that makes unit-spacing hexagons touch edge-to-edge
    r = cell_px / math.sqrt(3)

    def center_px(i, j):
        x = 0.5 * j - i
        y = 0.866 * j
        return x * cell_px, -y * cell_px  # negate y: PIL's y-axis points down

    if full_hex:
        i_lo, i_hi = -n, n - 1
        j_lo, j_hi = -n, n - 1
    else:
        i_lo, i_hi = 0, n - 1
        j_lo, j_hi = 0, n - 1

    # figure out pixel bounds across all cells so nothing gets clipped
    corners = [center_px(i, j) for i in (i_lo, i_hi) for j in (j_lo, j_hi)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x_min, x_max = min(xs) - r, max(xs) + r
    y_min, y_max = min(ys) - r, max(ys) + r

    width = int(x_max - x_min) + 2 * margin_px
    height = int(y_max - y_min) + 2 * margin_px
    ox = margin_px - x_min
    oy = margin_px - y_min

    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)

    cycle_id = palette = None
    if color_cycles:
        cycle_id, sizes, _ = find_cycles(grid, n)
        palette = cycle_palette(len(sizes))

    def hexagon(cx, cy, r):
        # pointy-top hexagon (vertex pointing up), matching the earlier
        # matplotlib RegularPolygon(orientation=0) rendering
        pts = []
        for k in range(6):
            angle = math.radians(90 + 60 * k)
            pts.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
        return pts

    for i in range(i_lo, i_hi + 1):
        for j in range(j_lo, j_hi + 1):
            if full_hex and is_extra_cell(i, j, n):
                continue
            ci, cj = equivalence(i, j, n) if full_hex else (i, j)
            cx, cy = center_px(i, j)
            cx, cy = cx + ox, cy + oy
            if not grid[ci][cj]:
                color = dead_color
            elif cycle_id is not None:
                color = palette[cycle_id[(ci, cj)]]
            else:
                color = live_color
            draw.polygon(hexagon(cx, cy, r), fill=color, outline=edge_color, width=1)

    img.save(path)
    return path


def write_grid_txt(grid, n, path):
    """Write the solved grid as plain 0/1 text - decoupled from any
    particular rendering approach so other programs can consume it.

    Format: first line is n, followed by n lines of n characters each
    ('1' for live, '0' for dead), row-major (row i, then columns j in
    that row), matching the (i, j) indexing used everywhere else here.
    """
    with open(path, "w") as f:
        f.write(f"{n}\n")
        for i in range(n):
            f.write("".join("1" if grid[i][j] else "0" for j in range(n)))
            f.write("\n")
    return path


def unique_path(path):
    """Append _1, _2, _3, ... before the extension until an unused path is
    found, so repeated runs (e.g. with no --seed) don't clobber earlier
    output."""
    base, ext = os.path.splitext(path)
    n = 1
    candidate = f"{base}_{n}{ext}"
    while os.path.exists(candidate):
        n += 1
        candidate = f"{base}_{n}{ext}"
    return candidate


def parse_kick_size(s):
    """Accepts either a plain integer cell count ("40") or a percentage
    of the grid ("5%")."""
    s = s.strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    return int(s)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="WalkSAT-style local search for the hex non-branching-path pattern."
    )
    parser.add_argument("-n", type=int, default=24,
                         help="grid is n x n (default: 24)")
    parser.add_argument("--p-noise", type=float, default=0.25,
                         help="probability of a random flip when no improving move exists (default: 0.25)")
    parser.add_argument("--max-iters", type=int, default=300_000,
                         help="iteration cap (default: 300000)")
    parser.add_argument("--seed", type=int, default=None,
                         help="RNG seed. Omit for a fresh random run each time (default: random)")
    parser.add_argument("--stall-window", type=int, default=3000,
                         help="iterations without improvement before a small kick (default: 3000)")
    parser.add_argument("--kick-window", type=int, default=15_000,
                         help="iterations without improvement before a big kick (default: 15000)")
    parser.add_argument("--kick-size", type=parse_kick_size, default=None,
                         help="cells randomized by a big kick - a plain count (e.g. 40) "
                              "or a percentage of the grid (e.g. 5%%) (default: 5%%)")
    parser.add_argument("--fixed-fraction", type=float, default=0.0,
                         help="fraction of cells hard-fixed live as anti-clump seeds, 0-1 (default: 0, off)")
    parser.add_argument("--allow-branch", action="store_true",
                         help="relax the live-cell rule to allow 2 or 3 equally-spaced "
                              "live neighbors (3 permits branch points), instead of "
                              "strictly 2 non-adjacent (default: off, strict paths only)")
    parser.add_argument("--branch-limit", type=parse_kick_size, default=0.05,
                         help="cap on branch cells, only used with --allow-branch - a "
                              "plain count (e.g. 8; usually small) or a percentage of "
                              "live cells (e.g. 5%%) (default: 5%%)")
    parser.add_argument("--wall-boundary", action="store_true",
                         help="fix the OUTER hexagon boundary (i=n-1 or j=n-1) permanently "
                              "dead and exempt from rule checking - a hard wall baked into "
                              "the solve itself, so paths naturally curve away from the "
                              "true hexagon edge instead of needing --prune-boundary "
                              "afterward in hex_render.py. The i=0/j=0 edges are left alone "
                              "(they fold into the hexagon's interior, not its outer "
                              "boundary) (default: off)")
    parser.add_argument("--lns-radius", type=int, default=3,
                         help="when stalled, exactly re-solve all cells within this hex "
                              "distance of a violated cell via branch-and-bound instead of "
                              "flipping one cell. 0 disables. Radius 1/2/3/4 frees 7/19/37/61 "
                              "cells; 3 worked best in testing (default: 3)")
    parser.add_argument("--lns-max-radius", type=int, default=None,
                         help="if above --lns-radius, retry failed repairs at larger radii up "
                              "to this one (default: off; slower, not helpful in testing)")
    parser.add_argument("--lns-stall", type=int, default=1500,
                         help="iterations without a new best before repairs begin (default: 1500)")
    parser.add_argument("--lns-every", type=int, default=250,
                         help="while stalled, one repair attempt per this many iterations (default: 250)")
    parser.add_argument("--lns-node-limit", type=int, default=20_000,
                         help="search-node cap per repair attempt (default: 20000)")
    parser.add_argument("--lns-current-first", action="store_true",
                         help="repair search tries each cell's existing value first instead of a "
                              "random one (faster per repair but worse overall in testing)")
    parser.add_argument("--no-lns-lookahead", action="store_true",
                         help="disable lookahead pruning in the repair search (slower in testing)")
    parser.add_argument("--log-repairs", action="store_true",
                         help="print one line per repair attempt (off by default; can be hundreds "
                              "of lines on a long run)")
    parser.add_argument("--lns-strict", action="store_true",
                         help="only accept repairs that strictly reduce violations "
                              "(default: also accept equal-quality different assignments)")
    parser.add_argument("-o", "--output", type=str, default="hex_output.png",
                         help="output PNG path (default: hex_output.png)")
    parser.add_argument("--cell-px", type=int, default=32,
                         help="hexagon size in pixels (default: 32)")
    parser.add_argument("--color-cycles", action="store_true",
                         help="color each distinct cycle of live cells differently in the "
                              "output image (cycles are found through the periodic boundary, so "
                              "a path crossing a seam is one cycle)")
    parser.add_argument("--full-hex", action="store_true",
                         help="render the full hexagon (3 rotated copies of the base "
                              "rhombus, auto-aligned via equivalence()) instead of just "
                              "the base rhombus (default: off)")
    parser.add_argument("--grid-output", type=str, default="hex_grid.txt",
                         help="output grid text path - first line n, then n lines "
                              "of n 0/1 characters (default: hex_grid.txt)")
    parser.add_argument("-q", "--quiet", action="store_true",
                         help="suppress progress output")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fixed_live = None
    if args.fixed_fraction > 0:
        all_cells = [(i, j) for i in range(args.n) for j in range(args.n)]
        k = int(args.n * args.n * args.fixed_fraction)
        fixed_live = set(rng.sample(all_cells, k=k))

    lns_stats = {}
    t_start = time.time()
    grid, flips = solve(args.n, fixed_live=fixed_live, p_noise=args.p_noise,
                         max_iters=args.max_iters, seed=args.seed,
                         stall_window=args.stall_window, kick_window=args.kick_window,
                         kick_size=args.kick_size, allow_branch=args.allow_branch,
                         branch_limit=args.branch_limit, wall_boundary=args.wall_boundary,
                         lns_radius=args.lns_radius, lns_max_radius=args.lns_max_radius,
                         lns_stall=args.lns_stall, lns_every=args.lns_every,
                         lns_node_limit=args.lns_node_limit,
                         lns_sideways=not args.lns_strict,
                         lns_current_first=args.lns_current_first,
                         lns_lookahead=not args.no_lns_lookahead,
                         log_repairs=args.log_repairs, stats=lns_stats,
                         verbose=not args.quiet)
    print("flips used:", flips)
    print(f"elapsed: {time.time() - t_start:.1f}s")
    if args.lns_radius > 0:
        print("repair stats:", lns_stats)

    nbr_table = build_neighbor_table(args.n)
    wall_cells = set()
    if args.wall_boundary:
        wall_cells = {(i, j) for i in range(args.n) for j in range(args.n)
                      if i == args.n - 1 or j == args.n - 1}
    bad = sum(1 for i in range(args.n) for j in range(args.n)
              if (i, j) not in wall_cells
              and not cell_ok(grid, args.n, i, j, nbr_table, allow_branch=args.allow_branch))
    print("remaining violations:", bad)

    cid_final, cycle_sizes, branching = find_cycles(grid, args.n, nbr_table)
    print(describe_cycles(cycle_sizes, branching))
    if not branching:
        lifts = cycle_lifts(grid, args.n, cid_final, cycle_sizes, nbr_table)
        print(f"closed loops in the full hexagon: {count_loops(cycle_sizes, lifts)} "
              f"({sum(1 for l in lifts if l)} cycle(s) connected, "
              f"{sum(1 for l in lifts if not l)} appearing as 3 separate copies)")

    out_path = unique_path(args.output)
    render_png(grid, args.n, out_path, cell_px=args.cell_px, full_hex=args.full_hex,
               color_cycles=args.color_cycles)
    print("saved image to", out_path)

    grid_path = unique_path(args.grid_output)
    write_grid_txt(grid, args.n, grid_path)
    print("saved grid to", grid_path)
