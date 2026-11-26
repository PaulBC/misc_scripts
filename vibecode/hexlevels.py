#!/usr/bin/env python3
"""
hexlevels.py -- mine radius-4 hex Truchet boards where greedy play is a trap.

WHAT IT FINDS
-------------
A board qualifies when

  1. it has n loops, n in --loops (default 3, 5, 7, 9, 11);
  2. its fewest flips to one loop ("best", exact, from the full table) is at
     least --min-best for that n -- hard for its loop count; and
  3. it is a first-move greedy trap: no move the greedy rule allows lies on
     any shortest path.  The greedy rule is: if some cell touches three
     loops, merge there (any such cell); otherwise make the sideways flip
     that starts a shortest run of sideways flips to a merge.  Every greedy
     move then raises the distance to a single loop, so greedy loses at least
     two flips however it breaks ties.

     With 3 loops no board is a trap: the best line is the shortest run of
     pushes to a merge, then the merge, which is exactly what greedy plays
     (optimal lines never need a split).  So at 3 loops condition 3 is
     dropped and a board qualifies on hardness alone -- a long push chain.

HOW
---
Deal a board the way hex-connect.html does (a random board, random merges
down to n loops, a random walk at n loops; the hardest of a few deals), then
walk among boards with the same loop count by sideways flips, preferring steps
that keep the board hard, and keep every qualifying board met on the way.
Each board costs one exact distance scan over the table (a few ms in numpy).

OUTPUT
------
Plain text, one board per line: 61 characters of 0 and 1, then a space and
the board's best (fewest flips to one loop).  Character i is the tile bit of
hexagon i -- the same order as `hextable.py --format text` and as the bit
layout everywhere else (`hextable.py --layout` prints it).

    0010111001010110000100111001010010000101100101010101000010100 9

The score column is optional on input: a line of bits alone is accepted
everywhere, and --score fills the column in.  Boards are unique by their bits;
with --append, boards already in the file are kept and not repeated.
hex-connect.html reads this file (hex-connect-levels.txt beside it) if present,
and uses the scores so it never has to compute them itself.

USAGE
-----
    python3 hexlevels.py --table hex-connect-r4.bin --out hex-connect-levels.txt --count 3000 --jobs 2
    python3 hexlevels.py --table r4.u64 --out hex-connect-levels.txt --count 1000 --append
    python3 hexlevels.py --check hex-connect-levels.txt --table r4.u64
    python3 hexlevels.py --score hex-connect-levels.txt --table r4.u64

--score adds (or corrects) the score column of every line, in place unless
--out names another file.  It takes about 3 ms a board.

--table takes either the raw cache (r4.u64) or the packed web file.  Needs
numpy (>= 2.0, for bitwise_count).  --check re-verifies every line of a file.
"""

import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hextable import make_figure, tile_partner, unpack_table  # noqa: E402

R = 4
CELLS, INDEX, CROSS = make_figure(R, 0)
N = len(CELLS)
PART = [[tile_partner(p, b) for p in range(6 * N)] for b in (0, 1)]
DEFAULT_MIN_BEST = {3: 6, 5: 7, 7: 9, 9: 11, 11: 12}

TAB = None


def load_table(path):
    global TAB
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\x1f\x8b":
        TAB = np.array(unpack_table(raw), dtype=np.uint64)
    else:
        TAB = np.frombuffer(raw, dtype="<u8").copy()


def classify(bits):
    """(loop count, merge cells, sideways cells)"""
    lab = [-1] * (6 * N)
    k = 0
    for s in range(6 * N):
        if lab[s] >= 0:
            continue
        p = s
        while lab[p] < 0:
            lab[p] = k
            q = PART[bits[p // 6]][p]
            lab[q] = k
            p = CROSS[q]
        k += 1
    merges, sideways = [], []
    for i in range(N):
        m = len(set(lab[6 * i:6 * i + 6]))
        if m == 3:
            merges.append(i)
        elif m == 2:
            sideways.append(i)
    return k, merges, sideways


def word(bits):
    return sum(b << i for i, b in enumerate(bits))


def best_and_geodesic(bits):
    """fewest flips to one loop, and every cell on some shortest path"""
    x = TAB ^ np.uint64(word(bits))
    c = np.bitwise_count(x)
    d = int(c.min())
    m = int(np.bitwise_or.reduce(x[c == d]))
    return d, {i for i in range(N) if m >> i & 1}


def greedy_moves(bits, max_depth=4, max_states=20000):
    """the moves the greedy rule allows, or None if the search is too deep"""
    _, merges, _ = classify(bits)
    if merges:
        return set(merges)
    layer = {(): set()}
    seen = {()}
    states = 0
    for _ in range(max_depth):
        nxt = {}
        for fs, firsts in layer.items():
            b = bits[:]
            for i in fs:
                b[i] ^= 1
            for i in classify(b)[2]:
                if i in fs:
                    continue
                key = tuple(sorted(fs + (i,)))
                if key in seen:
                    continue
                f = firsts if fs else {i}
                if key in nxt:
                    nxt[key] |= f
                    continue
                nxt[key] = set(f)
                states += 1
                if states > max_states:
                    return None
        seen |= set(nxt)
        hits = set()
        for fs, firsts in nxt.items():
            b = bits[:]
            for i in fs:
                b[i] ^= 1
            if classify(b)[1]:
                hits |= firsts
        if hits:
            return hits
        layer = nxt
    return None


def qualifies(bits, n, min_best):
    k, _, _ = classify(bits)
    if k != n:
        return False, None
    d, geo = best_and_geodesic(bits)
    if d < min_best:
        return False, d
    if n == 3:
        return True, d                       # no trap is possible at 3 loops
    g = greedy_moves(bits)
    return (g is not None and not (g & geo)), d


def deal(n, rng):
    """one board with n loops, dealt the way hex-connect.html deals"""
    for _ in range(300):
        bits = [rng.randrange(2) for _ in range(N)]
        ok = True
        for _ in range(5000):
            k, merges, sideways = classify(bits)
            if k == n:
                break
            pool = merges or sideways
            if k < n or not pool:
                ok = False
                break
            bits[rng.choice(pool)] ^= 1
        if not ok:
            continue
        for _ in range(rng.randrange(120)):
            k, _, sideways = classify(bits)
            if not sideways:
                break
            i = rng.choice(sideways)
            bits[i] ^= 1
            if classify(bits)[0] != n:
                bits[i] ^= 1
        if classify(bits)[0] == n:
            return bits
    return None


def mine(n, min_best, rng, walk=400, deals=6):
    """one walk; yields qualifying boards as words.

    One exact scan per board visited.  The scan also gives every neighbour's
    distance for free: a flip moves the distance by exactly one, down for a
    cell in the geodesic set and up for any other, so the walk chooses its
    direction before flipping.  Sideways flips keep the loop count."""
    start, sd = None, -1
    for _ in range(deals):
        b = deal(n, rng)
        if b is None:
            continue
        d, _ = best_and_geodesic(b)
        if d > sd:
            start, sd = b, d
    if start is None:
        return
    bits = start
    d, geo = best_and_geodesic(bits)
    for _ in range(walk):
        if d >= min_best:
            g = None if n == 3 else greedy_moves(bits)
            if n == 3 or (g is not None and not (g & geo)):
                yield word(bits), d
        _, _, sideways = classify(bits)
        if not sideways:
            return
        up = [i for i in sideways if i not in geo]
        down = [i for i in sideways if i in geo]
        # stay in the hard region: below the threshold climb if possible,
        # otherwise wander, leaning uphill
        if d < min_best and up:
            pool = up
        elif up and (not down or rng.random() < 0.6):
            pool = up
        else:
            pool = down
        i = rng.choice(pool)
        bits[i] ^= 1
        d, geo = best_and_geodesic(bits)


def _worker(args):
    loops, min_best, seed, table, budget_s = args
    load_table(table)
    rng = random.Random(seed)
    out = []
    t0 = time.time()
    while time.time() - t0 < budget_s:
        n = rng.choice(loops)
        out.extend((n, w, d) for w, d in mine(n, min_best[n], rng))
    return out


def to_line(w):
    return "".join(str((w >> i) & 1) for i in range(N))


def from_line(s):
    s = s.split()[0]
    if len(s) != N or set(s) - {"0", "1"}:
        raise ValueError(f"not a {N}-character 0/1 board: {s[:20]}...")
    return [int(ch) for ch in s]


def read_lines(path):
    """[(bits string, score or None)] for every board line in the file"""
    out = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            from_line(parts[0])                      # validates
            score = int(parts[1]) if len(parts) > 1 else None
            out.append((parts[0], score))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="r4.u64 or hex-connect-r4.bin")
    ap.add_argument("--out", help="text file to write")
    ap.add_argument("--count", type=int, default=1000, help="boards to find (default 1000)")
    ap.add_argument("--loops", default="3,5,7,9,11", help="loop counts, comma separated")
    ap.add_argument("--min-best", default="",
                    help="per loop count, e.g. 5:7,7:9 (default 3:6,5:7,7:9,9:11,11:12)")
    ap.add_argument("--append", action="store_true", help="keep the boards already in --out")
    ap.add_argument("--minutes", type=float, default=0, help="stop after this long (0: no limit)")
    ap.add_argument("--seed", type=int, help="random seed")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (default 1)")
    ap.add_argument("--check", metavar="FILE", help="re-verify every board in FILE and stop")
    ap.add_argument("--score", metavar="FILE",
                    help="add or correct the score column of FILE (in place, or to --out)")
    args = ap.parse_args()

    load_table(args.table)
    if len(TAB) != 1848868:
        sys.exit(f"{args.table}: {len(TAB)} entries, expected 1848868")

    min_best = dict(DEFAULT_MIN_BEST)
    for part in filter(None, args.min_best.split(",")):
        k, v = part.split(":")
        min_best[int(k)] = int(v)

    if args.check:
        lines = read_lines(args.check)
        bad, wrong, unscored, by_n = 0, 0, 0, {}
        for s, score in lines:
            bits = from_line(s)
            n = classify(bits)[0]
            d, geo = best_and_geodesic(bits)
            g = None if n == 3 else greedy_moves(bits)
            ok = n in min_best and d >= min_best[n] and (n == 3 or (g is not None and not (g & geo)))
            bad += not ok
            if score is None:
                unscored += 1
            elif score != d:
                wrong += 1
            by_n.setdefault(n, []).append(d)
        dup = len(lines) - len({s for s, _ in lines})
        print(f"{len(lines)} boards, {bad} fail, {dup} duplicates, "
              f"{wrong} wrong scores, {unscored} unscored")
        for n in sorted(by_n):
            ds = by_n[n]
            print(f"  {n} loops: {len(ds)} boards, best {min(ds)}..{max(ds)}")
        sys.exit(1 if bad or dup or wrong else 0)

    if args.score:
        lines = read_lines(args.score)
        out, seen_bits = [], set()
        added = changed = dups = 0
        t0 = time.time()
        for s, score in lines:
            if s in seen_bits:
                dups += 1
                continue
            seen_bits.add(s)
            d, _ = best_and_geodesic(from_line(s))
            added += score is None
            changed += score is not None and score != d
            out.append(f"{s} {d}")
        dest = args.out or args.score
        tmp = dest + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(out) + "\n")
        os.replace(tmp, dest)
        print(f"scored {len(out)} boards in {time.time() - t0:.1f}s -> {dest}: "
              f"{added} scores added, {changed} corrected, {dups} duplicate lines dropped")
        return

    if not args.out:
        sys.exit("--out is required unless --check or --score")
    loops = [int(x) for x in args.loops.split(",")]
    rng = random.Random(args.seed)
    seen = set()
    if args.append and os.path.exists(args.out):
        seen.update(s for s, _ in read_lines(args.out))
    have = len(seen)
    found = 0
    t0 = time.time()
    jobs = max(1, args.jobs)
    pool = None
    if jobs > 1:
        import multiprocessing as mp
        pool = mp.Pool(jobs)
    with open(args.out, "a" if args.append else "w") as f:
        while found < args.count:
            if args.minutes and time.time() - t0 > 60 * args.minutes:
                break
            # rounds of ~20 s per worker, so progress and the limits are checked
            tasks = [(loops, min_best, rng.randrange(1 << 30), args.table, 20.0)
                     for _ in range(jobs)]
            results = pool.map(_worker, tasks) if pool else [_worker(tasks[0])]
            for res in results:
                for n, w, d in res:
                    s = to_line(w)
                    if s in seen or found >= args.count:
                        continue
                    seen.add(s)
                    f.write(f"{s} {d}\n")
                    found += 1
            f.flush()
            print(f"\r  {found} new boards, {time.time() - t0:.0f}s", end="", file=sys.stderr)
    if pool:
        pool.close()
    print(f"\r  {found} new boards in {time.time() - t0:.0f}s -> {args.out}"
          + (f" ({have} were already there)" if args.append else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
