# Hex path patterns

Generate winding, brain-coral-like path patterns on a hexagonal grid, join them into a single continuous path, and render them as images - including grayscale heightmaps for casting a physical piece such as a coaster.

> **This is a Claude-generated project.** All of the code and this documentation were written by Claude, an AI model made by Anthropic, over several long conversations with the project's owner. The owner chose the goals and the design, supplied the reference pictures, ran the programs and decided what to keep; Claude wrote and revised the code and ran its own checks along the way. It has not been independently reviewed or audited, and there is no automated test suite in this repository. Treat it as experimental software: read the code and check the results before relying on them. See [Verification and limits](#verification-and-limits) for what was and wasn't tested.

## What it makes

A pattern is a set of **live** and **dead** hexagonal cells that obeys two rules:

1. **Live cells are pure pass-through points on a path.** Every live cell has exactly two live neighbors among its six, and those two are *not* adjacent to each other. So paths never branch and never dead-end.
2. **Dead cells never leave an empty patch.** No dead cell has three consecutive dead neighbors (equivalently, there is no empty 4-hex rhombus anywhere).

Together they give tight spirals and meanders with an organic, coral-like look. Because every live cell has two live neighbors, the live cells always form closed loops.

The rules can't be satisfied on an open grid (edge cells would lack a full ring of neighbors), so the grid is **periodic in a rotated way**: an n x n rhombus is the base tile, and three rotated copies of it meet at each seam. Three copies make a hexagon, and the whole pattern has 3-fold rotational symmetry. Everything is solved and stored on the single base tile; the hexagon is only built when drawing.

## Requirements

- Python 3.8 or newer (developed and tested on 3.12)
- [Pillow](https://pypi.org/project/pillow/) and [NumPy](https://pypi.org/project/numpy/)
- [SciPy](https://pypi.org/project/scipy/), only for the renderer's `heightmap` mode

```
pip install pillow numpy scipy
```

Put all the `.py` files in one folder; they import each other. The examples use `python`; on Windows use `py`.

## Quick start

Three commands: solve, connect the loops into one, render.

```
python hex_walksat.py -n 36 --max-iters 1000000 --wall-boundary --full-hex
python hex_connect.py hex_grid_1.txt --wall-boundary
python hex_render.py hex_grid_connected_1.txt --mode heightmap --shape hexagon \
       --profile rounded --thickness 1.1 --corner-radius 0.4 --floor 0 --remove-outer-wall
```

What each step does and writes:

| Step | Does | Writes |
|---|---|---|
| `hex_walksat.py` | Solves the pattern with a WalkSAT-style local search. `--wall-boundary` keeps the pattern inside a dead border; `--full-hex` makes the preview show the whole hexagon. | `hex_output_N.png` (preview) and `hex_grid_N.txt` (the grid) |
| `hex_connect.py` | Rewires neighborhoods so the separate loops become one continuous loop. | `hex_grid_connected_N.txt` |
| `hex_render.py` | Draws the grid. Here: a rounded-ridge heightmap of the hexagon with the outer wall loop removed. | `hex_render_N.png` |

Output names are numbered automatically (`_1`, `_2`, ...), so nothing is overwritten. The file number in your second command is whatever the first command reported on its `saved grid to ...` line.

For one recorded run of this exact pipeline at n=36, the solver finished in about 76,000 flips (roughly 22 seconds), and connect then took about 1.5 seconds. Your times will differ.

## The programs

| File | Purpose |
|---|---|
| `hex_walksat.py` | The solver. Also usable as a library (`generate()`). |
| `hex_connect.py` | Joins cycles into a single connected loop. |
| `hex_render.py` | Renders a grid file to a PNG (fill, skeleton or heightmap). Reads only the text grid. |
| `hex_regen.py` | Rebuilds, or makes variations of, a result from the history stored in its PNG. |
| `hex_common.py` | Shared plumbing: provenance metadata, writing grid files, output numbering. |
| `example_sequence.py` | The quick-start pipeline run from Python. |

Run any of them with `--help` for every option.

### `hex_walksat.py` - the solver

A direct local search in the WalkSAT family (no SAT solver, no CNF encoding): pick a violated cell, try flipping it or a neighbor, take the best move, and occasionally flip at random to escape local minima. When the search stalls, it also does a **neighborhood repair**: all cells within a few hex steps of a violation are re-solved exactly by branch-and-bound. Repairs make runs noticeably more reliable and are on by default; `--lns-radius 0` turns them off and gives the plain search.

Common options:

| Option | Meaning |
|---|---|
| `-n N` | The base tile is N x N (default 24). The hexagon has 3 x N x N cells. |
| `--max-iters` | Iteration cap (default 300000). Raise it for larger N. |
| `--seed` | RNG seed. If omitted, a seed is generated, printed and recorded, so the run can still be repeated. |
| `--wall-boundary` | Make the outer hexagon edge permanently dead and exempt from the rules, so paths curve away from it. The result has an extra "outer wall" loop hugging the border. |
| `--allow-branch`, `--branch-limit` | Allow a limited number of three-way branch points. (`hex_connect.py` does not support these grids.) |
| `--lns-radius` | Neighborhood repair radius (default 3; 0 disables). |
| `--full-hex` | Show the whole hexagon in the preview image instead of one rhombus. |
| `--color-cycles` | Color each distinct cycle in the preview. |
| `-o`, `--grid-output` | Output names for the preview PNG and the grid file. |
| `-q` | Suppress progress output. |

Rough guidance: up to about n=16-20 it is fast and reliable; around n=20-30 it usually converges but benefits from a generous `--max-iters`; beyond about n=30, expect to run longer or try several seeds. Success depends heavily on the seed.

### `hex_connect.py` - join the loops

A freshly solved pattern is a handful of separate loops. This tool rewires neighborhoods, always keeping every rule satisfied, until they form a single loop.

- **Phase 1** merges loops: it looks for the neighborhood whose refill leaves the fewest cycles.
- **Phase 2** runs only if that leaves one cycle that the hexagon shows as *three separate copies* (see [Cycles and loops](#cycles-and-loops)). It tries only neighborhoods around the hexagon's center, the only place a refill can turn that cycle into one connected loop.

It prints the cycle and loop counts when it finishes, and `-> a single connected loop` if it got there. **Always pass `--wall-boundary` if the grid was solved with it**; the grid file doesn't record that, and the tool keeps the outer wall loop fixed and counts only the interior. Useful options: `--radius`/`--max-radius`, `--time-limit`, `--seed`.

It is not guaranteed to succeed. If it ends at "stuck" or "time limit reached", the saved file is the best state found, not necessarily one loop.

### `hex_render.py` - draw it

| Option | Meaning |
|---|---|
| `--mode fill` | Colored hexagons, live vs. dead (default). |
| `--mode skeleton` | A line graph through the live cells. |
| `--mode heightmap` | Grayscale: height from distance to the path. `--profile rounded` gives a real rounded ridge (`--thickness`, `--corner-radius`); `--profile sigmoid` is a smooth S-curve (`--d0`, `--k`). `--floor` lifts the valleys off zero. Needs SciPy. |
| `--shape rhombus` / `hexagon` | The base tile, or the full hexagon of three rotated copies. |
| `--remove-outer-wall` | Remove the loop that encloses the whole pattern (meaningful for `--wall-boundary` solves). |
| `--prune-boundary` | Drop any loop cut off by the hexagon's edge. For solves *without* a wall. |
| `--invert` | Swap live and dead before drawing. |
| `--color-cycles` | Color each distinct cycle (fill and skeleton modes). |
| `--cell-px`, `-o` | Cell size in pixels, output name. |

`--radius` is documented in `--help` as a way to extend the hexagon to a larger region, but it was not exercised or checked in this project's testing; treat anything above 1 as unverified.

### `hex_regen.py` - rebuild or vary a result

Every PNG made by `hex_render.py` carries the history of how it was made (see [Provenance](#provenance-what-every-file-remembers)). `hex_regen.py` reads that history and runs the steps again.

```
python hex_regen.py hex_render_1.png                       # redo it exactly, checking each grid
python hex_regen.py hex_render_1.png --new-seeds           # same settings, new random seeds
python hex_regen.py hex_render_1.png --new-seeds --count 10   # a batch of ten variations
python hex_regen.py hex_render_1.png --from render --grid hex_grid_connected_1.txt \
       --set render.thickness=1.3                          # re-render the same grid, no re-solving
python hex_regen.py hex_render_1.png --dry-run             # show the steps, run nothing
```

- **Exact mode (default):** each step reuses its recorded seed and options, and each grid it produces is compared with the hash recorded for it. You're told "reproduced exactly" or "DIFFERENT" (the code has changed since, or the run isn't deterministic). The exit status is 1 if a grid differs.
- **Variations (`--new-seeds`):** a fresh seed for every step that has one. Once you have settings you like, any PNG that carries them is the recipe. With `--count N`, each variation must pass a quality gate (the solver finished, and the loops were joined into one connected loop if there is a connect step). A seed that fails is replaced by another, up to `--max-attempts` attempts (default 3 x N). Variation is within the fixed settings; only the random choices change. `--discard-failed` deletes the files of failed attempts, and a summary at the end lists every image made with its seeds.
- **`--from` / `--to` / `--set STEP.OPTION=VALUE`:** start or stop at a step (`walksat`, `connect`, `render`, or a number) and override options, e.g. `walksat.n=40`. Steps after a changed option aren't compared with the recorded grids.

The renderer records no hash of the image it writes, so a re-render is checked only through the grid it started from.

## Cycles and loops

A **cycle** is one closed ring of live cells in the base tile. `hex_walksat.py` prints how many there are and their sizes.

A **loop** is what you see in the full hexagon, and a single cycle can show up in two ways:

- as **three separate copies**, one per rotated rhombus; or
- as **one connected loop three times as long**, winding around the hexagon's center.

So one cycle is not always one loop. The hexagon's center is a vertex where three copies of the same cell meet, and that cell is always dead (live is impossible under Rule 1). The tools report "closed loops in the full hexagon", which counts 1 for a connected cycle and 3 for a cycle shown as copies. `hex_connect.py` aims for exactly one loop.

With `--wall-boundary`, the extra outer loop hugging the border is kept separate and fixed, and isn't counted. `hex_render.py --remove-outer-wall` deletes it for the final image.

Without a wall, the hexagon is just a window onto an infinite repeating pattern, so a single connected loop can still be cut into several pieces at the window edge. The loop count is still correct.

## Provenance: what every file remembers

Each program appends a record of what it did, and the chain is carried forward from file to file:

- In a **grid text file**, records are stored after the grid as `# meta {json}` lines.
- In a **PNG**, they're stored in the text chunks `hex_pipeline` (full records) and `hex_pipeline_commands` (just the commands, one per line).

A record holds the command, the full parsed options, a timestamp, the seed (generated if you didn't give one), a `reproduce` command with that seed filled in, key results, and a SHA-256 hash of the grid it produced. The next program compares the grid it was given with the previous record's hash and flags a grid that was edited in between.

To read the commands back out of a PNG:

```
python -c "from PIL import Image; print(Image.open('hex_render_1.png').text['hex_pipeline_commands'])"
```

Notes:

- Skipping a step is fine; the chain is only as long as the pipeline you actually ran.
- Runs started from Python record an equivalent command line that shows only the options that differ from their defaults. The complete options are always in the record, and `hex_regen.py` uses those.
- A recorded command repeats a run only if the code hasn't changed since.

## Using it from Python

Each program's work is available as a function taking the command-line options as keyword arguments (`--wall-boundary` becomes `wall_boundary=True`). They write the same files and records as the command line and return what they wrote.

```python
from hex_walksat import generate
from hex_connect import connect_file
from hex_render import render_file

solved = generate(n=36, max_iters=1_000_000, wall_boundary=True, full_hex=True)
if not solved.solved:
    raise SystemExit("the solver did not finish")

joined = connect_file(solved.grid_path, wall_boundary=True)
if not joined.single_connected_loop:
    raise SystemExit("could not reach a single connected loop")

image = render_file(joined.grid_path, mode="heightmap", shape="hexagon",
                    profile="rounded", thickness=1.1, corner_radius=0.4,
                    floor=0, remove_outer_wall=True)
print(image.image_path)
```

This is `example_sequence.py`. The regenerate script is also importable: `regenerate(...)` redoes a recorded run, and `variants(...)` makes a batch.

- Unknown options raise `TypeError`; unusable input or option combinations raise `hex_common.UsageError`. (On the command line these become ordinary error messages.)
- Each function takes `echo=` (default `print`); pass `echo=lambda *a, **k: None` to silence it.
- `generate()` uses the command line's defaults, so repairs are on. The lower-level `solve()` function leaves repairs off unless you ask for them.

## File format

The grid file is plain text: the first line is `n`, followed by `n` lines of `n` characters, `1` for live and `0` for dead, row by row. Anything after those `n + 1` lines (the `# meta` provenance lines) is ignored by readers of the format.

## Verification and limits

Because this software was written by an AI, here is what was and wasn't checked.

**Checked, with ad-hoc scripts during development (not shipped):**

- The solver's neighborhood repair was verified to be exact: its predicted change in violations matched the real change on every test instance, and all search variants agree on the optimum whenever the search completes.
- Cycle detection was checked against an independent flood fill of the plane, and the solver's and renderer's copies agree.
- The test for "connected loop versus three copies" was checked against a slow geometric walk on 93 cycles, with no disagreements.
- The command-line behavior was compared before and after each refactor (outputs, grid files, metadata, PNG pixels, error messages), and recorded commands were replayed in a shell to confirm they reproduce identical grids and images.
- `hex_connect.py` reached a single connected loop on every grid in its final test batches (n from 14 to 30, with and without a wall), and in one recorded n=36 run.

**Not checked, or known limits:**

- There is no automated test suite in this repository.
- **n=36 and above** has barely been tested beyond that one recorded run, and batch generation was only tested at small sizes (n=16). Larger sizes take longer.
- `hex_connect.py` does not guarantee a single loop; it can end stuck. It does not support `--allow-branch` grids.
- `hex_render.py --radius` above 1 was not tested (see above).
- Reproducing a run from its recorded command depends on the code being unchanged, and a connect run that used `--time-limit` depends on machine speed.
- Rendering and path-quality judgments (does it look like coral?) were made by the owner by eye; nothing here measures them.

**Rejected idea, for the record:** a rule forbidding even two consecutive dead neighbors was tried and shown to be impossible together with Rule 1 (exhaustive search at n=3, 4 and 5 found no solutions).

## Background

The solver is in the same family as WalkSAT (Selman, Kautz and Cohen) and Paul Callahan's Stabilizer Algorithm for Game of Life still lifes. The look aimed for is that of reaction-diffusion (Gray-Scott) patterns and brain coral.

## License

No license has been specified yet.
