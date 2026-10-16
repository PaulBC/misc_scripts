"""
Run the whole pipeline from Python: solve -> connect -> render.

This is the same code and the same files as typing the three commands, so
every step still writes its provenance record and the final PNG carries the
whole chain. Each function takes the command-line options as keyword
arguments (--wall-boundary becomes wall_boundary=True) and returns what it
wrote, so one step's output file feeds the next.

Keep this file, hex_common.py, hex_walksat.py, hex_connect.py and
hex_render.py in one folder.
"""

from hex_walksat import generate
from hex_connect import connect_file
from hex_render import render_file

solved = generate(n=36, max_iters=1_000_000, wall_boundary=True, full_hex=True)
if not solved.solved:
    raise SystemExit("the solver did not finish; try more iterations or another seed")

joined = connect_file(solved.grid_path, wall_boundary=True)
if not joined.single_connected_loop:
    raise SystemExit("could not reach a single connected loop")

image = render_file(joined.grid_path, mode="heightmap", shape="hexagon",
                    profile="rounded", thickness=1.1, corner_radius=0.4,
                    floor=0, remove_outer_wall=True)
print("final image:", image.image_path)
