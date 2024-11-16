import sys
import math
import numpy
import random

ROTATION = 36 *  math.pi / 180
RATIO = 2 / (math.sqrt(5) + 1)

FLIP_Y = numpy.matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
ROTATE = numpy.matrix([
  [math.cos(ROTATION), -math.sin(ROTATION), 0],
  [math.sin(ROTATION), math.cos(ROTATION), 0],
  [0, 0, 1]])
SCALE = numpy.matrix([[RATIO, 0, 0], [0, RATIO, 0], [0, 0, 1]])
SCALE_INV = numpy.linalg.inv(numpy.matrix([[RATIO, 0, 0], [0, RATIO, 0], [0, 0, 1]]))

# To give tiling a layered appearance (distance is relative to side length)
SHIFT_DISTANCE = 0.12
SHIFT_ANGLE = 43 * math.pi / 180

def translate(x, y):
  return numpy.matrix([[1, 0, x], [0, 1, y], [0, 0, 1]])

def to_components(transformation):
  xmult = transformation[0, 0]
  ymult = transformation[1, 0]
  theta = int(round(math.atan2(ymult, xmult) * 180 / math.pi))
  scale = math.sqrt(xmult * xmult + ymult * ymult)
  return(transformation[0, 2], transformation[1, 2], theta, scale)

TILE_CENTER = {'dart': translate(0.25, -0.18), 'kite': translate(0.405, -0.295)}

def distance(tile):
  shifted = tile[1] * TILE_CENTER[tile[0]]
  return math.sqrt(shifted[0, 2] ** 2 + shifted[1, 2] ** 2)

HALF_KITE_TO_HALF_KITE_1 = translate(1, 0) * ROTATE ** 7 * SCALE
HALF_KITE_TO_HALF_KITE_2 = translate(1, 0) * ROTATE ** 5 * FLIP_Y * SCALE
HALF_KITE_TO_HALF_DART = ROTATE ** 9 * FLIP_Y * SCALE

HALF_DART_TO_HALF_KITE = SCALE
HALF_DART_TO_HALF_DART = translate(1, 0) * ROTATE ** 6 * SCALE

SVG_HEAD = '''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   width="420mm"
   height="297mm"
   viewBox="0 0 420 297"
   version="1.1"
   id="svg8">
  <path style="fill:#7693b8;fill-opacity:1;stroke:#ff0000;stroke-width:0.02;stroke-linejoin:miter;stroke-opacity:1"
      d="M 0.069,-0.05 0.845,-0.05 0.458,-0.33 0.31,-0.79 z"
      id="dart"/>
  <path style="fill:#a9d0de;fill-opacity:1;stroke:#ff0000;stroke-width:0.02;stroke-linejoin:miter;stroke-opacity:1"
      d="M 0.069,-0.05 0.93,-0.05 0.77,-0.56 0.34,-0.87 z"
      id="kite"/>
  <g id="layer1" transform="scale(10)">'''

SVG_TAIL = '''</g></svg>'''

start_tile = sys.argv[1]
depth = int(sys.argv[2])
patch_center = (0, 0)
tile_limit = 100000
if len(sys.argv) > 5:
  patch_center = (float(sys.argv[3]), float(sys.argv[4]))
  tile_limit = int(sys.argv[5])

# set up starting tiles
rotation = translate(-patch_center[0], -patch_center[1])
tiles = []
for i in range(5):
  tiles.append((start_tile, rotation, '%d+_' % i))
  tiles.append((start_tile, rotation * FLIP_Y, '%d-_' % i))
  rotation = rotation * ROTATE * ROTATE

# apply deflation rules
for rep in range(depth):
  new_tiles = []
  for kind, transformation, rules in tiles:
    if kind == 'kite':
      new_tiles.append(('kite', transformation * HALF_KITE_TO_HALF_KITE_1, rules + 'k1'))
      new_tiles.append(('kite', transformation * HALF_KITE_TO_HALF_KITE_2, rules + 'k2'))
      new_tiles.append(('dart', transformation * HALF_KITE_TO_HALF_DART, rules + 'k3'))
    else:
      new_tiles.append(('kite', transformation * HALF_DART_TO_HALF_KITE, rules + 'd1'))
      new_tiles.append(('dart', transformation * HALF_DART_TO_HALF_DART, rules + 'd2'))

  # keep only closest tiles but scale the size back up 
  new_tiles.sort(key = distance)
  tiles = [(tile[0], SCALE_INV * tile[1], tile[2]) for tile in new_tiles[:2 * tile_limit]]

# eliminate flipped tile halves so we draw only whole non-flipped tiles
not_flipped = [tile for tile in tiles if numpy.linalg.det(tile[1]) > 0]

kites = [tile for tile in not_flipped if tile[0] == 'kite']
darts = [tile for tile in not_flipped if tile[0] == 'dart']

print(SVG_HEAD)

dart_colors = ['7693b8', '6d87a9', '647c9b', '5b718e', '536680', '4f627a', '576c88', '607795', '6982a3', '718db0']
kite_colors = ['a9d0de', 'afd8e7', 'b7e2f1', 'bfebfb', 'c6f5ff', 'cbfaff', 'c3f0ff', 'bbe7f6', 'b4deec', 'acd4e2']
dart_colors = ['7693b8'] * 10 
kite_colors = ['a9d0de'] * 10

palette = {}

# darts in shifted positions
shift = SHIFT_DISTANCE * RATIO ** depth
print('''    <g transform="translate(%f,%f)">''' % (shift * math.cos(SHIFT_ANGLE), shift * math.sin(SHIFT_ANGLE)))
for name, mat, rules in darts:
      components = to_components(mat)
      print('''      <use transform="translate(%f,%f) rotate(%d) scale(%f)"
        href="#dart" 
        id="dart_%s"/>''' % (components + (rules,)))
print('    </g>')

# kites
print('    <g>')
for name, mat, rules in kites:
      components = to_components(mat)
      print('''      <use transform="translate(%f,%f) rotate(%d) scale(%f)"
        href="#kite"
        id="kite_%s"/>''' % (components + (rules,)))
print('    </g>')

print(SVG_TAIL)
