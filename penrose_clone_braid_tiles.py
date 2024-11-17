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
  <g
     id="kite"
     transform="translate(-0.00338711,-0.00454242)">
    <path
       transform="translate(0.00338711,0.00454242)"
       style="font-variation-settings:normal;vector-effect:none;fill:none;fill-opacity:1;stroke:#bfbfbf;stroke-width:0.005;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1;stop-color:#000000"
       d="M 1.0004405,1.0124703e-4 0.80945815,-0.58768377 0.30945779,-0.95095488 4.4095622e-4,1.0119468e-4 Z"
       id="path905-9-5-71-2-0-1-3-5-9-1-8-3-7-3-3-7-1-5"
       sodipodi:nodetypes="ccccc" />
    <path
       id="path9938-6-9-5-6-5"
       style="color:#000000;font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:medium;line-height:normal;font-family:sans-serif;font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-alternates:normal;font-variant-east-asian:normal;font-feature-settings:normal;font-variation-settings:normal;text-indent:0;text-align:start;text-decoration:none;text-decoration-line:none;text-decoration-style:solid;text-decoration-color:#000000;letter-spacing:normal;word-spacing:normal;text-transform:none;writing-mode:lr-tb;direction:ltr;text-orientation:mixed;dominant-baseline:auto;baseline-shift:baseline;text-anchor:start;white-space:normal;shape-padding:0;shape-margin:0;inline-size:0;clip-rule:nonzero;display:inline;overflow:visible;visibility:visible;isolation:auto;mix-blend-mode:normal;color-interpolation:sRGB;color-interpolation-filters:linearRGB;solid-color:#000000;solid-opacity:1;vector-effect:none;fill:#dc3030;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:0.005;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1;color-rendering:auto;image-rendering:auto;shape-rendering:auto;text-rendering:auto;enable-background:accumulate;stop-color:#000000"
       d="m 0.69645211,0.00511419 0.15001655,-2.654e-5 c 1.7e-7,-0.04152897 -0.003204,-0.0824856 -0.009121,-0.12270563 l -0.14510767,0.04715464 c 0.002715,0.02489864 0.004212,0.05009388 0.004212,0.07557753 z m -0.054467,-0.26954434 0.1428596,-0.0464313 C 0.69327578,-0.53691798 0.50576569,-0.71810811 0.2639719,-0.79667162 l -0.0463538,0.14265276 c 0.19569178,0.0635841 0.34784656,0.20824356 0.42436701,0.38958871 z" />
    <path
       id="path9946-9-9-1-3-1"
       style="color:#000000;font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:medium;line-height:normal;font-family:sans-serif;font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-alternates:normal;font-variant-east-asian:normal;font-feature-settings:normal;font-variation-settings:normal;text-indent:0;text-align:start;text-decoration:none;text-decoration-line:none;text-decoration-style:solid;text-decoration-color:#000000;letter-spacing:normal;word-spacing:normal;text-transform:none;writing-mode:lr-tb;direction:ltr;text-orientation:mixed;dominant-baseline:auto;baseline-shift:baseline;text-anchor:start;white-space:normal;shape-padding:0;shape-margin:0;inline-size:0;clip-rule:nonzero;display:inline;overflow:visible;visibility:visible;isolation:auto;mix-blend-mode:normal;color-interpolation:sRGB;color-interpolation-filters:linearRGB;solid-color:#000000;solid-opacity:1;vector-effect:none;fill:#dc3030;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:0.005;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1;color-rendering:auto;image-rendering:auto;shape-rendering:auto;text-rendering:auto;enable-background:accumulate;stop-color:#000000"
       d="m 0.47127207,-0.04046444 c 0.0307498,1.4158e-4 0.0617709,-0.0044903 0.0920875,-0.01434027 L 0.94382706,-0.17844058 0.8974477,-0.32111919 0.51708322,-0.19748328 c -0.0600258,0.0195035 -0.12511435,-0.001675 -0.16221249,-0.0527358 -0.0370971,-0.0510604 -0.0370043,-0.11930044 1.061e-4,-0.17040299 l -1.0521e-4,1.9e-7 0.0682646,-0.0939736 c -0.0386799,-0.0312432 -0.0810735,-0.0584451 -0.1267365,-0.0806928 l -0.0628127,0.0864805 c -0.0749204,0.10317566 -0.0749461,0.24362075 2.3e-7,0.34677476 0.0562093,0.07736562 0.14543654,0.12114591 0.23768583,0.12156891 z m 0.0693495,-0.63587864 0.049403,-0.0680061 -0.12138817,-0.0881084 -0.0534077,0.0735355 c 0.0444598,0.0239864 0.0863234,0.0517093 0.12539287,0.082579 z" />
  </g>
  <g
     id="dart"
     transform="matrix(0.05,0,0,0.05,-2.3863411,-9.0735175)">
    <g
       id="g19"
       transform="translate(-6.2809906,-0.89816029)">
      <path
         style="font-variation-settings:normal;vector-effect:none;fill:none;fill-opacity:1;stroke:#bfbfbf;stroke-width:0.1;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1;stop-color:#000000"
         d="m 60.18815,163.34738 3.819653,11.7557 10.000001,7.26544 -19.999991,-1e-5 6.180337,-19.02113"
         id="path903-5-2-4-2-5-2-4-5-7-1-7-1-0-7-9-2"
         sodipodi:nodetypes="ccccc" />
      <path
         style="color:#000000;font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:medium;line-height:normal;font-family:sans-serif;font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-alternates:normal;font-variant-east-asian:normal;font-feature-settings:normal;font-variation-settings:normal;text-indent:0;text-align:start;text-decoration:none;text-decoration-line:none;text-decoration-style:solid;text-decoration-color:#000000;letter-spacing:normal;word-spacing:normal;text-transform:none;writing-mode:lr-tb;direction:ltr;text-orientation:mixed;dominant-baseline:auto;baseline-shift:baseline;text-anchor:start;white-space:normal;shape-padding:0;shape-margin:0;inline-size:0;clip-rule:nonzero;display:inline;overflow:visible;visibility:visible;opacity:1;isolation:auto;mix-blend-mode:normal;color-interpolation:sRGB;color-interpolation-filters:linearRGB;solid-color:#000000;solid-opacity:1;vector-effect:none;fill:#dc3030;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:0.1;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1;color-rendering:auto;image-rendering:auto;shape-rendering:auto;text-rendering:auto;enable-background:accumulate;stop-color:#000000;stop-opacity:1"
         d="m 61.888672,168.57812 c -2.820845,0.91655 -4.740234,3.55939 -4.740234,6.5254 v 7.26562 h 3 v -7.26562 c 0,-1.67818 1.069978,-3.1533 2.666015,-3.67188 z"
         id="path9859-8" />
      <path
         style="color:#000000;font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:medium;line-height:normal;font-family:sans-serif;font-variant-ligatures:normal;font-variant-position:normal;font-variant-caps:normal;font-variant-numeric:normal;font-variant-alternates:normal;font-variant-east-asian:normal;font-feature-settings:normal;font-variation-settings:normal;text-indent:0;text-align:start;text-decoration:none;text-decoration-line:none;text-decoration-style:solid;text-decoration-color:#000000;letter-spacing:normal;word-spacing:normal;text-transform:none;writing-mode:lr-tb;direction:ltr;text-orientation:mixed;dominant-baseline:auto;baseline-shift:baseline;text-anchor:start;white-space:normal;shape-padding:0;shape-margin:0;inline-size:0;clip-rule:nonzero;display:inline;overflow:visible;visibility:visible;opacity:1;isolation:auto;mix-blend-mode:normal;color-interpolation:sRGB;color-interpolation-filters:linearRGB;solid-color:#000000;solid-opacity:1;vector-effect:none;fill:#dc3030;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:0.1;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1;color-rendering:auto;image-rendering:auto;shape-rendering:auto;text-rendering:auto;enable-background:accumulate;stop-color:#000000;stop-opacity:1"
         d="m 60.681627,181.23495 1.207045,0.392 c 2.82082,0.91652 5.92655,-0.0926 7.669922,-2.49218 l -2.427735,-1.76368 c -0.9864,1.3577 -2.72035,1.92093 -4.316406,1.40235 l -2.147979,-0.69758 z"
         id="path9863-9"
         sodipodi:nodetypes="ccccccc" />
      <path
         style="font-variation-settings:normal;opacity:1;vector-effect:none;fill:#dc3030;fill-opacity:1;stroke:none;stroke-width:0.1;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1;stop-color:#000000;stop-opacity:1"
         d="m 55.90501,176.52953 -0.926494,2.85328 1.669921,0.54232 v -3.15417 l -0.743427,-0.24143"
         id="path9869-9" />
    </g>
  </g>
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
