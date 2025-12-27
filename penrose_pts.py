import sys
import math
import numpy as np

def matrix(element_00, element_01, element_10, element_11, translation_x=0, translation_y=0):
  return np.array([[element_00, element_01, translation_x],
                   [element_10, element_11, translation_y],
                   [0, 0, 1]])
COS36 = math.cos(math.radians(36))
SIN36 = math.sin(math.radians(36))
COS18 = math.cos(math.radians(18))
SIN18 = math.sin(math.radians(18))

TRANSFORM = {
  'kki': matrix(SIN18, -COS18, -COS18, -SIN18),
  'kks': matrix(-COS36, SIN36, SIN36, COS36, COS18/SIN18 * SIN36, -SIN36),
  'kkl': matrix(1, 0, 0, -1),
  'ddl': matrix(1, 0, 0, -1),
  'ddi': matrix(SIN18, -COS18, -COS18, -SIN18),
  'kds': matrix(-COS36, SIN36, -SIN36, -COS36, 2 * COS36, 0),
  'dks': np.linalg.inv(matrix(-COS36, SIN36, -SIN36, -COS36, 2 * COS36, 0)),
  'kdl': matrix(-1, 0, 0, -1, 1, 0),
  'dkl': matrix(-1, 0, 0, -1, 1, 0)
}

CENTER = np.array([[0.0, 0.0, 1.0]]).T
BASE = np.array([[1.0, 0.0, 1.0]]).T
KITE_HALF_PEAK = np.array([[COS36, -SIN36, 1]]).T
DART_HALF_PEAK = np.array([[0.5, -0.5 * SIN36 / COS36, 1]]).T

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
  <g id="layer1" transform="scale(10)">'''

SVG_TAIL = '''</g></svg>'''

COLORS = {
  ('k', True): 'acd4e2',
  ('k', False): '748f98',
  ('d', True): '833535',
  ('d', False): 'e28787'
}

PHI_INV = (math.sqrt(5) - 1) / 2
SPLIT_POINT = np.array([[KITE_HALF_PEAK[0][0] * PHI_INV,
                         KITE_HALF_PEAK[1][0] * PHI_INV,
                         1]]).T
SEEN = set()

def tile(transformation, node):
    # 1. Start with the Origin (0,0,1) which is a Black dot (Rhomb vertex)
    relative_points = [np.array([[0, 0, 1]]).T]
    relative_points.append(np.array([[1, 0, 1]]).T)
    is_kite = node[-1] == 'k'
    
    # 2. Add the Red dot ONLY if it's a Kite
    if is_kite:
        relative_points.append(KITE_HALF_PEAK)
        relative_points.append(SPLIT_POINT)
    else:
        relative_points.append(DART_HALF_PEAK)
    
    # 3. Transform and Render
    points_matrix = np.hstack(relative_points)
    global_points = transformation @ points_matrix
    
    dots_svg = ""
    for i in range(global_points.shape[1]):
        gx, gy = global_points[0, i], global_points[1, i]
        gx = round(gx, 8)
        gy = round(gy, 8)
        if not (gx, gy) in SEEN:
          SEEN.add((gx, gy))
          color = "black" if i <= 2 else "red"
          dots_svg += '<circle cx="%f" cy="%f" r="0.04" style="fill:%s"/>\n' % (gx, gy, color)
        
    return dots_svg

def traverse(parent, node, side, adjacency, seen, transformation):
  if node not in seen:
    if parent:
      transformation = transformation @ TRANSFORM[parent[-1] + node[-1] + side]
    center = np.round(0.0001 + transformation @ CENTER, decimals = 2)
    base = np.round(0.0001 + transformation @ BASE, decimals = 2)
    peak = np.round(0.0001 + transformation @ 
                       (KITE_HALF_PEAK if node[-1] == 'k' else DART_HALF_PEAK), decimals = 2) 
    print(tile(transformation, node))
    seen.add(node)
    sides = adjacency[node]
    for k in sorted(sides):
      traverse(node, sides[k], k, adjacency, seen, transformation)

adjacency = {} 
for line in sys.stdin:
  (node, neighbor, side) = line.split()
  neighbors = adjacency.get(node, {})
  neighbors[side] = neighbor
  adjacency[node] = neighbors
  neighbors = adjacency.get(neighbor, {})
  neighbors[side] = node
  adjacency[neighbor] = neighbors

triangles = sorted(adjacency.keys())

print(SVG_HEAD)
traverse(None, triangles[0], None, adjacency, set(), matrix(1, 0, 0, 1))
print(SVG_TAIL)

