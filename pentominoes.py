import sys

PENTOMINOES = {
  'w': '.**!**!*!',
  'x': '.*!***!.*!',
  'f': '.**!**!.*!',
  'n': '.***!**!',
  'z': '.**!.*!**!',
  'v': '***!*!*!',
  'p': '***!**!',
  'u': '***!*.*!',
  'l': '****!*!',
  't': '***!.*!.*!',
  'y': '****!.*!',
  'i': '*****!'
}

def to_coordinates(pentomino):
  coordinates = []
  pos = (0, 0)
  for c in pentomino:
    if c == '*':
      coordinates.append(pos) 
   
    if c == '!':
      pos = (pos[0] + 1, 0) 
    else:
      pos = (pos[0], pos[1] + 1)
  return coordinates

def normalize_position(coordinates):
  imin = min(i for i, j in coordinates)
  jmin = min(j for i, j in coordinates)
  return [(i - imin, j - jmin) for i, j in coordinates]

def symmetries(coordinates):
  result = [coordinates]
  result += [[(-i, j) for i, j in coords] for coords in result]
  result += [[(i, -j) for i, j in coords] for coords in result]
  result += [[(j, i) for i, j in coords] for coords in result]
  return sorted(set(tuple(sorted(normalize_position(coords))) for coords in result))

def placements(coordinates, ibound, jbound):
  imax = max(i for i, j in coordinates)
  jmax = max(j for i, j in coordinates)
  for i in range(ibound - imax):
    for j in range(jbound - jmax):
      yield tuple((i + di, j + dj) for di, dj in coordinates)

def name(letter, cells):
  return letter + '_' + '_'.join('%d%d' % (i, j) for i, j in cells)

def coords_for_name(name):
  return name[2:].split('_')

NROWS = 10
NCOLS = 6
TOP_LEFT = '00'
TOP_RIGHT = '0%d' % (NCOLS - 1)
BOTTOM_LEFT = '%d0' % (NROWS - 1)
BOTTOM_RIGHT = '%d%d' % (NROWS - 1, NCOLS - 1)

def coord_to_pentomino(choices):
  res = {}
  for p in choices:
    for coord in p[2:].split('_'):
      res[coord] = p 
  return res

def state_key(pentominoes, cells):
  return (tuple(sorted(pentominoes)), tuple(sorted(cells)))

def is_canonical(choices):
    placement = coord_to_pentomino(choices)
    tl = placement[TOP_LEFT][0]
    tr = placement[TOP_RIGHT][0]
    bl = placement[BOTTOM_LEFT][0]
    br = placement[BOTTOM_RIGHT][0]
    return tl == min(tl, tr, bl, br)

# Note: only try to memoize deadends if a set is provided.
def place_pentominoes(pentominoes, cells, choices, min_coord_to_pentomino, deadends):
  if len(pentominoes) == 0:
    yield choices 
  elif deadends == None or state_key(pentominoes, cells) not in deadends:
    nextcoord = min(cells)
    found = False
    for p in min_coord_to_pentomino[nextcoord]:
      coords = coords_for_name(p)
      symbol = p[0]
      if (symbol in pentominoes
          and len(cells.intersection(coords)) == len(coords)):
        pentominoes.remove(symbol)
        cells.difference_update(coords)
        for res in place_pentominoes(pentominoes, cells,
                                     choices + [p], min_coord_to_pentomino, deadends):
          found = True
          yield res      
        cells.update(coords)
        pentominoes.add(symbol)
    if deadends != None and not found:
      deadends.add(state_key(pentominoes, cells)) 

pentomino_names = []
min_coord_to_pentomino = {}
for k in sorted(PENTOMINOES):
  names = []
  for sym in symmetries(to_coordinates(PENTOMINOES[k])):
    for p in placements(sym, NROWS, NCOLS):
      nm = name(k, p)
      pentomino_names.append(nm)
      min_coord_to_pentomino.setdefault(nm[2:4], []).append(nm)

all_cells = set() 
for p in pentomino_names:
  all_cells.update(coords_for_name(p))

deadends = set() if len(sys.argv) > 1 and sys.argv[1] == 'memoize' else None
ix = 0
for res in place_pentominoes(set(PENTOMINOES), all_cells, [],
                             min_coord_to_pentomino, deadends):
  if is_canonical(res):
    ix += 1
    print(ix, " ".join(res))
