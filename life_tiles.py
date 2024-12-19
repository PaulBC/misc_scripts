LONELY = {'L0': '0000', 'L1': '0001', 'L2': '0010'}
SURVIVAL = {'S1': '0001', 'S2': '0010', 'S3': '0011', 'S4': '0101'}
CROWDING = {'C1': '01', 'C2': '11', 'C3': '0011'}

def canonical_shift(window_str):
  return min([window_str[i:] + window_str[:i] for i in range(len(window_str))])

def canonical(window_str):
  return min(canonical_shift(window_str), canonical_shift(window_str[::-1])) 

def count_bits(window_str):
  return sum(int(c) for c in window_str)

def fit_pieces(pieces, symbols, layout, accumulated, seen, limit):
  if len(layout) >= 8:
    realized = canonical(layout)
    if realized not in seen and count_bits(realized) <= limit and len(layout) == 8:
      seen.add(realized)
      accumulated.append((realized, symbols))
  else:
    for symbol, piece in pieces.items():
      fit_pieces(pieces, symbols + '+' + symbol, layout + piece, accumulated, seen, limit)
    for symbol, piece in pieces.items():
      fit_pieces(pieces, symbols + '-' + symbol, layout + piece[::-1], accumulated, seen, limit)
  return accumulated

OUTPUT = '''
00000000 +L0+L0
00000001 +L0+L1
00010001 +L1+L1
00001001 +L1+L2
00000011 +L1-L1
00000101 +L1-L2
00010001 +S1+S1
00001001 +S1+S2
00010011 +S1+S3
00010101 +S1+S4
00000011 +S1-S1
00000101 +S1-S2
00000111 +S1-S3
00001011 +S1-S4
00100101 +S2+S4
01010101 +C1+C1+C1+C1
01010111 +C1+C1+C1+C2
00101011 +C1+C1+C1-C1
01011111 +C1+C1+C2+C2
00101111 +C1+C1+C2-C1
01011011 +C1+C1-C1+C2
00101101 +C1+C1-C1-C1
00010111 +C1+C1-C3
01110111 +C1+C2+C1+C2
00110111 +C1+C2+C1-C1
01111111 +C1+C2+C2+C2
00111111 +C1+C2+C2-C1
01101111 +C1+C2-C1+C2
00011111 +C1+C2-C3
00100111 +C1+C3-C1
00110011 +C1-C1+C1-C1
00011011 +C1-C1+C3
11111111 +C2+C2+C2+C2
00001111 +C3-C3
'''

for result in fit_pieces(LONELY, '', '', [], set(), 2):
  print(*result)

for result in fit_pieces(SURVIVAL, '', '', [], set(), 3):
  print(*result)

for result in fit_pieces(CROWDING, '', '', [], set(), 6):
  print(*result)



import sys
from collections import Counter

LONELY = {
  '00000001': {'L0': 1, 'L1B': 1},
  '00000011': {'L1A': 2},
  '00000101': {'L1A': 1, 'L1B': 1},
  '00001001': {'L1A': 1, 'L1B': 1},
  '00010001': {'L1A': 2}
}

SURVIVAL = {
  '00000011': {'S0': 1,'S1A': 2},
  '00000101': {'S0': 1, 'S1A': 1, 'S1B': 1},
  '00000111': {'S1A': 1, 'S2A': 1},
  '00001001': {'S0': 1, 'S1A': 1, 'S1B': 1},
  '00001011': {'S1A': 1, 'S2B': 1},
  '00010001': {'S0': 1, 'S1A': 2},
  '00010011': {'S1A': 1, 'S2A': 1},
  '00010101': {'S1A': 1, 'S2B': 1},
  '00100101': {'S1B': 1, 'S2B': 1}
}

CROWDING = {
  '00001111': {'C3': 2},
  '00010111': {'C1': 2, 'C3': 1},
  '00011011': {'C1': 2, 'C3': 1},
  '00011111': {'C1': 1, 'C2': 1, 'C3': 1},
  '00100111': {'C1': 2, 'C3': 1},
  '00101011': {'C1': 2, 'C3': 1},
  '00101101': {'C1': 4},
  '00101111': {'C1': 1, 'C2': 1, 'C3': 1},
  '00110011': {'C3': 2},
  '00110111': {'C1': 1, 'C2': 1, 'C3': 1},
  '00111111': {'C2': 2, 'C3': 1},
  '01010101': {'C1': 4},
  '01010111': {'C1': 3, 'C2': 1},
  '01011011': {'C1': 3, 'C2': 1},
  '01011111': {'C1': 2, 'C2': 2},
  '01101111': {'C1': 2, 'C2': 2},
  '01110111': {'C1': 2, 'C2': 2}
}

WINDOWS = {
  '0001': {'W1': 1},
  '0011': {'W2A': 1},
  '0101': {'W2B': 1},
  '0111': {'W3': 1},
  '1111': {'W4': 1}
}

BORDERS = {
  '0001': {'B3': 1},
  '0011': {'B2': 1},
  '0101': {'B1': 2},
  '0111': {'B1': 1},
  '1111': {}
}

def rle_to_coordinates(rle):
  lines = rle.strip().split('$')
  coordinates = []
  y = 0
  for line in lines:
    x = 0
    count = ''
    for char in line:
      if char.isdigit():
        count += char
      else:
        count = int(count) if count != '' else 1
        if char == 'o':
          for i in range(count):
            coordinates.append((x + i, y))
        x += count
        count = ''
    y += 1
  return coordinates


def canonical_shift(window_str):
  return min([window_str[i:] + window_str[:i] for i in range(len(window_str))])

def cell(live_cells, i, j):
  return '1' if (i, j) in live_cells else '0'

def to_windows(live_cells):
  windows = {}
  for i, j in live_cells:
    for di in range(-1, 1):
      for dj in range(-1, 1):
        wi, wj = i + di, j + dj
        if (wi, wj) not in windows:
          windows[(wi, wj)] = canonical_shift(''.join([cell(live_cells, ti, tj) 
           for (ti, tj) in [(wi, wj), (wi, wj + 1), (wi + 1, wj + 1), (wi + 1, wj)]]))
  return windows

def all_neighbors(live_cells):
  neighborhoods = {}
  left_right = set(list(live_cells) + [(i, j - 1) for (i, j) in live_cells] + [(i, j + 1) for (i, j) in live_cells] )
  up_down = sorted(set([(i - 1, j) for (i, j) in left_right] + [(i + 1, j) for (i, j) in left_right] + list(left_right)))
  for (ci, cj) in up_down:
    neighborhood = [cell(live_cells, ti, tj) for (ti, tj) in 
     [(ci - 1, cj - 1), (ci - 1, cj), (ci - 1, cj + 1), (ci, cj + 1), (ci + 1, cj + 1), (ci + 1, cj), (ci + 1, cj -1 ), (ci, cj - 1)]]
    neighborhood_string = ''.join(neighborhood)
    value = cell(live_cells, ci, cj)
    canonical = min(canonical_shift(neighborhood_string[::-1]), canonical_shift(neighborhood_string))
    neighbor_count = sum(c == '1' for c in neighborhood)
    pieces = SURVIVAL[canonical] if value == '1' else (LONELY[canonical] if neighbor_count <= 2 else CROWDING[canonical])
    neighborhoods[(ci, cj)] = (value, neighbor_count, canonical, pieces,
                               BORDERS[canonical_shift(''.join(str(x) for x in corner_values(neighborhood_string)))] 
                               if value == '0' else {})
  return neighborhoods

def corner_values(neighborhood):
  return [int(bool(sum(int(neighborhood[(i + di) % 8]) for di in range(3)))) for i in range(1, 9, 2)]

def all_pieces(rle_string):
  live_cells = set(rle_to_coordinates(rle_string)) 
  pieces = Counter()
  for k, v in all_neighbors(live_cells).items():
    pieces += Counter(v[3])
    pieces += Counter(v[4])
  for k, v in to_windows(live_cells).items():
    pieces += Counter(WINDOWS[v])
  return ' '.join('%sx%d' % pair for pair in sorted(pieces.items()))

# Simplified RLE string without header
rle_string = '2o$2o!'
rle_string = 'b2o$bobo$3bo$3b2o!'

if len(sys.argv) > 1:
  rle_string = sys.argv[1]

print(all_pieces(rle_string))

