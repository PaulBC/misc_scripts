import sys

NEW_EDGE = {
  'd': [('dd', 'dk', 's')],
  'k': [('kdk', 'kk', 'i'), ('kdd', 'kdk', 's')]}

SPLIT_EDGE = {
  ('k', 'k', 'i'): [('kdd', 'kdd', 'l'), ('kk', 'kk', 's')],
  ('k', 'k', 'l'): [('kdd', 'kdd', 'i'), ('kdk', 'kdk', 'l')],          
  ('k', 'k', 's'): [('kk', 'kk', 'l')],
  ('d', 'd', 'i'): [('dk', 'dk', 'i')],
  ('d', 'd', 'l'): [('dd', 'dd', 'i'), ('dk', 'dk', 'l')],
  ('d', 'k', 's'): [('dd', 'kk', 'l')],
  ('d', 'k', 'l'): [('dd', 'kdd', 'i'), ('dk', 'kdk', 'l')]
}

def new_edge(triangle):
  return [(triangle[:-1] + t1, triangle[:-1] + t2, side)
          for t1, t2, side in NEW_EDGE[triangle[-1]]] 

def split_edge(edge):
  triangle1, triangle2, side = edge
  return [(triangle1[:-1] + t1, triangle2[:-1] + t2, split_side)
           for t1, t2, split_side in SPLIT_EDGE[(triangle1[-1], triangle2[-1], side)]] 

depth = int(sys.argv[1])

edges = [('0+k', '0-k', 'i'),
         ('0-k', '1+k', 'l'),
         ('1+k', '1-k', 'i'),
         ('1-k', '2+k', 'l'),
         ('2+k', '2-k', 'i'),
         ('2-k', '3+k', 'l'),
         ('3+k', '3-k', 'i'),
         ('3-k', '4+k', 'l'),
         ('4+k', '4-k', 'i'),
         ('4-k', '0+k', 'l')] 

triangles = sorted(set(triangle for edge in edges for triangle in edge[:2]))

for i in range(1, depth):
  new_edges = [edge for triangle in triangles for edge in new_edge(triangle)]
  split_edges = [child_edge for edge in edges for child_edge in split_edge(edge)] 
  edges = new_edges + split_edges
  triangles = sorted(set(triangle for edge in edges for triangle in edge[:2]))

for node, neighbor, side in sorted(edges):
  print(node, neighbor, side)
