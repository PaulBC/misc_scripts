def parens(n):
  for res in parens_recur(n * 2, 0, []):
    yield res

def parens_recur(n, depth, sofar):
  if n == len(sofar):
    yield "".join(sofar)
  else:
    if len(sofar) + depth < n:
      sofar.append("(")
      for res in parens_recur(n, depth + 1, sofar):
        yield res
      sofar.pop()
    if depth > 0:
      sofar.append(")")
      for res in parens_recur(n, depth - 1, sofar):
        yield res
      sofar.pop()

def links(paren_string):
  stack = []
  for i in range(len(paren_string)):
    if paren_string[i] == '(':
      stack.append(i)
    else:
      yield (stack.pop(), i)

def perm(links):
  links = list(links)
  ix = [0] * len(links) * 2
  for i, j in links:
    ix[i], ix[j] = j, i
  return ix

def rotate(links):
  return [len(links) - i -1 for i in links[::-1]]

def loops(uplinks, downlinks):
  seen = [False for x in uplinks]
  loops = []
  for i in range(len(uplinks)):
    if not seen[i]:
      loop = []
      j = i
      while not seen[j]:
        seen[j] = True
        loop.append(j)
        j = uplinks[j]
        seen[j] = True
        loop.append(j)
        j = downlinks[j]
      loops.append(loop)
  return loops

def to_curve(loops, scale, x, y):
  yield "-fill black -stroke none -draw 'fill-rule evenodd translate %5.3f %5.3f path \"" % (x, y)
  for loop in loops:
    n = len(loop)
    yield 'M %5.3f 0' % (loop[0] * scale)
    for i in range(n):
      x1 = loop[i] * scale
      x2 = loop[(i + 1) % n] * scale
      sweep = ((0 if x1 > x2 else 1) + i) % 2
      radius = abs(x2 - x1) * 0.5
      yield 'A %5.3f %5.3f 0 0 %5.3f %5.3f 0' % (radius, radius, sweep, x2)
    yield ' Z'
  yield "\"'"

def to_upper(links, scale, x, y):
  edges = [i + (1 - (i % 2) * 2) for i in range(len(links))]
  yield "-fill black -stroke none -draw 'fill-rule evenodd translate %5.3f %5.3f path \"" % (x, y)
  for loop in loops(links, edges):
    n = len(loop)
    yield 'M %5.3f 0' % (loop[0] * scale)
    for i in range(n):
      x1 = loop[i] * scale
      x2 = loop[(i + 1) % n] * scale
      sweep = ((0 if x1 > x2 else 1) + i) % 2
      radius = abs(x2 - x1) * 0.5
      if i % 2 == 0:
        yield 'A %5.3f %5.3f 0 0 %5.3f %5.3f 0' % (radius, radius, sweep, x2)
      else:
        yield 'L %5.3f 0' % x2
    yield ' Z'
  yield "\"'"

n = 5
allix = [perm(links(s)) for s in parens(n)]

'''
for i in range(len(allix)):
  for j in range(len(allix)):
    print("*" if 1 == len(loops(allix[i], rotate(allix[j]))) else ".", end=' ')
  print()
  '''

def nextpos(x, y):
   x += 85
   if x > 1700:
     x = 50
     y += 90
   return x, y

x = 50
y = 100
lines = ["magick -size 1800x1800 canvas:none "]
for i in range(len(allix)):
  lines.extend(to_upper(allix[i], 7, x, y))
  lines.append("-fill none -stroke blue -draw 'translate %5.3f %5.3f rectangle %5.3f 0 %5.3f %5.3f'" %
               (x, y, -5, (2 * n - 1) * 8 + 5, -n * 8 - 5))
  x, y = nextpos(x, y)

x = 50
y += 90

for i in range(len(allix)):
  for j in range(len(allix)):
    path = loops(allix[i], rotate(allix[j]))
    if len(path) == 1:
      lines.extend(to_curve(path, 7, x, y))
      lines.append("-fill none -stroke blue -draw 'translate %5.3f %5.3f rectangle %5.3f 0 %5.3f %5.3f'" %
               (x, y, -5, (2 * n - 1) * 8 + 5, -n * 8 - 5))
      lines.append("-fill none -stroke blue -draw 'translate %5.3f %5.3f rectangle %5.3f 0 %5.3f %5.3f'" %
               (x, y, -5, (2 * n - 1) * 8 + 5, n * 8 - 5))
      x, y = nextpos(x, y)
  lines.append("-fill red -draw 'translate %5.3f %5.3f rectangle 20 -3 26 3'" % (x, y)) 
  x, y = nextpos(x, y)
lines.append("t.png")

print(" \\\n".join(lines))