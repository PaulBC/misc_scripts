import sys

print('''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
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
  <g id="layer1" transform="scale(10)">''')

for line in sys.stdin:
  (x, y) = line.split()
  print('<circle cx="%s" cy="%s" r="0.04" style="fill:black"/>' % (x, y))

print('</g></svg>')
