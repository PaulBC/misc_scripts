import json
import sys
from PIL import Image
def parse_json(s):
  return json.loads(s.rstrip('\x00'))
for file_name in sys.argv[1:]:
  # Load the image
  img = Image.open(file_name)
  # Get basic metadata
  metadata = {key: parse_json(value) for (key, value) in img.info.items()}
  print(file_name)
  print(json.dumps(metadata, indent=2))
  print()
