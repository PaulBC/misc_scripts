import numpy as np
from scipy.spatial import Voronoi
import sys

def read_points(filename):
    """Read points from a file in the format: x y (one point per line)"""
    points = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                x, y = map(float, line.split())
                points.append([x, y])
    return np.array(points)

SEEN = set()

def write_vertices(vertices, filename):
    """Write vertices to a file in the format: x y (one vertex per line)"""
    with open(filename, 'w') as f:
        for vertex in vertices:
            (x, y) = (round(vertex[0], 4), round(vertex[1], 4))
            if (x, y) not in SEEN:
                SEEN.add((x, y))
                if abs(vertex[0]) < 100 and abs(vertex[1]) < 100:
                  f.write(f"{vertex[0]:.6f} {vertex[1]:.6f}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python voronoi.py <input_file> [output_file]")
        print("Example: python voronoi.py points.txt vertices.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "vertices.txt"
    
    # Read input points
    points = read_points(input_file)
    print(f"Read {len(points)} points from {input_file}")
    
    # Compute Voronoi diagram
    vor = Voronoi(points)
    print(f"Computed Voronoi diagram with {len(vor.vertices)} vertices")
    
    # Write vertices to output file
    write_vertices(vor.vertices, output_file)
    print(f"Wrote {len(vor.vertices)} vertices to {output_file}")
    
    # Optional: print some statistics
    print(f"\nNumber of regions: {len(vor.regions)}")
    print(f"Number of ridges: {len(vor.ridge_vertices)}")

if __name__ == "__main__":
    main()
