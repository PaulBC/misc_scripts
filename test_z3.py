from z3 import *
import sys

# Example Parameters
n = 5  # Number of line segments
m = 3  # At least m segments must satisfy the condition
C = float(sys.argv[1])  # Threshold value for y
slopes = [3, -2, 0.2, 3.5, -2]  # Slopes (m_i)
intercepts = [0, 4, 1.5, -1, 3]  # Intercepts (b_i)

# Define the real variable x (0 <= x <= 1)
x = Real('x')
constraints = [x >= 0, x <= 1]  # x is within [0, 1]

# Define the line segments and the condition y_i(x) > C
ys = [slopes[i] * x + intercepts[i] for i in range(n)]
conditions = [y > C for y in ys]

# At least m conditions must be true
bool_vars = [Bool(f'cond_{i}') for i in range(n)]
for i in range(n):
    constraints.append(Implies(bool_vars[i], conditions[i]))  # Bool var implies condition
constraints.append(Sum([If(b, 1, 0) for b in bool_vars]) >= m)  # At least m true

# Solve the problem
solver = Solver()
solver.add(constraints)

if solver.check() == sat:
    model = solver.model()
    print("Solution found!")
    print(f"x = {model[x]}")
    for i in range(n):
        print(f"Line {i+1}: y = {slopes[i]}*x + {intercepts[i]} > {C} is {'satisfied' if model[bool_vars[i]] else 'not satisfied'}")
else:
    print("No solution found.")