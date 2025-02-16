import random
import string
import sys

# Parameters
length = 26  # Length of the evolving string
k = 1  # Number of random changes per iteration
s = 50  # Number of samples per generation
show = 100  # How often to print progress

if len(sys.argv) > 1:
    (k, s, show) = (int(tok) for tok in sys.argv[1:])

# Generate a random initial string
def random_string(length):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

# Mutate the string by randomly changing k characters
def mutate(parent, k):
    child = list(parent)
    for _ in range(k):
        idx = random.randint(0, len(parent) - 1)
        child[idx] = random.choice(string.ascii_uppercase)
    return ''.join(child)

def fitness(candidate):
    pair_score = sum(1 for i in range(len(candidate) - 1)
                     if abs(ord(candidate[i]) - ord(candidate[i+1])) == 1)
    diversity_score = len(set(candidate))  # Number of distinct characters
    return pair_score + diversity_score

# Main evolutionary algorithm
def weasel_algorithm(k, s, show):
    candidate = random_string(length)
    generation = 0

    while True:
        generation += 1
        mutants = [mutate(candidate, k) for _ in range(s)]
        candidate = max(mutants, key=fitness)

        if generation % show == 0:
            print(f"Gen {generation}: {candidate}, Fitness: {fitness(candidate)}")
            sys.stdout.flush()

        # Stop if maximum possible fitness is reached (length - 1 consecutive pairs)
        if fitness(candidate) == min(length, 26) + length - 1:
            break

    print(f"Final match in {generation} generations!")
    print(f"Gen {generation}: {candidate}")

# Run the algorithm
weasel_algorithm(k, s, show)
