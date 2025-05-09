# Environment parameters
ENV_WIDTH = 500                # Width of the environment
ENV_HEIGHT = 500               # Height of the environment
CELL_RADIUS = 5                # Radius of each cell
GRID_WIDTH = 2                 # Width of the vector field grid
GRID_HEIGHT = 2                # Height of the vector field grid
NUM_CELLS = 50                # Number of cells in the environment
TIMESTEPS = 500                # Number of timesteps to run each simulation

# Parameters for evolutionary strategies (1+1 ES or GA)
NUM_GENERATIONS = 50           # Number of generations for GA/ES
EPOCHS = 1                # Number of epochs for fitness evaluation

# Vector field dimensions
OUTPUT_DIM = (GRID_WIDTH, GRID_HEIGHT, 2)  # Dimensions of the vector field (x, y components)

USE_ZAPGPT = False         # Toggle between baseline and LLM pipeline
FITNESS_GOAL = "cluster"  # options: "cluster", "scatter", "two_groups"
SEED = 2025                  # Reproducibility
MUTATION_SIGMA = 0.1       # For (1+1) ES
SAVE_TRAJECTORY = False     # Log positions for visual debugging

METRIC = "ollama" #or "cosine"
