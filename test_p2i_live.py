import os
import sys
import torch
import pygame
import numpy as np
import random
import re
from sentence_transformers import SentenceTransformer

from cnn_p2i import CNN_P2I
from environment import CellEnvironment
from config import *

# === Device setup ===
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load P2I model ===
p2i = CNN_P2I(output_dim=(GRID_HEIGHT, GRID_WIDTH, 2)).to(device)

# setting random seed
Seed_number = 2025

def set_seed(seed=Seed_number):
    np.random.seed(Seed_number)
    random.seed(Seed_number)
    torch.manual_seed(Seed_number)

input_prompt = "form a cluster"
safe_prompt = re.sub(r'\W+', '_', input_prompt.strip().lower())

weights_path = f"best_weights.pt"
if not os.path.exists(weights_path):
    print(f"No pretrained weights found at: {weights_path}")
    sys.exit(1)

weights = torch.load(weights_path, map_location=device)
p2i.set_weights(weights)  # Custom setter function
p2i.eval()
print(f"✅ Loaded weights from: {weights_path}")

# === Load Text Encoder ===
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# === Pygame setup ===
pygame.init()
screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))
font = pygame.font.SysFont(None, 14)
clock = pygame.time.Clock()

# === Prepare fixed initial cell positions ===
def generate_fixed_positions(num_cells, width, height, radius, seed=42):
    rng = np.random.default_rng(seed)
    positions = []
    for _ in range(num_cells):
        x = rng.uniform(radius, width - radius)
        y = rng.uniform(radius, height - radius)
        positions.append((x, y))
    return np.array(positions)

fixed_positions = generate_fixed_positions(NUM_CELLS, ENV_WIDTH, ENV_HEIGHT, CELL_RADIUS)

# === Draw Functions ===
HIGHLIGHT_CELL_INDEX = 0  # Pick the same cell across runs

def draw_cells(env, step_count, prompt):
    screen.fill((0, 0, 0))

    # Draw faded trajectories for all cells
    for idx, cell in enumerate(env.cells):
        recent_trajectory = cell.trajectory  # last 50 steps
        points = [(int(pos[0]), int(pos[1])) for idx2, pos in enumerate(recent_trajectory) if idx2 % 5 == 0]
        if len(points) > 1:
            color = (100, 100, 255) if idx != HIGHLIGHT_CELL_INDEX else (0, 255, 0)  # Green if highlight
            pygame.draw.lines(screen, color, False, points, 2 if idx == HIGHLIGHT_CELL_INDEX else 1)

    # Draw all current cells
    for cell in env.cells:
        x, y = cell.position
        pygame.draw.circle(screen, (255, 0, 0), (int(x), int(y)), int(cell.radius))

    # Draw prompt and step
    prompt_text = font.render(f"Prompt: {prompt}", True, (255, 255, 255))
    screen.blit(prompt_text, (10, 10))
    step_text = font.render(f"Step: {step_count}", True, (255, 255, 255))
    screen.blit(step_text, (10, 30))

    pygame.display.flip()

# === Main Simulation Function ===
import os

def run_simulation(prompt):
    # Update window title
    pygame.display.set_caption(f"Testing: {prompt}")

    # Re-set random seed before environment creation
    set_seed(Seed_number)

    # Encode prompt into vector field
    embedding = text_encoder.encode([prompt], convert_to_tensor=True).to(device).float()
    with torch.no_grad():
        vf = p2i(embedding).squeeze(0).cpu().numpy()

    # Setup environment
    env = CellEnvironment(
        num_cells=NUM_CELLS,
        width=ENV_WIDTH,
        height=ENV_HEIGHT,
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        cell_radius=CELL_RADIUS,
        #positions=fixed_positions  # optional if you prefer manual positions
    )
    env.vector_field = vf

    # Prepare save directory
    safe_prompt = re.sub(r'\W+', '_', prompt.strip().lower())
    save_dir = f"saved_frames/{safe_prompt}"
    os.makedirs(save_dir, exist_ok=True)

    steps = 0
    running = True
    save_first_frame = True

    while running and steps < 500:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        env.update(dt=1.2)
        if steps == 0:
            draw_cells(env, steps, prompt="")
        else:
            draw_cells(env, steps, prompt)

        # Save first and last frame
        if steps == 0 and save_first_frame:
            pygame.image.save(screen, os.path.join(save_dir, "frame_start.png"))
            save_first_frame = False
        elif steps == 499:
            pygame.image.save(screen, os.path.join(save_dir, "frame_end.png"))

        clock.tick(30)
        steps += 1

    pygame.time.wait(1000)  # Wait 1 second after sim ends

# === Main Loop ===
if __name__ == "__main__":
    while True:
        user_input = input("\nPrompt (or type 'exit'): ").strip()
        if user_input.lower() == "exit":
            pygame.quit()
            break
        run_simulation(user_input)
