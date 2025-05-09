import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import sys
import torch
import pygame
import numpy as np
import random
import re
import speech_recognition as sr
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

# Random seed
Seed_number = 2025

def set_seed(seed=Seed_number):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

input_prompt = "form a cluster"
safe_prompt = re.sub(r'\W+', '_', input_prompt.strip().lower())

weights_path = f"best_weights.pt"
if not os.path.exists(weights_path):
    print(f"No pretrained weights found at: {weights_path}")
    sys.exit(1)

weights = torch.load(weights_path, map_location=device)
p2i.set_weights(weights)
p2i.eval()
print(f"✅ Loaded weights from: {weights_path}")

# === Load Text Encoder ===
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# === Pygame setup ===
pygame.init()
screen = pygame.display.set_mode((ENV_WIDTH, ENV_HEIGHT))
font = pygame.font.SysFont(None, 18)
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

# === Draw Function ===
HIGHLIGHT_CELL_INDEX = 0

def draw_cells(env, step_count, prompt):
    screen.fill((0, 0, 0))
    
    # Draw faded trajectories
    for idx, cell in enumerate(env.cells):
        recent_trajectory = cell.trajectory
        points = [(int(pos[0]), int(pos[1])) for idx2, pos in enumerate(recent_trajectory) if idx2 % 5 == 0]
        if len(points) > 1:
            color = (100, 100, 255) if idx != HIGHLIGHT_CELL_INDEX else (0, 255, 0)
            pygame.draw.lines(screen, color, False, points, 2 if idx == HIGHLIGHT_CELL_INDEX else 1)

    # Draw cells
    for cell in env.cells:
        x, y = cell.position
        pygame.draw.circle(screen, (255, 0, 0), (int(x), int(y)), int(cell.radius))
    
    # Draw prompt and step
    prompt_text = font.render(f"Prompt: {prompt}", True, (255, 255, 255))
    screen.blit(prompt_text, (10, 10))
    step_text = font.render(f"Step: {step_count}", True, (255, 255, 255))
    screen.blit(step_text, (10, 30))
    
    pygame.display.flip()

# === Speech Recognition Function ===
def listen_for_prompt():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Please speak your prompt...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(f" Recognized prompt: {text}")
        return text
    except sr.UnknownValueError:
        print(" Could not understand audio")
        return None
    except sr.RequestError:
        print(" Speech Recognition API error")
        return None

# === Main Simulation Function ===
import time

def draw_prompt_multiline(prompt, font, screen, y_start, line_spacing=10, max_width=None):
    words = prompt.split(' ')
    lines = []
    current_line = ""
    
    # Measure text width and split
    for word in words:
        test_line = current_line + word + " "
        text_surface = font.render(test_line, True, (255, 255, 255))
        if max_width and text_surface.get_width() > max_width:
            lines.append(current_line)
            current_line = word + " "
        else:
            current_line = test_line
    lines.append(current_line)

    # Now draw each line
    for idx, line in enumerate(lines):
        text_surface = font.render(line.strip(), True, (255, 255, 255))
        x_pos = (ENV_WIDTH - text_surface.get_width()) // 2
        y_pos = y_start + idx * (text_surface.get_height() + line_spacing)
        screen.blit(text_surface, (x_pos, y_pos))

def run_simulation(prompt):
    # Update window title
    pygame.display.set_caption(f"Prompt: {prompt}")

    # Reset seed
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
        cell_radius=CELL_RADIUS
    )
    env.vector_field = vf

    # --- Show prompt and countdown ---
    screen.fill((0, 0, 0))
    
    # Font sizes
    prompt_font = pygame.font.SysFont(None, 36)
    countdown_font = pygame.font.SysFont(None, 100)

    # Draw prompt text
    #prompt_text = prompt_font.render(f"The prompt is: '{prompt}'", True, (255, 255, 255))
    # Multiline prompt draw
    draw_prompt_multiline(f"The prompt is: '{prompt}'", prompt_font, screen, ENV_HEIGHT//2 - 100, max_width=ENV_WIDTH - 100)
    
    #screen.blit(prompt_text, (ENV_WIDTH//2 - prompt_text.get_width()//2, ENV_HEIGHT//2 - 100))
    
    pygame.display.flip()
    time.sleep(1.5)

    # Countdown numbers
    for count in [3, 2, 1]:
        screen.fill((0, 0, 0))
        #screen.blit(prompt_text, (ENV_WIDTH//2 - prompt_text.get_width()//2, ENV_HEIGHT//2 - 100))
        draw_prompt_multiline(f"The prompt is: '{prompt}'", prompt_font, screen, ENV_HEIGHT//2 - 100, max_width=ENV_WIDTH - 100)

        countdown_text = countdown_font.render(str(count), True, (255, 255, 0))
        screen.blit(countdown_text, (ENV_WIDTH//2 - countdown_text.get_width()//2, ENV_HEIGHT//2))
        
        pygame.display.flip()
        time.sleep(1)

    # Short pause
    time.sleep(0.5)

    # --- Now run the simulation ---
    steps = 0
    running = True
    while running and steps < 500:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        env.update(dt=1.2)
        draw_cells(env, steps, prompt)
        clock.tick(30)
        steps += 1

    pygame.time.wait(1000)

# === Main Loop ===
if __name__ == "__main__":
    while True:
        prompt = listen_for_prompt()
        if prompt:
            run_simulation(prompt)
