import torch
import numpy as np
import os
import re
import json
import base64
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from sentence_transformers import SentenceTransformer
from pathlib import Path
import csv
import sys

from cnn_p2i import CNN_P2I
from environment import CellEnvironment
from config import *

# === Device Setup ===
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# === Seed for reproducibility ===
np.random.seed(SEED)
torch.manual_seed(SEED)

import argparse

# === CLI: Single or Multiple Prompts ===
parser = argparse.ArgumentParser(description="Run ZapGPT on a given prompt or list of prompts.")
parser.add_argument("--prompt", type=str, help="Single prompt to test")
parser.add_argument("--prompts", nargs="+", help="List of prompts to test (evaluated sequentially)")
args = parser.parse_args()

if args.prompts:
    PROMPTS = args.prompts
elif args.prompt:
    PROMPTS = [args.prompt]
else:
    PROMPTS = ["assemble the cells"]  # Default

# === Prompt Configuration ===
#PROMPT = "assemble the cells"
PROMPT = PROMPTS[0]
print(f"Testing Prompt: {PROMPT}")
VLM_MODEL = "mistral-small3.1"
TRIALS = 30
MODEL_PATH = "best_weights.pt"

# === Paths ===
prompt_dir = PROMPT.lower().replace(" ", "_").replace("?", "").replace(".", "")
save_dir = Path("results/visualize") / prompt_dir
save_dir.mkdir(parents=True, exist_ok=True)

# === Load Prompt Embedding ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
prompt_embedding = embedder.encode([PROMPT], convert_to_tensor=True).to(device)

# === Load Weights ===
if not os.path.exists(MODEL_PATH):
    print(f"No pretrained weights found at: {MODEL_PATH}")
    sys.exit(1)

weights = torch.load(MODEL_PATH, map_location=device)
p2i = CNN_P2I(output_dim=OUTPUT_DIM).to(device)
p2i.set_weights(weights)
p2i.eval()
print(f"✅ Loaded weights from: {MODEL_PATH}")

# === Vision-Language Model Utilities ===
def ask_ollama_vision(model_name, image_bytes, question):
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": question,
            "images": [image_base64],
            "stream": True,
            "options": {"num_predict": 200, "temperature": 0.0},
        },
        stream=True,
    )
    full_output = ""
    for line in response.iter_lines():
        if line:
            try:
                json_line = json.loads(line.decode("utf-8"))
                full_output += json_line.get("response", "")
            except json.JSONDecodeError:
                continue
    return full_output.strip()

def score_with_ollama(prompt, description):
    import ollama
    sys_prompt = f"""
    A vision-language model observed a particle simulation and generated this description:
    "{description}"
    The original user goal was: "{prompt}"
    On a scale from 0.0 to 1.0, how well does the description match the user's goal?
    Only return a numeric score. No explanation.
    """
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": sys_prompt}],
        options={"temperature": 0.0}
    )
    content = response["message"]["content"]
    match = re.search(r"\d*\.\d+|\d+", content)
    return float(match.group()) if match else 0.0

# === Run Trials ===
scores = []
for trial in range(TRIALS):
    print(f"\n Trial {trial}")

    with torch.no_grad():
        vector_field = p2i(prompt_embedding).squeeze(0).cpu().numpy()

    env = CellEnvironment(
        num_cells=NUM_CELLS,
        width=ENV_WIDTH,
        height=ENV_HEIGHT,
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        cell_radius=CELL_RADIUS
    )
    env.vector_field = vector_field

    for _ in range(TIMESTEPS):
        env.update()

    final_positions = np.array([cell.position for cell in env.cells])

    # === Save Image ===
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(0, ENV_WIDTH)
    ax.set_ylim(0, ENV_HEIGHT)
    ax.set_aspect('equal')
    ax.set_title(f"Final Cell Positions\nPrompt: '{PROMPT}'")
    ax.axis('off')
    ax.add_collection(PatchCollection([Circle(pos, CELL_RADIUS) for pos in final_positions], color='blue', alpha=0.8))
    plt.savefig(save_dir / f"trial_{trial}.png", bbox_inches='tight')
    plt.close()

    # === Vision-Language Model Evaluation ===
    with open(save_dir / f"trial_{trial}.png", "rb") as f:
        image_bytes = f.read()

    vision_prompt = f"Describe the overall distribution and shape of dots in this image. The original prompt was: '{PROMPT}'."
    description = ask_ollama_vision(VLM_MODEL, image_bytes, vision_prompt)
    score = score_with_ollama(PROMPT, description)
    scores.append(score)

    with open(save_dir / f"description_score_{trial}.txt", "w") as f:
        f.write(f"-- Description:\n{description}\n\n")
        f.write(f"-- Score: {score:.4f}\n")

    print(f"🔢 Score: {score:.4f}")

# === Save Scores CSV ===
csv_path = save_dir / "scores.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["prompt_name", "trial", "score"])
    for trial, score in enumerate(scores):
        writer.writerow([PROMPT, trial, f"{score:.4f}"])

print(f"\n Scores saved to: {csv_path}")
print(f" Average Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")