# ALife_ZapGPT

Free-form Language Prompting for Simulated Cellular Control

This repository provides the implementation and pretrained models for our ALife 2025 submission.  
It demonstrates how free-form natural language prompts can be used to steer collective behavior  
in a simulated cellular environment, using a closed-loop architecture we call **ZapGPT**.

## Repository Structure

- `cell.py`: Defines the individual cell agent and its physics-based interactions.
- `environment.py`: Defines the cell environment and manages agent dynamics.
- `cnn_p2i.py`: Implements the CNN-based Prompt-to-Intervention (P2I) model.
- `config.py`: Contains configuration parameters for the simulation and evolutionary strategy.
- `test_p2i_live.py`: Interactive simulation with live text-prompt input.
- `test_p2i_voice.py`: Voice-based prompt input (optional, requires microphone support).
- `test_zapgpt_prompt.py`: Batch testing script for evaluating multiple prompts.
- `best_weights.pt`: Pretrained P2I model weights for generating vector fields.
- `saved_frames/`: Directory for saving simulation output frames.

## Installation

Clone this repository and install the required Python packages.

### Install with pip:

```bash
pip install torch numpy pygame sentence-transformers

Optional:
For voice input support, additional packages like speechrecognition or pyaudio may be required.


Pre-trained Model Weights

This repository includes best_weights.pt, containing pre-trained weights
for the Prompt-to-Intervention (P2I) model.

All simulation scripts (test_p2i_live.py, test_p2i_voice.py, test_zapgpt_prompt.py)
automatically load this file to generate behaviors based on your prompts.

You can replace this file with your own trained weights if you wish to evolve or fine-tune new behaviors.

## How to Run

### 1. Interactive Prompt Simulation

Run the interactive demo where you can enter free-form language prompts:

```bash
python test_p2i_live.py

A Pygame window will open.  
You can enter any free-form natural language prompt, for example:

- form a cluster
- assemble the cells
- bring all agents into a cluster
- can you organize the objects into one cluster
- clustering
- could the objects be scattered away
- drift apart from one another
- scatter apart
- scattering

After the simulation runs, the start and end frames will be saved to:
saved_frames/<your_prompt>/

### Outputs

Simulation outputs are saved as images in the `saved_frames/` directory.  
Each prompt creates its own subfolder containing:

- `frame_start.png`: The initial configuration of cells.
- `frame_end.png`: The final configuration after simulation completes.

Example:

### Voice Input Simulation (Optional)

You can try voice-based interaction if you have a microphone installed.  
Run the following script:

```bash
python test_p2i_voice.py

This script will listen for your spoken prompt and attempt to convert it to text.
After recognition, the simulation will run based on your spoken instruction.

### Reproducing Paper Results: Batch Prompt Evaluation

To reproduce the testing on multiple unseen prompts,  
you can run the following command with a list of prompts:

```bash
python test_zapgpt_prompt.py --prompts "form a cluster" "scatter apart" "assemble the cells"

Alternatively, test a single prompt:
python test_zapgpt_prompt.py --prompt "form a cluster"

### Output

For each prompt, the script will:
	•	Run 30 independent simulation trials.
	•	Save final images in results/visualize/<prompt>/trial_<n>.png.
	•	Evaluate results using a vision-language model (requires Ollama server).
	•	Save descriptions and alignment scores in .txt and .csv files.

results/visualize/form_a_cluster/
    trial_0.png
    description_score_0.txt
    ...
    scores.csv

Note:
This script requires a running Ollama server for image-to-text and scoring evaluation.
Refer to Ollama documentation for setup: https://ollama.com

## Requirements

- Python 3.8 or higher
- PyTorch (for model execution)
- SentenceTransformers (for prompt embedding)
- Numpy, Matplotlib, Pygame (for simulation and visualization)
- Optional: SpeechRecognition, PyAudio (for voice input)
- Optional: Ollama API running locally (for VLM-based scoring)

### Install core requirements:

```bash
pip install torch numpy pygame matplotlib sentence-transformers


## License

This project is licensed under the Apache License 2.0.

You are free to use, modify, and distribute this code under the terms of the license.

See the [LICENSE](LICENSE) file for full details.
