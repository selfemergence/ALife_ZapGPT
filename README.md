# ALife_ZapGPT

Free-form Language Prompting for Simulated Cellular Control

This repository provides the implementation and pretrained models for our ALife 2025 submission.  
It demonstrates how free-form natural language prompts can be used to steer collective behavior  
in a simulated cellular environment, using a closed-loop architecture we call **ZapGPT**.

# Repository Structure

- `cell.py`: Defines the individual cell agent and its physics-based interactions.
- `environment.py`: Defines the cell environment and manages agent dynamics.
- `cnn_p2i.py`: Implements the CNN-based Prompt-to-Intervention (P2I) model.
- `config.py`: Contains configuration parameters for the simulation and evolutionary strategy.
- `test_p2i_live.py`: Interactive simulation with live text-prompt input.
- `test_p2i_voice.py`: Voice-based prompt input (optional, requires microphone support).
- `test_zapgpt_prompt.py`: Batch testing script for evaluating multiple prompts.
- `best_weights.pt`: Pretrained P2I model weights for generating vector fields.
- `saved_frames/`: Directory for saving simulation output frames.

# Installation

Clone this repository and install the required Python packages.

# Install with pip:
