import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN_P2I(nn.Module):
    """
    CNN-based P2I Model: Converts BERT embeddings into vector fields using convolutional upsampling.
    """
    def __init__(self, output_dim=(10, 10, 2)):  # Can be (5,5,2), (10,10,2), etc.
        super(CNN_P2I, self).__init__()

        self.output_dim = output_dim  # (grid_width, grid_height, 2)

        # Linear layer to map 768 BERT embedding → feature map of size (64, 5, 5)
        #self.fc = nn.Linear(768, 64 * 5 * 5)  # 64 filters, 5x5 feature map
        self.fc = nn.Linear(384, 64 * 5 * 5)  # sentence transformer has 384 output, instead of 768

        # Transposed convolutions for upsampling
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upscale 5x5 → 10x10
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)  # Keep 10x10
        self.conv3 = nn.ConvTranspose2d(16, 2, kernel_size=3, stride=1, padding=1)  # Final layer (2 channels)

    def forward(self, prompt_embedding):
        """
        Generate vector field from precomputed BERT embedding.

        Args:
            prompt_embedding (torch.Tensor): Shape (batch_size, 768)

        Returns:
            torch.Tensor: Vector field of shape (batch_size, output_dim[0], output_dim[1], 2)
        """
        batch_size = prompt_embedding.shape[0]

        # Transform 768-dim embedding → initial feature map (batch, 64, 5, 5)
        x = self.fc(prompt_embedding)
        x = x.view(batch_size, 64, 5, 5)  # Reshape to (batch_size, 64, 5, 5)

        # Upsample with transposed convolutions
        x = F.relu(self.conv1(x))  # 5x5 → 10x10
        x = F.relu(self.conv2(x))  # Keep 10x10
        x = torch.tanh(self.conv3(x))  # Final layer (2 channels)

        # 🔹 Ensure the output shape exactly matches `output_dim`
        x = F.interpolate(x, size=(self.output_dim[0], self.output_dim[1]), mode="bilinear", align_corners=False)

        # Reshape to (batch, grid_width, grid_height, 2)
        return x.permute(0, 2, 3, 1)  # (batch, H, W, 2)

    def get_weights(self):
        """
        Retrieve weights of the model.

        Returns:
            list: A list of cloned parameter tensors.
        """
        return [p.clone().detach() for p in self.parameters()]

    def set_weights(self, weights):
        """
        Set model weights from a list of tensors.

        Args:
            weights (list): A list of tensors to assign to the model.
        """
        with torch.no_grad():
            for param, new_weight in zip(self.parameters(), weights):
                param.copy_(new_weight)
                
    
    def get_flat_weights(self):
        return torch.cat([p.view(-1) for p in self.parameters()]).detach()

    def set_flat_weights(self, flat_weights):
        """
        Set model weights from a flattened 1D tensor.
        """
        pointer = 0
        with torch.no_grad():
            for param in self.parameters():
                numel = param.numel()
                param.copy_(flat_weights[pointer:pointer + numel].view_as(param))
                pointer += numel
