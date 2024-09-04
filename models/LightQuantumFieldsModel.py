import pygame
from pygame.locals import *
from OpenGL.GL import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from losses.custom_losses import CustomLosses
from datasets import BicycleDataset
from transformers import SwinModel, SwinConfig
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from utils.sh_utils import SH2RGB 
import os
import sys
import subprocess
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader
)

# Add ACEZero to the Python path
sys.path.append('/home/nicolas/LightQuantumField3D/acezero')
from acezero import ace_zero

# Fourier Positional Encoding function
class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_frequencies=10):
        super(FourierPositionalEncoding, self).__init__()
        self.num_frequencies = num_frequencies

    def forward(self, x):
        frequencies = torch.arange(self.num_frequencies, dtype=torch.float32, device=x.device)
        frequencies = 2 ** frequencies  # Exponential frequencies
        freq_x = x.unsqueeze(-1) * frequencies * torch.pi
        encoding = torch.cat([torch.sin(freq_x), torch.cos(freq_x)], dim=-1)
        return encoding.view(x.shape[0], -1)

# Checkpoint functions
def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        best_model_filename = os.path.join("models", "best_model.pth.tar")
        torch.save(state, best_model_filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            print(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
            return model, optimizer, start_epoch, best_loss
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch.")
            return model, optimizer, 0, float('inf')
    else:
        print(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")
        return model, optimizer, 0, float('inf')

# BEV transformation function
def bev_transform(voxel_grid):
    bev_projection = torch.max(voxel_grid, dim=2)[0]  # Projecting along z-axis
    return bev_projection

# Quantum Path Layer with Monte Carlo Simulation
class QuantumPathLayer(nn.Module):
    def __init__(self, num_paths=1000):
        super(QuantumPathLayer, self).__init__()
        self.num_paths = num_paths
    
    def forward(self, mesh, voxel):
        path_weights = torch.rand(self.num_paths, device=mesh.device)
        weighted_paths = torch.einsum('nd,n->nd', mesh, path_weights)
        combined_output = weighted_paths.sum(dim=0) / self.num_paths
        return combined_output

# Light Quantum Fields Model with Differentiable Rendering
class LightQuantumFieldsModel(nn.Module):
    def __init__(self):
        super(LightQuantumFieldsModel, self).__init__()
        
        # Swin Transformer V2 Configuration
        self.swin_config = SwinConfig(image_size=224, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
        self.swin_transformer = SwinModel(self.swin_config)
        
        # ResNet-FPN backbone for feature fusion
        self.resnet_fpn = resnet_fpn_backbone('resnet50', pretrained=True)
        
        # Voxel layers
        self.voxel_layer1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.voxel_layer2 = nn.ReLU()
        self.voxel_layer3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)

        # BEV-specific layers for 2D projection
        self.bev_layer1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bev_layer2 = nn.ReLU()
        self.bev_layer3 = nn.Conv2d(128, 3, kernel_size=1)

        # Quantum path integrals with Monte Carlo simulation
        self.quantum_path_layer = QuantumPathLayer()

        # Spherical Harmonics for lighting or radiance modeling
        self.spherical_harmonics = SH2RGB(4)

        # Color and Density layers
        self.density_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Scalar output for density σ
        )

        self.color_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # RGB output for color c
        )

        # Fourier Positional Encoding
        self.positional_encoding = FourierPositionalEncoding()

    def forward(self, x, viewing_direction):
        x = self.positional_encoding(x)  # Apply Fourier positional encoding
        swin_output = self.swin_transformer(x).last_hidden_state
        fpn_output = self.resnet_fpn(swin_output)
        
        voxel_output = x.view(-1, 1, 32, 32, 32)
        voxel_output = self.voxel_layer1(voxel_output)
        voxel_output = self.voxel_layer2(voxel_output)
        voxel_output = self.voxel_layer3(voxel_output)
        
        bev_input = bev_transform(voxel_output)
        bev_output = self.bev_layer1(bev_input)
        bev_output = self.bev_layer2(bev_output)
        bev_output = self.bev_layer3(bev_output)

        sh_lighting = self.spherical_harmonics(bev_output)
        combined_output = self.quantum_path_layer(fpn_output, sh_lighting)

        # Adjust for viewing direction (θ, φ)
        adjusted_output = self.adjust_for_viewing_direction(combined_output, viewing_direction)

        # Calculate density (σ) and color (c)
        density = self.density_layer(adjusted_output)
        color = self.color_layer(adjusted_output)

        return color, density

    def adjust_for_viewing_direction(self, combined_output, viewing_direction):
        # Implement refraction and viewing direction adjustments here
        return combined_output

# Run ACEZero for camera pose generation
def run_ace_zero(rgb_files, results_folder):
    args = [
        "python", "/home/nicolas/LightQuantumField3D/acezero/ace_zero.py",
        rgb_files,
        results_folder,
        "--iterations_max", "100",
        "--registration_threshold", "0.99",
    ]
    subprocess.run(args, check=True)

def load_poses(results_folder):
    # Load poses from ACEZero output
    pass

# Example rendering function
def render_scene():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    pygame.display.flip()

# Training loop
def train():
    model = LightQuantumFieldsModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_functions = CustomLosses()
    
    os.makedirs('models', exist_ok=True)
    
    # Load checkpoint
    checkpoint_path = "models/checkpoint.pth.tar"
    model, optimizer, start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer)
    
    dataset = BicycleDataset(root_dir='/mnt/c/Users/dells/Downloads/360_v2/bicycle')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    num_epochs = 10
    results_folder = "results/ace_zero_output"
    rgb_files = "/path/to/rgb/files/*.jpg"

    for epoch in range(start_epoch, num_epochs):
        run_ace_zero(rgb_files, results_folder)
        predicted_poses = load_poses(results_folder)
        
        model.train()
        running_loss = 0.0
        for i, (images, _, bounds) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward pass: get the model output (color and density)
            outputs, densities = model(images, predicted_poses)
            
            # Compute the hybrid loss between the output and ground-truth
            loss = loss_functions.hybrid_loss(outputs, predicted_poses, bounds)
            loss.backward()

            # Perform optimization step
            optimizer.step()

            running_loss += loss.item()

            # Render the scene for visual feedback (real-time OpenGL rendering)
            render_scene()

        # Calculate the average loss for the epoch
        epoch_loss = running_loss / len(dataloader)

        # Check if this is the best model (minimum loss)
        is_best = epoch_loss < best_loss
        best_loss = min(epoch_loss, best_loss)

        # Save the model checkpoint after each epoch
        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint_state, is_best, filename=f"models/checkpoint_epoch_{epoch + 1}.pth.tar")

    pygame.quit()

    # Save the final trained model
    torch.save(model.state_dict(), 'models/light_quantum_fields_model_bicycle_final.pth')

if __name__ == "__main__":
    train()

