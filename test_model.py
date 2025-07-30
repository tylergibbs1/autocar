#!/usr/bin/env python3
"""Quick test to verify trained model loading"""
import torch
import torch.nn as nn
import os

# Same device selection
def pick_device(n_vehicles: int, horizon: int) -> torch.device:
    work = n_vehicles * horizon
    use_cuda = torch.cuda.is_available() and work >= 1500
    return torch.device('cuda' if use_cuda else 'cpu')

device = pick_device(16, 15)
print(f"Device: {device}")

# Same neural network architecture  
class NeuralMPCController(nn.Module):
    def __init__(self, horizon: int = 15, max_vehicles: int = 20):
        super().__init__()
        self.horizon = horizon
        self.max_vehicles = max_vehicles
        
        input_size = 3 + 3 * (max_vehicles - 1) + 2
        
        self.control_network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, horizon)
        )
    
    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.control_network(state_tensor)

# Test loading
print("Creating neural network...")
neural_mpc = NeuralMPCController(max_vehicles=16)
neural_mpc = neural_mpc.to(device=device, dtype=torch.float32)

print("Loading trained model...")
if os.path.exists("trained_neural_mpc.pth"):
    checkpoint = torch.load("trained_neural_mpc.pth", map_location=device)
    neural_mpc.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in neural_mpc.parameters())}")
    
    # Test inference
    neural_mpc.eval()
    test_input = torch.randn(1, 3 + 3 * 15 + 2).to(device)  # Sample input for max_vehicles=16
    with torch.no_grad():
        output = neural_mpc(test_input)
    print(f"Test inference successful! Output shape: {output.shape}")
    
else:
    print("trained_neural_mpc.pth not found")