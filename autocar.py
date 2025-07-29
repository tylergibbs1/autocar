"""
Modern Autonomous Vehicle Traffic Simulation with Advanced ML + Panda3D 3D Visualization
Testing hypothesis: Can autonomous vehicles eliminate traffic lights through coordination?

Architecture:
- GRPO (Group Relative Policy Optimization) for efficient RL
- Graph Neural Networks for vehicle relationship modeling
- Learned communication protocols between vehicles
- Continuous action spaces for realistic control
- Transformer networks for sequential decision making
- Panda3D for immersive 3D visualization
"""

import torch
import torch.nn as nn   
import torch.nn.functional as F
import torch.optim as optim

# Advanced ML dependencies (optional)
try:
    import torch_geometric
    from torch_geometric.nn import GATConv, SAGEConv, GCNConv
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import to_dense_batch
    ADVANCED_ML_AVAILABLE = True
    print("🔥 Advanced ML with GraphSAGE enabled!")
except ImportError:
    print("📦 Advanced ML features require torch_geometric")
    print("💡 Install with: pip install torch_geometric")
    print("🔄 Falling back to standard training...")
    ADVANCED_ML_AVAILABLE = False

# Panda3D dependencies (optional)
try:
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task
    from direct.actor.Actor import Actor
    from panda3d.core import (
        CardMaker, NodePath, DirectionalLight, AmbientLight, 
        PointLight, Fog, TransparencyAttrib, TextNode,
        PerspectiveLens, OrthographicLens, Camera,
        LColor, Vec3, Point3, CollisionTraverser,
        CollisionHandlerQueue, CollisionRay, CollisionNode,
        CollisionSphere, BitMask32, Texture, TextureStage,
        SamplerState, WindowProperties, FrameBufferProperties,
        GraphicsPipe, Shader, ShaderAttrib
    )
    from direct.showbase.DirectObject import DirectObject
    from direct.gui.OnscreenText import OnscreenText
    from direct.gui.DirectGui import DirectButton, DirectFrame
    PANDA3D_AVAILABLE = True
    print("🎮 Panda3D available - 3D visualization enabled!")
except ImportError:
    print("📦 Panda3D not found - install with: pip install panda3d")
    print("🔄 Falling back to 2D pygame visualization...")
    PANDA3D_AVAILABLE = False

import pygame
import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional
from collections import deque
import time
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich.panel import Panel

# Initialize rich console for better logging
console = Console()

# ============================================================================
# INTERSECTION COORDINATION SYSTEM
# ============================================================================

class IntersectionManager:
    """Proper intersection coordination with reservation protocol"""
    def __init__(self, x: float, y: float, radius: float = 40):
        self.x = x
        self.y = y
        self.radius = radius
        self.reservations = []  # [(vehicle_id, eta_start, eta_end), ...]
        self.current_time = 0
        
    def request_reservation(self, vehicle_id: int, eta: float, duration: float = 2.0) -> bool:
        """Request intersection slot - returns True if granted"""
        eta_start = eta
        eta_end = eta + duration
        
        # Check for conflicts with existing reservations
        for vid, start, end in self.reservations:
            if vid == vehicle_id:
                continue  # Update existing reservation
            if not (eta_end < start or eta_start > end):  # Overlap check
                return False
        
        # Remove old reservation for this vehicle
        self.reservations = [(vid, s, e) for vid, s, e in self.reservations if vid != vehicle_id]
        
        # Add new reservation
        self.reservations.append((vehicle_id, eta_start, eta_end))
        return True
    
    def update(self, dt: float):
        """Update intersection state"""
        self.current_time += dt
        
        # Clean expired reservations
        self.reservations = [(vid, s, e) for vid, s, e in self.reservations 
                           if e > self.current_time]
    
    
    def is_vehicle_cleared(self, vehicle_id: int) -> bool:
        """Check if vehicle has valid reservation for current time"""
        for vid, start, end in self.reservations:
            if vid == vehicle_id and start <= self.current_time <= end:
                return True
        return False

class FixedCycleSignal:
    """Baseline traffic light for comparison"""
    def __init__(self, cycle_time: float = 30.0):
        self.cycle_time = cycle_time
        self.current_time = 0
        
    def update(self, dt: float):
        self.current_time += dt
        
    def current_state(self) -> str:
        """Returns 'NS' (north-south green) or 'EW' (east-west green)"""
        phase = (self.current_time % self.cycle_time) / self.cycle_time
        return 'NS' if phase < 0.5 else 'EW'
    
    def can_proceed(self, vehicle_heading: float) -> bool:
        """Check if vehicle can proceed based on heading and light state"""
        state = self.current_state()
        
        # Normalize heading to [0, 2π]
        heading = vehicle_heading % (2 * math.pi)
        
        # North/South directions (0° ± 45°, 180° ± 45°)
        is_ns = (heading < math.pi/4) or (heading > 7*math.pi/4) or (3*math.pi/4 < heading < 5*math.pi/4)
        
        if state == 'NS':
            return is_ns
        else:  # 'EW'
            return not is_ns

# ============================================================================
# NEURAL NETWORK COMPONENTS
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention for transformer"""
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(context), attn_weights

class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x, attn_weights

class TrafficTransformer(nn.Module):
    """Transformer for traffic coordination"""
    def __init__(self, input_dim: int = 64, d_model: int = 256, n_heads: int = 8, 
                 n_layers: int = 4, d_ff: int = 1024, max_seq_len: int = 20, 
                 output_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def create_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        x = self.input_projection(x)
        
        if seq_len <= self.pos_encoding.size(1):
            x += self.pos_encoding[:, :seq_len, :].to(x.device)
        
        x = self.dropout(x)
        
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask)
            attention_weights.append(attn_weights)
        
        output = self.output_projection(x[:, -1, :])
        
        return output, attention_weights

# ============================================================================
# KALMAN FILTER FOR STATE ESTIMATION
# ============================================================================

class KalmanFilter:
    """Extended Kalman Filter for vehicle state estimation"""
    def __init__(self, state_dim: int = 4, obs_dim: int = 2):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        self.x = np.zeros(state_dim)  # [x, y, vx, vy]
        self.P = np.eye(state_dim) * 1.0
        self.Q = np.eye(state_dim) * 0.1
        self.R = np.eye(obs_dim) * 0.5
        self.F = np.eye(state_dim)
        
        self.H = np.zeros((obs_dim, state_dim))
        self.H[0, 0] = 1.0  # observe x
        self.H[1, 1] = 1.0  # observe y
        
    def predict(self, dt: float):
        self.F[0, 2] = dt  # x += vx * dt
        self.F[1, 3] = dt  # y += vy * dt
        
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        

# ============================================================================
# AUTONOMOUS VEHICLE
# ============================================================================

class AutonomousVehicle:
    """Modern autonomous vehicle with transformer + Kalman filter"""
    _next_id = 1  # Class variable for unique IDs
    
    def __init__(self, x: float, y: float, heading: float):
        # Assign unique sequential ID
        self.id = AutonomousVehicle._next_id
        AutonomousVehicle._next_id += 1
        
        # Physical state
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = random.uniform(1.0, 3.0)  # Start with small initial speed instead of 0
        self.acceleration = 0.0
        self.max_speed = 45.0  # AUTONOMOUS SPEEDS: 162 km/h for true coordination
        self.max_acceleration = 3.0
        self.max_deceleration = -8.0
        
        # Autonomous systems
        self.kalman_filter = KalmanFilter()
        self.transformer = None  # Set externally
        
        # Sensors
        self.sensor_range = 100.0
        self.sensor_noise_std = 0.1
        
        # State tracking
        self.target_speed = random.uniform(35, 42)  # HIGHWAY-LEVEL autonomous speeds
        self.behavior_mode = 'normal'
        self.observation_history = deque(maxlen=20)
        self.action_history = deque(maxlen=20)
        self.collision_risk = 0.0
        self.total_reward = 0.0
        
        # Intersection coordination
        self.has_reservation = False
        self.reservation_eta = 0.0
        self.waiting_time = 0.0
        self.last_intersection_request = None
        
        # Progress tracking for delta rewards
        self.prev_exit_dist = self.calculate_exit_distance()
        
    def get_observations(self, other_vehicles: List['AutonomousVehicle'], 
                        intersections: List) -> torch.Tensor:
        """Get sensor observations and convert to neural network input"""
        features = np.zeros(64)
        
        # Ego vehicle features (8 dims)
        features[0:8] = [
            self.x / 1000.0,
            self.y / 1000.0,
            (self.heading % (2 * math.pi)) / (2 * math.pi),  # Ensure heading is in [0, 1]
            self.speed / self.max_speed,
            self.acceleration / max(abs(self.max_acceleration), abs(self.max_deceleration)),  # Symmetric normalization
            self.target_speed / self.max_speed,
            self.collision_risk,
            len(self.observation_history) / 20.0
        ]
        
        # Nearby vehicles (up to 7 vehicles, 6 features each = 42 dims)
        nearby = []
        for vehicle in other_vehicles:
            if vehicle.id != self.id:
                distance = math.sqrt((vehicle.x - self.x)**2 + (vehicle.y - self.y)**2)
                if distance <= self.sensor_range:
                    # Add sensor noise
                    noise_x = np.random.normal(0, self.sensor_noise_std * distance)
                    noise_y = np.random.normal(0, self.sensor_noise_std * distance)
                    nearby.append((distance, vehicle, noise_x, noise_y))
        
        # Random sample up to 7 neighbors to avoid invisibility of extra cars
        if len(nearby) > 7:
            nearby = random.sample(nearby, 7)
        
        for i, (dist, vehicle, noise_x, noise_y) in enumerate(nearby):
            base_idx = 8 + i * 6
            features[base_idx:base_idx+6] = [
                (vehicle.x + noise_x - self.x) / self.sensor_range,
                (vehicle.y + noise_y - self.y) / self.sensor_range,
                vehicle.speed * math.cos(vehicle.heading) / self.max_speed,
                vehicle.speed * math.sin(vehicle.heading) / self.max_speed,
                dist / self.sensor_range,
                np.clip((self.speed - vehicle.speed) / self.max_speed, -1, 1)  # Prevent overflow
            ]
        
        # Intersection information (6 dims)
        if intersections:
            intersection = min(intersections, 
                key=lambda i: math.sqrt((i.x - self.x)**2 + (i.y - self.y)**2))
            
            dist = math.sqrt((intersection.x - self.x)**2 + (intersection.y - self.y)**2)
            
            # Calculate occupancy (number of active reservations)
            occupancy = len(intersection.reservations) if hasattr(intersection, 'reservations') else 0
            
            features[50:56] = [
                (intersection.x - self.x) / self.sensor_range,
                (intersection.y - self.y) / self.sensor_range,
                dist / self.sensor_range,
                occupancy / 5.0,
                1.0 if dist < 50 else 0.0,
                1.0 if self.has_reservation else 0.0  # Whether vehicle has reservation
            ]
        
        # Behavioral context (8 dims)
        behavior_encoding = {'normal': 0.25, 'aggressive': 0.5, 'conservative': 0.75, 'cooperative': 1.0}
        features[56:64] = [
            behavior_encoding.get(self.behavior_mode, 0.25),
            random.random() * 0.1,  # Exploration noise
            0, 0, 0, 0, 0, 0  # Reserved
        ]
        
        # Create tensor on correct device if available
        tensor = torch.FloatTensor(features)
        if hasattr(self, '_trainer') and hasattr(self._trainer, 'device'):
            tensor = tensor.to(self._trainer.device)
        return tensor
    
    def calculate_exit_distance(self) -> float:
        """Calculate minimum distance to any exit boundary"""
        return min([
            abs(self.x - 20),    # West exit
            abs(self.x - 1180),  # East exit  
            abs(self.y - 20),    # North exit
            abs(self.y - 780)    # South exit
        ])
    
    def lane_center(self) -> Tuple[float, float]:
        """
        Returns the (x,y) coordinate of the proper lane center the vehicle
        should be in, offset from the road center line.
        """
        lane_offset = 3.0  # 3 meters from center line
        
        if abs(self.heading) < 0.2:                  # east‑bound (right lane)
            road_center_y = 300 if self.y < 400 else 500
            return self.x, road_center_y - lane_offset  # Right lane (south of center)
        if abs(self.heading - math.pi) < 0.2:        # west‑bound (right lane) 
            road_center_y = 300 if self.y < 400 else 500
            return self.x, road_center_y + lane_offset  # Right lane (north of center)
        if abs(self.heading - math.pi/2) < 0.2:      # south‑bound (right lane)
            road_center_x = 400 if self.x < 600 else 800
            return road_center_x + lane_offset, self.y  # Right lane (east of center)
        # north‑bound (right lane)
        road_center_x = 400 if self.x < 600 else 800
        return road_center_x - lane_offset, self.y  # Right lane (west of center)

    def lane_deviation(self) -> float:
        """
        Signed lateral error (metres). Positive = right of centre,
        Negative = left of centre (from driver's perspective).
        """
        cx, cy = self.lane_center()
        # lateral axis is perpendicular to heading
        lateral_vec = (-math.sin(self.heading),  math.cos(self.heading))
        return (self.x - cx) * lateral_vec[0] + (self.y - cy) * lateral_vec[1]
    
    def prepare_sequence(self) -> torch.Tensor:
        """Prepare observation sequence for batched inference"""
        seq_len = 5
        
        # Guard against empty history
        if not self.observation_history:
            device = self._trainer.device if hasattr(self, '_trainer') and hasattr(self._trainer, 'device') else 'cpu'
            return torch.zeros(seq_len, 64, device=device)
            
        if len(self.observation_history) < seq_len:
            sequence = [torch.zeros_like(self.observation_history[0]) for _ in range(seq_len - len(self.observation_history))]
            sequence.extend(list(self.observation_history))
        else:
            sequence = list(self.observation_history)[-seq_len:]
        return torch.stack(sequence)
    
    def make_decision(self, observations: torch.Tensor = None, q_values: torch.Tensor = None):
        """Make driving decision - returns (action, log_prob) for GRPO"""
        
        # TEMPORARY DEBUG: Force simple forward movement to test basic vehicle movement
        # This bypasses the neural network entirely to test if vehicles can move
        if hasattr(self, '_debug_simple_movement') and self._debug_simple_movement:
            # Simple forward action: [acceleration, steering, communication]
            device = self._trainer.device if hasattr(self, '_trainer') and hasattr(self._trainer, 'device') else 'cpu'
            action = torch.tensor([0.8, 0.0, 0.0], device=device)  # Strong forward acceleration
            return action, torch.tensor(0.0, device=device)
        
        # ADVANCED PATH: GNN + Communication + Continuous Actions
        if (hasattr(self, 'policy_net') and hasattr(self.policy_net, 'continuous_actions') 
            and self.policy_net.continuous_actions and ADVANCED_ML_AVAILABLE):
            
            # Get communication messages
            received_messages = getattr(self, 'received_messages', [])
            if received_messages:
                comm_vector = torch.mean(torch.stack(received_messages), dim=0)
                self.received_messages = []  # Clear buffer
            else:
                device = self.policy_net.action_mean.weight.device if hasattr(self.policy_net, 'action_mean') else 'cpu'
                comm_vector = torch.zeros(8, device=device)
            
            # Get all vehicles 
            all_vehicles = getattr(self, '_all_vehicles', [self])
            
            # Forward through advanced transformer
            mean, log_std = self.policy_net(all_vehicles, self, comm_vector)
            
            # Sample continuous action
            std = torch.exp(log_std.clamp(-5, 2))
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()  # Sum over action dimensions
            
            # Store for communication broadcasting
            if len(action) > 2:
                self.comm_intensity = torch.sigmoid(action[2]).item()
            
            # Broadcast message if needed (simplified - remove complex broadcasting for now)
            # if (hasattr(self, 'communication') and self.comm_intensity > 0.5):
            #     message = self.policy_net.get_communication_message(all_vehicles, self)
            #     self.communication.broadcast_message(self, all_vehicles, message)
            
            return action.detach(), log_prob.detach()
        
        # FALLBACK PATH: Discrete transformer (should rarely be used now)
        if hasattr(self, 'transformer') and self.transformer is not None:
            if observations is not None:
                self.observation_history.append(observations)
            
            if observations is None:
                action = random.randint(0, 3)
                device = 'cpu'  # Discrete actions typically stay on CPU
                return torch.tensor(action, device=device), torch.tensor(0.0, device=device)
                
            input_sequence = self.prepare_sequence().unsqueeze(0)
            
            with torch.no_grad():
                q_vals, _ = self.transformer(input_sequence)
                action = torch.argmax(q_vals).item()
            
            self.action_history.append(action)
            device = 'cpu'  # Discrete actions typically stay on CPU
            return torch.tensor(action, device=device), torch.tensor(0.0, device=device)  # No log_prob for discrete
        
        # Emergency fallback - use simple forward movement
        print("⚠️  WARNING: Vehicle has no policy network - using forward movement!")
        # Simple continuous action: [moderate acceleration, no steering, no communication]
        device = self._trainer.device if hasattr(self, '_trainer') and hasattr(self._trainer, 'device') else 'cpu'
        action = torch.tensor([0.5, 0.0, 0.0], device=device)  # Should give ~3 m/s² acceleration
        return action, torch.tensor(0.0, device=device)
    
    def execute_action(self, action, dt: float):
        """Execute control action - supports both discrete and continuous actions"""
        
        # Handle continuous actions: [acceleration, steering, communication_intensity]
        if isinstance(action, (list, tuple, torch.Tensor)) and len(action) >= 2:
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            
            # Continuous action space with baseline movement
            base_acceleration = 1.0 if self.speed < self.target_speed * 0.8 else 0.0  # Baseline to maintain movement
            target_acceleration = base_acceleration + float(action[0]) * 6.0  # Scale to [-6, 6] + baseline
            steering_input = float(action[1])  # [-1, 1] for lane changes
            comm_intensity = float(action[2]) if len(action) > 2 else 0.0
            
            # Apply acceleration limits
            target_acceleration = np.clip(target_acceleration, -6.0, 4.0)
            
            # Store communication intensity for message broadcasting
            self.comm_intensity = comm_intensity
            
        else:
            # Discrete action space (legacy support)
            # Action mapping: 0=accelerate, 1=maintain, 2=brake, 3=cooperate
            if action == 0:  # Accelerate
                target_acceleration = min(2.0, self.max_acceleration)
            elif action == 1:  # Maintain
                target_acceleration = 0.0
            elif action == 2:  # Brake
                target_acceleration = max(-3.0, self.max_deceleration)
            else:  # Cooperate (yield)
                target_acceleration = max(-1.0, self.max_deceleration)
                self.behavior_mode = 'cooperative'
            
            steering_input = 0.0  # No steering for discrete actions
            self.comm_intensity = 0.0
        
        # Safety override for collision avoidance
        if self.collision_risk > 0.8:
            target_acceleration = min(target_acceleration, -1.5)
        elif self.collision_risk > 0.6:
            target_acceleration = min(target_acceleration, -0.3)
        
        self.acceleration = target_acceleration
        
        # Update speed with realistic dynamics
        self.speed = max(0, min(self.max_speed, self.speed + self.acceleration * dt))
        
        # Ensure minimum movement to prevent getting stuck (more aggressive)
        if self.speed < 2.0 and self.acceleration >= 0:
            self.speed = max(self.speed, 2.0)  # Maintain at least 2 m/s
        
        # If vehicle is way below target speed, give it a boost
        if self.speed < self.target_speed * 0.3:
            self.speed += 1.0 * dt  # Gradual speed boost
        
        # Enhanced lane following with steering input
        if abs(self.heading) < 0.2:  # Heading East (0°)
            target_y = 300 if abs(self.y - 300) < abs(self.y - 500) else 500
            lane_offset = 3.0  # Right lane offset
            target_y += lane_offset + steering_input * 8.0  # Steering adjustment
            
        elif abs(self.heading - math.pi) < 0.2:  # Heading West (180°)
            target_y = 300 if abs(self.y - 300) < abs(self.y - 500) else 500
            lane_offset = -3.0  # Right lane offset
            target_y += lane_offset + steering_input * 8.0
            
        elif abs(self.heading - math.pi/2) < 0.2:  # Heading South (90°)
            target_x = 400 if abs(self.x - 400) < abs(self.x - 800) else 800
            lane_offset = 3.0
            target_x += lane_offset + steering_input * 8.0
            
        elif abs(self.heading + math.pi/2) < 0.2:  # Heading North (-90°)
            target_x = 400 if abs(self.x - 400) < abs(self.x - 800) else 800
            lane_offset = -3.0
            target_x += lane_offset + steering_input * 8.0
        
        # Gentle lane correction with steering influence
        correction_strength = 1.2  # Stronger for continuous control
        if abs(self.heading) < 0.2 or abs(self.heading - math.pi) < 0.2:  # Horizontal
            if abs(self.y - target_y) > 2.0:  # Allow some deviation
                self.y += correction_strength * dt * (target_y - self.y)
        else:  # Vertical
            if abs(self.x - target_x) > 2.0:
                self.x += correction_strength * dt * (target_x - self.x)
        
        # Update position - maintain spawn heading to reach exits
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
    
    def update_physics_only(self, dt: float, other_vehicles: List['AutonomousVehicle'], 
                           intersections: List):
        """Update physics and observations only (for batch inference)"""
        # Kalman filter prediction
        self.kalman_filter.predict(dt)
        
        # Simulate noisy GPS measurement
        noisy_position = np.array([
            self.x + np.random.normal(0, 0.5),
            self.y + np.random.normal(0, 0.5)
        ])
        self.kalman_filter.update(noisy_position)
        
        # Intersection coordination
        self.handle_intersection_coordination(intersections, dt)
        
        # Get observations and update history
        observations = self.get_observations(other_vehicles, intersections)
        self.observation_history.append(observations)
        self.calculate_collision_risk(other_vehicles)
    
    def update(self, dt: float, other_vehicles: List['AutonomousVehicle'], 
               intersections: List):
        """Full update loop (for non-batch inference)"""
        self.update_physics_only(dt, other_vehicles, intersections)
        
        # Make decision and execute
        action = self.make_decision()
        self.execute_action(action, dt)
    
    def handle_intersection_coordination(self, intersections: List, dt: float):
        """Handle intersection coordination (reservation or traffic light)"""
        # Find nearest intersection
        nearest_intersection = None
        min_distance = float('inf')
        
        for intersection in intersections:
            if hasattr(intersection, 'x'):  # IntersectionManager
                dist = math.sqrt((intersection.x - self.x)**2 + (intersection.y - self.y)**2)
                if dist < min_distance:
                    min_distance = dist
                    nearest_intersection = intersection
        
        if nearest_intersection and hasattr(nearest_intersection, 'request_reservation'):
            # Check if using traffic lights (baseline mode)
            trainer = getattr(self, '_trainer', None)
            use_lights = trainer and getattr(trainer, 'use_traffic_lights', False)
            
            if use_lights:
                # Traffic light logic
                if min_distance < 30:
                    # Find corresponding light (assumes same indexing)
                    light_idx = 0  # Default
                    if hasattr(trainer, 'intersections'):
                        for i, isec in enumerate(trainer.intersections):
                            if isec.get('manager') == nearest_intersection:
                                light_idx = i
                                break
                    
                    if light_idx < len(trainer.traffic_lights):
                        light = trainer.traffic_lights[light_idx]
                        if not light.can_proceed(self.heading):
                            self.waiting_time += dt
                            self.target_speed = min(self.target_speed, 5.0)
                        else:
                            self.waiting_time = 0
            else:
                # Reservation logic (original)
                if min_distance > 0 and self.speed > 0.1:  # Fix divide-by-zero
                    eta = nearest_intersection.current_time + (min_distance / max(self.speed, 0.1))
                    
                    # Dynamic duration based on speed and intersection size
                    duration = (nearest_intersection.radius * 2 + 10) / max(self.speed, 1.0)
                    
                    # Request reservation if within range and don't have one
                    if min_distance < 80 and not self.has_reservation:
                        if nearest_intersection.request_reservation(self.id, eta, duration):
                            self.has_reservation = True
                            self.reservation_eta = eta
                            self.last_intersection_request = nearest_intersection
                    
                    # Check if we should wait (no reservation and close to intersection)
                    if min_distance < 25 and not nearest_intersection.is_vehicle_cleared(self.id):
                        self.waiting_time += dt
                        # Don't stop completely - allow slow movement
                        self.target_speed = min(self.target_speed, 8.0)
                        # If waiting too long, be more aggressive
                        if self.waiting_time > 3.0:
                            self.target_speed = min(self.target_speed, 15.0)
                    else:
                        self.waiting_time = 0
                
                # Clear reservation if we've passed through
                if self.has_reservation and min_distance > 60:
                    self.has_reservation = False
                    self.last_intersection_request = None
    
    def calculate_collision_risk(self, other_vehicles: List['AutonomousVehicle']) -> float:
        """Calculate collision risk using time-to-collision"""
        min_ttc = float('inf')
        
        for vehicle in other_vehicles:
            if vehicle.id == self.id:
                continue
                
            dx = vehicle.x - self.x
            dy = vehicle.y - self.y
            dvx = vehicle.speed * math.cos(vehicle.heading) - self.speed * math.cos(self.heading)
            dvy = vehicle.speed * math.sin(vehicle.heading) - self.speed * math.sin(self.heading)
            
            if dvx**2 + dvy**2 > 1e-6:
                t_cpa = -(dx * dvx + dy * dvy) / (dvx**2 + dvy**2)
                
                if t_cpa > 0:
                    x_cpa = dx + dvx * t_cpa
                    y_cpa = dy + dvy * t_cpa
                    dist_cpa = math.sqrt(x_cpa**2 + y_cpa**2)
                    
                    if dist_cpa < 8.0:  # Reduced safety distance
                        min_ttc = min(min_ttc, t_cpa)
        
        if min_ttc < float('inf'):
            self.collision_risk = max(0, 1 - min_ttc / 8.0)  # Less sensitive
        else:
            self.collision_risk = 0
            
        return self.collision_risk
    
    def draw(self, screen):
        """Draw vehicle"""
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        
        length, width = 12, 6
        corners = [
            (-length/2, -width/2), (length/2, -width/2),
            (length/2, width/2), (-length/2, width/2)
        ]
        
        rotated_corners = []
        for corner_x, corner_y in corners:
            rotated_x = corner_x * cos_h - corner_y * sin_h + self.x
            rotated_y = corner_x * sin_h + corner_y * cos_h + self.y
            rotated_corners.append((rotated_x, rotated_y))
        
        # Color based on behavior - BRIGHT colors for visibility
        colors = {
            'normal': (50, 255, 50),      # Bright green
            'aggressive': (255, 50, 50),   # Bright red  
            'conservative': (50, 150, 255), # Bright blue
            'cooperative': (255, 255, 50)   # Bright yellow
        }
        color = colors.get(self.behavior_mode, (200, 200, 200))
        
        # Override color if badly off-lane
        if abs(self.lane_deviation()) > 3.0:
            color = (255, 80, 80)  # Red for poor lane keeping
        
        pygame.draw.polygon(screen, color, rotated_corners)
        
        # Direction indicator
        front_x = self.x + cos_h * length/2
        front_y = self.y + sin_h * length/2
        pygame.draw.circle(screen, (255, 255, 255), (int(front_x), int(front_y)), 2)
        
        # Collision risk indicator
        if self.collision_risk > 0.3:
            risk_color = (255, int(255 * (1 - self.collision_risk)), 0)
            pygame.draw.circle(screen, risk_color, (int(self.x), int(self.y)), 
                             int(8 + self.collision_risk * 10), 2)

# ============================================================================
# TRAINING VISUALIZATION
# ============================================================================

class TrainingVisualization:
    """Real-time training visualization in pygame"""
    def __init__(self, width: int = 1400, height: int = 900):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🧠 Autonomous Vehicle Training - Live Neural Network Learning")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.title_font = pygame.font.Font(None, 36)
        self.header_font = pygame.font.Font(None, 28)
        self.text_font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
        # Layout areas
        self.sim_area = pygame.Rect(20, 60, 800, 600)
        self.metrics_area = pygame.Rect(840, 60, 540, 300)
        self.progress_area = pygame.Rect(840, 380, 540, 280)
        self.controls_area = pygame.Rect(20, 680, 1360, 200)
        
        # Training data
        self.episode_history = {'rewards': [], 'throughput': [], 'loss': [], 'epsilon': []}
        self.current_episode = 0
        self.total_episodes = 0
        self.episode_start_time = 0
        self.training_start_time = 0
        
        # Real-time metrics
        self.current_metrics = {
            'reward': 0, 'throughput': 0, 'loss': 0, 'epsilon': 1.0,
            'vehicles': 0, 'cooperation': 0, 'collision_risk': 0, 'lane_dev': 0
        }
        
        # Intersections for mini-simulation - properly scaled
        scale_x = self.sim_area.width / 1200
        scale_y = self.sim_area.height / 800
        
        self.intersections = [
            {'x': self.sim_area.x + int(400 * scale_x), 'y': self.sim_area.y + int(300 * scale_y), 'occupancy': 0},
            {'x': self.sim_area.x + int(800 * scale_x), 'y': self.sim_area.y + int(300 * scale_y), 'occupancy': 0},
            {'x': self.sim_area.x + int(400 * scale_x), 'y': self.sim_area.y + int(500 * scale_y), 'occupancy': 0},
            {'x': self.sim_area.x + int(800 * scale_x), 'y': self.sim_area.y + int(500 * scale_y), 'occupancy': 0}
        ]
        
        self.running = True
        
    def start_training(self, total_episodes: int):
        """Initialize training session"""
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.training_start_time = time.time()
        self.episode_history = {'rewards': [], 'throughput': [], 'loss': [], 'epsilon': []}
        
    def update_episode(self, episode: int, results: Dict, epsilon: float):
        """Update with episode results"""
        self.current_episode = episode
        self.episode_history['rewards'].append(results['avg_reward'])
        self.episode_history['throughput'].append(results['throughput'])
        self.episode_history['loss'].append(results['avg_loss'])
        self.episode_history['epsilon'].append(epsilon)
        
        self.current_metrics.update({
            'reward': results['avg_reward'],
            'throughput': results['throughput'],
            'loss': results['avg_loss'],
            'epsilon': epsilon
        })
        
    def update_realtime_metrics(self, vehicles: List[AutonomousVehicle], intersections: List):
        """Update real-time simulation metrics"""
        if vehicles:
            self.current_metrics.update({
                'vehicles': len(vehicles),
                'cooperation': sum(1 for v in vehicles if v.behavior_mode == 'cooperative') / len(vehicles),
                'collision_risk': np.mean([v.collision_risk for v in vehicles]),
                'lane_dev': np.mean([abs(v.lane_deviation()) for v in vehicles])
            })
            
            # Update intersection occupancy for visualization
            for i, intersection in enumerate(intersections):
                if i < len(self.intersections):
                    self.intersections[i]['occupancy'] = intersection.get('occupancy', 0)
    
    def draw_simulation_area(self, vehicles: List[AutonomousVehicle] = None):
        """Draw mini simulation visualization"""
        # Background
        pygame.draw.rect(self.screen, (30, 30, 40), self.sim_area)
        pygame.draw.rect(self.screen, (100, 100, 100), self.sim_area, 2)
        
        # Title
        title = self.header_font.render("🚗 Live Training Simulation", True, (255, 255, 255))
        self.screen.blit(title, (self.sim_area.x + 10, self.sim_area.y - 35))
        
        # Roads - properly centered and aligned
        road_color = (60, 60, 75)
        road_width = 60
        
        # Horizontal roads (scaled to match actual coordinates)
        h_road_y1 = self.sim_area.y + int(300 * (self.sim_area.height / 800)) - road_width//2
        h_road_y2 = self.sim_area.y + int(500 * (self.sim_area.height / 800)) - road_width//2
        
        pygame.draw.rect(self.screen, road_color, 
                        (self.sim_area.x, h_road_y1, self.sim_area.width, road_width))
        pygame.draw.rect(self.screen, road_color, 
                        (self.sim_area.x, h_road_y2, self.sim_area.width, road_width))
        
        # Vertical roads (scaled to match actual coordinates)
        v_road_x1 = self.sim_area.x + int(400 * (self.sim_area.width / 1200)) - road_width//2
        v_road_x2 = self.sim_area.x + int(800 * (self.sim_area.width / 1200)) - road_width//2
        
        pygame.draw.rect(self.screen, road_color, 
                        (v_road_x1, self.sim_area.y, road_width, self.sim_area.height))
        pygame.draw.rect(self.screen, road_color, 
                        (v_road_x2, self.sim_area.y, road_width, self.sim_area.height))
        
        # Intersections
        for intersection in self.intersections:
            occupancy = intersection.get('occupancy', 0)
            if occupancy > 3:
                color = (120, 40, 40)   # Dark red - congested
            elif occupancy > 1:
                color = (120, 80, 40)   # Dark orange - moderate
            elif occupancy > 0:
                color = (100, 100, 40)  # Dark yellow - light traffic
            else:
                color = (40, 80, 40)    # Dark green - clear (much darker!)
                
            pygame.draw.rect(self.screen, color, 
                           (intersection['x'] - 30, intersection['y'] - 30, 60, 60))
            
            # Occupancy number
            # Only show text for high occupancy (performance optimization)
            if occupancy > 2:
                occ_text = self.small_font.render(str(occupancy), True, (0, 0, 0))
                text_rect = occ_text.get_rect(center=(intersection['x'], intersection['y']))
                self.screen.blit(occ_text, text_rect)
        
        # Scale factors for positioning elements in simulation area
        scale_x = self.sim_area.width / 1200
        scale_y = self.sim_area.height / 800
        
        # Lane centre dashed lines (optional visual aid)
        dash_color = (160, 160, 180)
        dash_len = int(8 * scale_x)
        gap = int(12 * scale_x)
        # horizontal lanes
        for y_lane in [300, 500]:
            y = self.sim_area.y + int(y_lane * scale_y)
            x0 = self.sim_area.x
            while x0 < self.sim_area.right:
                pygame.draw.line(self.screen, dash_color,
                                 (x0, y), (min(x0+dash_len, self.sim_area.right), y), 1)
                x0 += dash_len + gap
        # vertical lanes
        for x_lane in [400, 800]:
            x = self.sim_area.x + int(x_lane * scale_x)
            y0 = self.sim_area.y
            while y0 < self.sim_area.bottom:
                pygame.draw.line(self.screen, dash_color,
                                 (x, y0), (x, min(y0+dash_len, self.sim_area.bottom)), 1)
                y0 += dash_len + gap
        
        # Draw sample vehicles if provided
        if vehicles and len(vehicles) > 0:
            for i, vehicle in enumerate(vehicles[:20]):  # Show max 20 vehicles
                # Scale position
                scaled_x = self.sim_area.x + vehicle.x * scale_x
                scaled_y = self.sim_area.y + vehicle.y * scale_y
                
                # Vehicle color based on behavior - BRIGHT colors for visibility
                colors = {
                    'normal': (50, 255, 50),      # Bright green
                    'aggressive': (255, 50, 50),   # Bright red  
                    'conservative': (50, 150, 255), # Bright blue
                    'cooperative': (255, 255, 50)   # Bright yellow
                }
                color = colors.get(vehicle.behavior_mode, (200, 200, 200))
                
                # Override color if badly off-lane
                if abs(vehicle.lane_deviation()) > 3.0:
                    color = (255, 80, 80)  # Red for poor lane keeping
                
                # Draw as larger, more visible circle
                pygame.draw.circle(self.screen, color, (int(scaled_x), int(scaled_y)), 6)
                # Add white outline for maximum visibility
                pygame.draw.circle(self.screen, (255, 255, 255), (int(scaled_x), int(scaled_y)), 6, 1)
                
                # Collision risk indicator
                if vehicle.collision_risk > 0.3:
                    risk_color = (255, int(255 * (1 - vehicle.collision_risk)), 0)
                    pygame.draw.circle(self.screen, risk_color, 
                                     (int(scaled_x), int(scaled_y)), 
                                     int(6 + vehicle.collision_risk * 8), 2)
    
    def draw_metrics_panel(self):
        """Draw real-time metrics"""
        # Background
        pygame.draw.rect(self.screen, (40, 40, 50), self.metrics_area)
        pygame.draw.rect(self.screen, (100, 100, 100), self.metrics_area, 2)
        
        # Title
        title = self.header_font.render("📊 Training Metrics", True, (255, 255, 255))
        self.screen.blit(title, (self.metrics_area.x + 10, self.metrics_area.y - 35))
        
        y_offset = self.metrics_area.y + 20
        line_height = 25
        
        # Episode progress
        progress_pct = (self.current_episode / max(1, self.total_episodes)) * 100
        episode_text = f"Episode: {self.current_episode}/{self.total_episodes} ({progress_pct:.1f}%)"
        text = self.text_font.render(episode_text, True, (255, 255, 255))
        self.screen.blit(text, (self.metrics_area.x + 15, y_offset))
        y_offset += line_height
        
        # Progress bar
        bar_width = 400
        bar_height = 20
        progress_width = int((self.current_episode / max(1, self.total_episodes)) * bar_width)
        
        pygame.draw.rect(self.screen, (60, 60, 60), 
                        (self.metrics_area.x + 15, y_offset, bar_width, bar_height))
        pygame.draw.rect(self.screen, (100, 255, 100), 
                        (self.metrics_area.x + 15, y_offset, progress_width, bar_height))
        y_offset += line_height + 10
        
        # Current metrics
        metrics_data = [
            ("🎯 Reward", f"{self.current_metrics['reward']:6.2f}", (100, 255, 100)),
            ("🚗 Throughput", f"{self.current_metrics['throughput']:3d} vehicles", (100, 200, 255)),
            ("📉 Loss", f"{self.current_metrics['loss']:6.4f}", (255, 150, 100)),
            ("🎲 Exploration", f"{self.current_metrics['epsilon']:.1%}", (255, 255, 100)),
            ("👥 Active Vehicles", f"{self.current_metrics['vehicles']:2d}", (200, 150, 255)),
            ("🤝 Cooperation", f"{self.current_metrics['cooperation']:.1%}", (255, 200, 100)),
            ("⚠️  Collision Risk", f"{self.current_metrics['collision_risk']:.1%}", (255, 100, 100)),
            ("🛣️  Lane Dev.", f"{self.current_metrics['lane_dev']:.2f} m", (150, 200, 255))
        ]
        
        for label, value, color in metrics_data:
            label_text = self.text_font.render(label, True, (200, 200, 200))
            value_text = self.text_font.render(value, True, color)
            
            self.screen.blit(label_text, (self.metrics_area.x + 15, y_offset))
            self.screen.blit(value_text, (self.metrics_area.x + 200, y_offset))
            y_offset += line_height
        
        # Time stats
        if self.training_start_time > 0:
            elapsed = time.time() - self.training_start_time
            estimated_total = (elapsed / max(1, self.current_episode)) * self.total_episodes
            remaining = max(0, estimated_total - elapsed)
            
            time_text = f"⏱️ Elapsed: {elapsed/60:.1f}m | Remaining: {remaining/60:.1f}m"
            text = self.small_font.render(time_text, True, (150, 150, 150))
            self.screen.blit(text, (self.metrics_area.x + 15, y_offset))
    
    def draw_progress_graphs(self):
        """Draw mini progress graphs"""
        # Background
        pygame.draw.rect(self.screen, (40, 40, 50), self.progress_area)
        pygame.draw.rect(self.screen, (100, 100, 100), self.progress_area, 2)
        
        # Title
        title = self.header_font.render("📈 Learning Progress", True, (255, 255, 255))
        self.screen.blit(title, (self.progress_area.x + 10, self.progress_area.y - 35))
        
        if len(self.episode_history['rewards']) < 2:
            no_data_text = self.text_font.render("Collecting training data...", True, (150, 150, 150))
            self.screen.blit(no_data_text, (self.progress_area.x + 50, self.progress_area.y + 100))
            return
        
        # Graph area
        graph_width = self.progress_area.width - 40
        graph_height = (self.progress_area.height - 60) // 2
        
        # Rewards graph
        reward_area = pygame.Rect(self.progress_area.x + 20, self.progress_area.y + 20, 
                                 graph_width, graph_height)
        self.draw_mini_graph(reward_area, self.episode_history['rewards'], 
                           "Reward", (100, 255, 100))
        
        # Throughput graph
        throughput_area = pygame.Rect(self.progress_area.x + 20, 
                                    self.progress_area.y + graph_height + 40, 
                                    graph_width, graph_height)
        self.draw_mini_graph(throughput_area, self.episode_history['throughput'], 
                           "Throughput", (100, 200, 255))
    
    def draw_mini_graph(self, area: pygame.Rect, data: List[float], title: str, color: Tuple[int, int, int]):
        """Draw a mini line graph"""
        if len(data) < 2:
            return
            
        # Graph background
        pygame.draw.rect(self.screen, (25, 25, 30), area)
        pygame.draw.rect(self.screen, (80, 80, 80), area, 1)
        
        # Title
        title_text = self.small_font.render(title, True, (200, 200, 200))
        self.screen.blit(title_text, (area.x + 5, area.y + 5))
        
        # Data points
        if len(data) > 1:
            # Normalize data
            data_min = min(data)
            data_max = max(data)
            data_range = max(data_max - data_min, 1e-6)
            
            # Draw line
            points = []
            for i, value in enumerate(data[-min(50, len(data)):]):  # Show last 50 points
                x = area.x + 10 + (i / max(1, len(data[-50:]) - 1)) * (area.width - 20)
                y = area.y + area.height - 30 - ((value - data_min) / data_range) * (area.height - 50)
                points.append((int(x), int(y)))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
            
            # Current value
            current_text = self.small_font.render(f"{data[-1]:.2f}", True, color)
            self.screen.blit(current_text, (area.x + area.width - 60, area.y + 5))
    
    def draw_controls_help(self):
        """Draw control instructions and training status"""
        # Background
        pygame.draw.rect(self.screen, (20, 20, 30), self.controls_area)
        pygame.draw.rect(self.screen, (100, 100, 100), self.controls_area, 2)
        
        # Title
        title = self.header_font.render("🎮 Training Controls & Status", True, (255, 255, 255))
        self.screen.blit(title, (self.controls_area.x + 10, self.controls_area.y - 35))
        
        y_offset = self.controls_area.y + 20
        
        # Controls
        controls = [
            "ESC = Stop training early",
            "SPACE = Pause/Resume (if implemented)",
            "S = Save current model checkpoint"
        ]
        
        controls_text = self.text_font.render("Controls: " + " | ".join(controls), True, (200, 200, 200))
        self.screen.blit(controls_text, (self.controls_area.x + 15, y_offset))
        y_offset += 30
        
        # Learning phase description
        if self.current_episode <= 20:
            phase = "🌱 EXPLORATION PHASE: Vehicles learning basic coordination"
            phase_color = (255, 200, 100)
        elif self.current_episode <= 50:
            phase = "🧠 LEARNING PHASE: Discovering intersection patterns"
            phase_color = (100, 200, 255)
        elif self.current_episode <= 80:
            phase = "⚡ OPTIMIZATION PHASE: Refining coordination strategies"
            phase_color = (100, 255, 100)
        else:
            phase = "🎯 MASTERY PHASE: Perfecting autonomous coordination"
            phase_color = (255, 100, 255)
        
        phase_text = self.text_font.render(phase, True, phase_color)
        self.screen.blit(phase_text, (self.controls_area.x + 15, y_offset))
        y_offset += 30
        
        # Performance assessment
        if len(self.episode_history['rewards']) >= 10:
            recent_avg = np.mean(self.episode_history['rewards'][-10:])
            if recent_avg > 10:
                assessment = "🔥 EXCELLENT learning progress!"
                assess_color = (100, 255, 100)
            elif recent_avg > 5:
                assessment = "✅ GOOD learning progress"
                assess_color = (150, 255, 150)
            elif recent_avg > 0:
                assessment = "⚠️  MODERATE learning progress"
                assess_color = (255, 255, 100)
            else:
                assessment = "🐌 SLOW learning - may need more time"
                assess_color = (255, 150, 100)
            
            assess_text = self.text_font.render(assessment, True, assess_color)
            self.screen.blit(assess_text, (self.controls_area.x + 15, y_offset))
            y_offset += 30
        
        # Neural network insights
        insights = []
        if self.current_metrics['epsilon'] < 0.1:
            insights.append("🎯 High exploitation - using learned strategies")
        elif self.current_metrics['epsilon'] > 0.5:
            insights.append("🎲 High exploration - discovering new behaviors")
        
        if self.current_metrics['cooperation'] > 0.3:
            insights.append("🤝 Vehicles learning to cooperate!")
        
        if self.current_metrics['collision_risk'] < 0.2:
            insights.append("✅ Low collision risk - safe coordination")
        
        if insights:
            insight_text = self.small_font.render("Insights: " + " | ".join(insights), True, (150, 255, 150))
            self.screen.blit(insight_text, (self.controls_area.x + 15, y_offset))
    
    def handle_events(self) -> bool:
        """Handle pygame events, return False to stop training"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_s:
                    # Save checkpoint (would be handled by trainer)
                    print("💾 Checkpoint save requested...")
        return True
    
    def update_display(self, vehicles: List[AutonomousVehicle] = None, intersections: List = None):
        """Update the entire display"""
        # Clear screen
        self.screen.fill((15, 15, 20))
        
        # Title
        title = self.title_font.render("🧠 Autonomous Vehicle Neural Network Training", True, (255, 255, 255))
        title_rect = title.get_rect(center=(self.screen.get_width() // 2, 30))
        self.screen.blit(title, title_rect)
        
        # Update real-time metrics
        if vehicles and intersections:
            self.update_realtime_metrics(vehicles, intersections)
        
        # Draw all panels
        self.draw_simulation_area(vehicles)
        self.draw_metrics_panel()
        self.draw_progress_graphs()
        self.draw_controls_help()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS for smooth autonomous vehicle visualization
    
    def close(self):
        """Clean up pygame"""
        pygame.quit()

# ============================================================================
# PANDA3D 3D VISUALIZATION SYSTEM
# ============================================================================

class Panda3DVisualization(ShowBase):
    """Immersive 3D visualization using Panda3D"""
    def __init__(self, width: int = 1600, height: int = 900):
        if not PANDA3D_AVAILABLE:
            raise ImportError("Panda3D not available. Install with: pip install panda3d")
        
        ShowBase.__init__(self)
        
        # Initialize attributes first before calling any methods that use them
        self.vehicle_nodes = {}  # Maps vehicle IDs to NodePaths
        self.intersection_nodes = []
        
        # Window configuration
        props = WindowProperties()
        props.setTitle("🚗 Autonomous Vehicle 3D Simulation - Neural Network Training")
        props.setSize(width, height)
        self.win.requestProperties(props)
        
        # Camera setup - bird's eye view for traffic monitoring
        self.camera.setPos(600, -800, 400)  # Position above and behind the scene
        self.camera.lookAt(600, 400, 0)     # Look at center of intersection area
        
        # Training metrics
        self.current_episode = 0
        self.total_episodes = 0
        self.training_metrics = {
            'reward': 0, 'throughput': 0, 'loss': 0, 'epsilon': 0,
            'vehicles': 0, 'cooperation': 0, 'collision_risk': 0
        }
        
        # Enhanced lighting for realistic 3D environment
        self.setup_lighting()
        
        # Create 3D environment
        self.setup_environment()
        
        # UI elements
        self.setup_ui()
        
        # Animation and update tasks
        self.taskMgr.add(self.update_scene, "update_scene")
        
        # Camera controls
        self.setup_camera_controls()
        
    def setup_lighting(self):
        """Create realistic lighting for the 3D scene"""
        # Ambient light for overall illumination
        ambient_light = AmbientLight('ambient')
        ambient_light.setColor((0.3, 0.3, 0.4, 1))
        ambient_np = self.render.attachNewNode(ambient_light)
        self.render.setLight(ambient_np)
        
        # Directional light (sun) for shadows and depth
        sun_light = DirectionalLight('sun')
        sun_light.setColor((0.8, 0.8, 0.7, 1))
        sun_light.setDirection(Vec3(-1, -1, -0.5))
        sun_np = self.render.attachNewNode(sun_light)
        sun_np.setPos(600, 400, 500)
        self.render.setLight(sun_np)
        
        # Additional point lights at intersections for better visibility
        for i, pos in enumerate([(400, 300), (800, 300), (400, 500), (800, 500)]):
            point_light = PointLight(f'intersection_light_{i}')
            point_light.setColor((0.4, 0.4, 0.5, 1))
            point_np = self.render.attachNewNode(point_light)
            point_np.setPos(pos[0], pos[1], 50)
            self.render.setLight(point_np)
        
        # Atmospheric fog for depth perception
        fog = Fog('atmosphere')
        fog.setColor(0.8, 0.8, 0.9)
        fog.setExpDensity(0.0008)
        self.render.setFog(fog)
    
    def setup_environment(self):
        """Create the 3D road environment"""
        # Ground plane
        ground = self.create_ground_plane()
        ground.reparentTo(self.render)
        
        # Road system
        self.create_road_system()
        
        # Intersections
        self.create_intersections()
        
        # Environmental elements
        self.create_environment_details()
    
    def create_ground_plane(self):
        """Create a textured ground plane"""
        cm = CardMaker('ground')
        cm.setFrame(-100, 1300, -100, 900)  # Large ground area
        ground = self.render.attachNewNode(cm.generate())
        ground.setP(-90)  # Rotate to be horizontal
        ground.setColor(0.2, 0.4, 0.2, 1)  # Dark green grass
        
        # Add subtle texture variation
        ground.setTexScale(TextureStage.getDefault(), 50, 50)
        return ground
    
    def create_road_system(self):
        """Create 3D road network"""
        # Road material properties
        road_color = (0.3, 0.3, 0.35, 1)
        lane_line_color = (0.9, 0.9, 0.9, 1)
        
        # Horizontal roads
        for y in [300, 500]:
            road = self.create_road_segment(0, y-30, 1200, 60, road_color)
            road.reparentTo(self.render)
            
            # Lane markings
            for x in range(50, 1200, 40):
                marking = self.create_lane_marking(x, y, 20, 2, lane_line_color)
                marking.reparentTo(self.render)
        
        # Vertical roads  
        for x in [400, 800]:
            road = self.create_road_segment(x-30, 0, 60, 800, road_color)
            road.reparentTo(self.render)
            
            # Lane markings
            for y in range(50, 800, 40):
                marking = self.create_lane_marking(x, y, 2, 20, lane_line_color)
                marking.reparentTo(self.render)
    
    def create_road_segment(self, x, y, width, height, color):
        """Create a single road segment"""
        cm = CardMaker('road')
        cm.setFrame(0, width, 0, height)
        road = self.render.attachNewNode(cm.generate())
        road.setPos(x, y, 0.1)  # Slightly above ground
        road.setColor(*color)
        road.setP(-90)  # Make horizontal
        return road
    
    def create_lane_marking(self, x, y, width, height, color):
        """Create lane markings"""
        cm = CardMaker('lane_marking')
        cm.setFrame(-width/2, width/2, -height/2, height/2)
        marking = self.render.attachNewNode(cm.generate())
        marking.setPos(x, y, 0.2)  # Above road surface
        marking.setColor(*color)
        marking.setP(-90)
        return marking
    
    def create_intersections(self):
        """Create 3D intersection visualizations"""
        intersection_positions = [(400, 300), (800, 300), (400, 500), (800, 500)]
        
        for i, (x, y) in enumerate(intersection_positions):
            # Intersection platform
            cm = CardMaker(f'intersection_{i}')
            cm.setFrame(-40, 40, -40, 40)
            intersection = self.render.attachNewNode(cm.generate())
            intersection.setPos(x, y, 0.15)
            intersection.setColor(0.4, 0.4, 0.4, 1)  # Darker intersection color
            intersection.setP(-90)
            
            # Traffic coordination indicator (will change color based on occupancy)
            indicator = self.create_traffic_indicator(x, y)
            indicator.reparentTo(self.render)
            
            self.intersection_nodes.append({
                'node': intersection,
                'indicator': indicator,
                'position': (x, y),
                'occupancy': 0
            })
    
    def create_traffic_indicator(self, x, y):
        """Create traffic coordination status indicator"""
        cm = CardMaker('indicator')
        cm.setFrame(-5, 5, -5, 5)
        indicator = self.render.attachNewNode(cm.generate())
        indicator.setPos(x, y, 20)  # Elevated above intersection
        indicator.setColor(0, 1, 0, 0.7)  # Green = clear
        indicator.setTransparency(TransparencyAttrib.MAlpha)
        indicator.setP(-90)
        indicator.setBillboardAxis()  # Always face camera
        return indicator
    
    def create_environment_details(self):
        """Add environmental details for immersion"""
        # Simple buildings/structures around the road network
        building_positions = [
            (200, 150, 30), (600, 150, 25), (1000, 150, 35),
            (200, 650, 28), (600, 650, 32), (1000, 650, 30),
            (150, 400, 20), (1050, 400, 25)
        ]
        
        for i, (x, y, height) in enumerate(building_positions):
            building = self.create_building(x, y, height)
            building.reparentTo(self.render)
    
    def create_building(self, x, y, height):
        """Create simple building structures"""
        cm = CardMaker('building')
        cm.setFrame(-15, 15, 0, height)
        building = self.render.attachNewNode(cm.generate())
        building.setPos(x, y, 0)
        building.setColor(0.6, 0.6, 0.7, 1)
        building.lookAt(600, 400, 0)  # Face toward center
        return building
    
    def create_vehicle_model(self, vehicle_id: int):
        """Create 3D vehicle model"""
        # Create vehicle body
        cm = CardMaker(f'vehicle_{vehicle_id}')
        cm.setFrame(-6, 6, -3, 3)  # 12x6 meter vehicle
        vehicle_node = self.render.attachNewNode(cm.generate())
        vehicle_node.setColor(0.8, 0.2, 0.2, 1)  # Red vehicle
        vehicle_node.setP(-90)  # Make horizontal
        vehicle_node.setZ(1)  # Slightly above road
        
        # Add vehicle details
        # Windshield
        windshield = CardMaker('windshield')
        windshield.setFrame(-4, 4, -2, 2)
        windshield_node = vehicle_node.attachNewNode(windshield.generate())
        windshield_node.setPos(0, 0, 2)
        windshield_node.setColor(0.2, 0.2, 0.4, 0.8)
        windshield_node.setTransparency(TransparencyAttrib.MAlpha)
        
        # Direction indicator
        indicator = CardMaker('direction')
        indicator.setFrame(-1, 1, -1, 1)
        indicator_node = vehicle_node.attachNewNode(indicator.generate())
        indicator_node.setPos(5, 0, 1)  # Front of vehicle
        indicator_node.setColor(1, 1, 1, 1)  # White
        
        return vehicle_node
    
    def setup_ui(self):
        """Setup 3D UI elements"""
        # Title
        self.title_text = OnscreenText(
            text="🚗 Autonomous Vehicle 3D Neural Network Training",
            pos=(0, 0.9), scale=0.08, fg=(1, 1, 1, 1),
            align=TextNode.ACenter, mayChange=True
        )
        
        # Training metrics display
        self.metrics_text = OnscreenText(
            text="", pos=(-0.95, 0.8), scale=0.05, fg=(1, 1, 1, 1),
            align=TextNode.ALeft, mayChange=True
        )
        
        # Episode progress
        self.progress_text = OnscreenText(
            text="", pos=(-0.95, 0.6), scale=0.05, fg=(0.8, 1, 0.8, 1),
            align=TextNode.ALeft, mayChange=True
        )
        
        # Performance stats
        self.performance_text = OnscreenText(
            text="", pos=(-0.95, 0.4), scale=0.05, fg=(1, 0.8, 0.8, 1),
            align=TextNode.ALeft, mayChange=True
        )
        
        # Controls help
        self.controls_text = OnscreenText(
            text="Controls: Mouse = Rotate Camera | Wheel = Zoom | ESC = Exit",
            pos=(0, -0.9), scale=0.04, fg=(0.8, 0.8, 0.8, 1),
            align=TextNode.ACenter, mayChange=False
        )
    
    def setup_camera_controls(self):
        """Setup interactive camera controls"""
        # Mouse camera control
        self.camera_pivot = self.render.attachNewNode("camera_pivot")
        self.camera_pivot.setPos(600, 400, 0)  # Center of scene
        
        # Accept mouse events
        self.accept("mouse1", self.start_camera_rotate)
        self.accept("mouse1-up", self.stop_camera_rotate)
        self.accept("wheel_up", self.zoom_in)
        self.accept("wheel_down", self.zoom_out)
        
        # Accept escape key for training mode
        self.accept("escape", self.request_exit)
        
        self.camera_rotating = False
        self.last_mouse_pos = None
        self._escape_pressed = False
    
    def start_camera_rotate(self):
        """Start camera rotation"""
        self.camera_rotating = True
        self.last_mouse_pos = None
    
    def stop_camera_rotate(self):
        """Stop camera rotation"""
        self.camera_rotating = False
    
    def zoom_in(self):
        """Zoom camera in"""
        current_pos = self.camera.getPos()
        center = Vec3(600, 400, 0)
        direction = current_pos - center
        direction.normalize()
        new_pos = current_pos - direction * 50
        self.camera.setPos(new_pos)
    
    def zoom_out(self):
        """Zoom camera out"""
        current_pos = self.camera.getPos()
        center = Vec3(600, 400, 0)
        direction = current_pos - center
        direction.normalize()
        new_pos = current_pos + direction * 50
        self.camera.setPos(new_pos)
    
    def request_exit(self):
        """Request exit from training/simulation"""
        self._escape_pressed = True
        print("🛑 Exit requested via ESC key...")
    
    def update_vehicles(self, vehicles: List):
        """Update 3D vehicle positions and states"""
        current_vehicle_ids = set(v.id for v in vehicles)
        
        # Remove vehicles that no longer exist
        for vehicle_id in list(self.vehicle_nodes.keys()):
            if vehicle_id not in current_vehicle_ids:
                self.vehicle_nodes[vehicle_id].removeNode()
                del self.vehicle_nodes[vehicle_id]
        
        # Update existing vehicles and create new ones
        for vehicle in vehicles:
            if vehicle.id not in self.vehicle_nodes:
                # Create new vehicle
                self.vehicle_nodes[vehicle.id] = self.create_vehicle_model(vehicle.id)
            
            # Update vehicle position and orientation
            vehicle_node = self.vehicle_nodes[vehicle.id]
            vehicle_node.setPos(vehicle.x, vehicle.y, 1)
            vehicle_node.setH(math.degrees(-vehicle.heading) + 90)  # Convert to Panda3D heading
            
            # Color based on behavior and state
            if vehicle.behavior_mode == 'cooperative':
                color = (0.2, 0.8, 0.2, 1)  # Green for cooperative
            elif vehicle.collision_risk > 0.5:
                color = (1, 0.5, 0, 1)  # Orange for high risk
            elif vehicle.speed < 5:
                color = (0.8, 0.8, 0.2, 1)  # Yellow for slow
            else:
                color = (0.2, 0.5, 1, 1)  # Blue for normal
            
            vehicle_node.setColor(*color)
            
            # Scale based on speed (visual feedback)
            speed_scale = 1.0 + (vehicle.speed / 45.0) * 0.3
            vehicle_node.setScale(speed_scale, 1, 1)
    
    def update_intersections(self, intersections: List):
        """Update intersection status indicators"""
        for i, intersection_data in enumerate(intersections):
            if i < len(self.intersection_nodes):
                # Calculate occupancy
                occupancy = intersection_data.get('occupancy', 0)
                
                # Update indicator color based on traffic
                indicator = self.intersection_nodes[i]['indicator']
                if occupancy > 3:
                    indicator.setColor(1, 0, 0, 0.8)  # Red - congested
                elif occupancy > 1:
                    indicator.setColor(1, 0.5, 0, 0.8)  # Orange - moderate
                elif occupancy > 0:
                    indicator.setColor(1, 1, 0, 0.8)  # Yellow - light traffic
                else:
                    indicator.setColor(0, 1, 0, 0.7)  # Green - clear
                
                # Animate indicator based on activity
                scale = 1.0 + occupancy * 0.2
                indicator.setScale(scale)
    
    def update_metrics_display(self):
        """Update on-screen metrics"""
        # Training progress
        progress_pct = (self.current_episode / max(1, self.total_episodes)) * 100
        
        metrics_text = f"""Episode: {self.current_episode}/{self.total_episodes} ({progress_pct:.1f}%)
Reward: {self.training_metrics['reward']:6.2f}
Throughput: {self.training_metrics['throughput']:3d} vehicles
Loss: {self.training_metrics['loss']:6.4f}
Active Vehicles: {self.training_metrics['vehicles']:2d}"""
        
        self.metrics_text.setText(metrics_text)
        
        # Performance assessment
        if self.training_metrics['throughput'] > 40:
            perf_status = "🔥 EXCELLENT Performance"
            perf_color = (0.2, 1, 0.2, 1)
        elif self.training_metrics['throughput'] > 25:
            perf_status = "✅ GOOD Performance"
            perf_color = (0.8, 1, 0.2, 1)
        elif self.training_metrics['throughput'] > 15:
            perf_status = "⚠️ LEARNING Phase"
            perf_color = (1, 1, 0.2, 1)
        else:
            perf_status = "🐌 NEEDS TRAINING"
            perf_color = (1, 0.5, 0.2, 1)
        
        performance_text = f"""{perf_status}
Cooperation: {self.training_metrics['cooperation']:.1%}
Collision Risk: {self.training_metrics['collision_risk']:.1%}
Neural Network Learning Active"""
        
        self.performance_text.setText(performance_text)
        self.performance_text.setFg(perf_color)
        
        # Learning phase
        if self.current_episode <= 20:
            phase = "🌱 EXPLORATION: Discovering coordination patterns"
        elif self.current_episode <= 50:
            phase = "🧠 LEARNING: Optimizing intersection strategies"
        elif self.current_episode <= 80:
            phase = "⚡ REFINEMENT: Perfecting coordination"
        else:
            phase = "🎯 MASTERY: Expert autonomous coordination"
        
        self.progress_text.setText(phase)
    
    def update_scene(self, task):
        """Main scene update task"""
        # Update UI
        self.update_metrics_display()
        
        # Handle camera rotation
        if self.camera_rotating and self.mouseWatcherNode.hasMouse():
            mouse_pos = (self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY())
            if self.last_mouse_pos:
                dx = mouse_pos[0] - self.last_mouse_pos[0]
                dy = mouse_pos[1] - self.last_mouse_pos[1]
                
                # Rotate camera around scene center
                current_pos = self.camera.getPos()
                center = Vec3(600, 400, 0)
                
                # Horizontal rotation
                angle_h = dx * 180
                self.camera.setPos(center)
                self.camera.setH(self.camera.getH() + angle_h)
                
                # Move camera back to distance
                self.camera.setPos(current_pos)
                
            self.last_mouse_pos = mouse_pos
        
        return task.cont
    
    def start_training(self, total_episodes: int):
        """Initialize training visualization"""
        self.total_episodes = total_episodes
        self.current_episode = 0
        print("🎮 3D Training visualization started!")
    
    def update_episode(self, episode: int, results: Dict, epsilon: float = 0.0):
        """Update with episode results"""
        self.current_episode = episode
        self.training_metrics.update({
            'reward': results['avg_reward'],
            'throughput': results['throughput'],
            'loss': results['avg_loss'],
            'epsilon': epsilon
        })
    
    def update_realtime_metrics(self, vehicles: List, intersections: List):
        """Update real-time simulation metrics"""
        if vehicles:
            self.training_metrics.update({
                'vehicles': len(vehicles),
                'cooperation': sum(1 for v in vehicles if v.behavior_mode == 'cooperative') / len(vehicles),
                'collision_risk': np.mean([v.collision_risk for v in vehicles])
            })
        
        # Update 3D scene
        self.update_vehicles(vehicles)
        self.update_intersections(intersections)
    
    def handle_events(self) -> bool:
        """Handle input events, return False to stop training"""
        # Check for ESC key to exit (only if explicitly requested)
        if hasattr(self, 'userExit') and self.userExit:
            return False
        
        # Check for explicit escape key press
        if hasattr(self, '_escape_pressed') and self._escape_pressed:
            return False
            
        return True
    
    def update_display(self, vehicles: List = None, intersections: List = None):
        """Update the 3D display"""
        if vehicles and intersections:
            self.update_realtime_metrics(vehicles, intersections)
        
        # Process Panda3D events without blocking
        try:
            # Run one frame of Panda3D's task manager
            self.taskMgr.step()
        except:
            # If there's an issue with the window, don't crash the training
            pass
    
    def close(self):
        """Clean up Panda3D"""
        self.userExit = True

# ============================================================================
# TRAINING SYSTEM (UPDATED)
# ============================================================================

class TrafficTrainer:
    """Real GRPO-based training system with GPU optimization"""
    def __init__(self, use_advanced_ml: bool = True):
        if use_advanced_ml and not ADVANCED_ML_AVAILABLE:
            raise ImportError("Install torch_geometric for advanced ML: pip install torch_geometric")
        
        # Choose the right policy network
        if use_advanced_ml:
            self.policy_net = AdvancedTrafficTransformer(continuous_actions=True)
            # Initialize action log_std properly for continuous actions
            if hasattr(self.policy_net, 'action_log_std'):
                nn.init.constant_(self.policy_net.action_log_std.bias, -1.0)
            
            # Optimized learning rate and optimizer for GPU
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.001, weight_decay=0.01)
            self.grpo = GRPO(self.policy_net, self.optimizer)
            self.use_advanced_ml = True
            
            # Set to GPU if available with optimizations
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.policy_net.to(self.device)
            
            # GPU optimizations
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"🎮 Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                
                # Enable GPU optimizations
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat-32 for faster training
                torch.backends.cudnn.allow_tf32 = True
                
                # Enable mixed precision training for RTX cards
                self.use_mixed_precision = True
                self.scaler = torch.cuda.amp.GradScaler()
                print("🚀 GPU optimizations enabled: TF32, cuDNN benchmark, mixed precision")
            else:
                print(f"🎮 Using CPU (GPU not available)")
                self.use_mixed_precision = False
                self.scaler = None
            print(f"   Device: {self.device}")
        else:
            # Fallback to discrete transformer
            self.policy_net = TrafficTransformer()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
            self.grpo = None
            self.use_advanced_ml = False
            self.device = torch.device('cpu')
            self.use_mixed_precision = False
            self.scaler = None
        
        # GRPO trajectory collection with larger batches for GPU
        self.trajectory = []
        self.rollout_length = 64 if torch.cuda.is_available() else 32  # Larger batches for GPU
        
        # Environment
        self.vehicles = []
        self.use_traffic_lights = False  # Experiment flag
        
        # Intersection managers (no-light coordination)
        self.intersection_managers = [
            IntersectionManager(400, 300),
            IntersectionManager(800, 300), 
            IntersectionManager(400, 500),
            IntersectionManager(800, 500)
        ]
        
        # Traffic light baseline
        self.traffic_lights = [
            FixedCycleSignal(30.0),  # 30-second cycle
            FixedCycleSignal(30.0),
            FixedCycleSignal(30.0), 
            FixedCycleSignal(30.0)
        ]
        
        # Legacy format for compatibility
        self.intersections = [
            {'x': 400, 'y': 300, 'manager': self.intersection_managers[0], 'light': self.traffic_lights[0]},
            {'x': 800, 'y': 300, 'manager': self.intersection_managers[1], 'light': self.traffic_lights[1]},
            {'x': 400, 'y': 500, 'manager': self.intersection_managers[2], 'light': self.traffic_lights[2]},
            {'x': 800, 'y': 500, 'manager': self.intersection_managers[3], 'light': self.traffic_lights[3]}
        ]
        
        self.training_history = {'rewards': [], 'throughput': [], 'loss': []}
        
    def spawn_vehicle(self) -> AutonomousVehicle:
        """Spawn vehicle at random entry point - collision-free spawning"""
        # Entry points closer to intersections (offset from road center)
        lane_offset = 3.0  # 3 meters from center line
        entry_points = [
            # Horizontal roads - right lanes - MUCH CLOSER TO INTERSECTIONS
            (150, 300 - lane_offset, 0),    # West entrance, heading East (right lane)
            (1050, 300 + lane_offset, math.pi),  # East entrance, heading West (right lane)
            (150, 500 - lane_offset, 0),    # West entrance, heading East (right lane) 
            (1050, 500 + lane_offset, math.pi),  # East entrance, heading West (right lane)
            # Vertical roads - right lanes - MUCH CLOSER TO INTERSECTIONS
            (400 + lane_offset, 150, math.pi/2),   # North entrance, heading South (right lane)
            (400 - lane_offset, 650, -math.pi/2),  # South entrance, heading North (right lane)
            (800 + lane_offset, 150, math.pi/2),   # North entrance, heading South (right lane)
            (800 - lane_offset, 650, -math.pi/2),  # South entrance, heading North (right lane)
        ]
        
        max_attempts = 50  # Prevent infinite loops
        
        for attempt in range(max_attempts):
            x, y, heading = random.choice(entry_points)
            
            # Add variation to spread out spawning (but keep in lane)
            if heading == 0 or heading == math.pi:  # Horizontal roads
                x += random.uniform(-30, 30)  # Spread along road
                y += random.uniform(-1.5, 1.5)  # Small lane variation (stay in lane)
            else:  # Vertical roads
                x += random.uniform(-1.5, 1.5)  # Small lane variation (stay in lane)
                y += random.uniform(-30, 30)  # Spread along road
            
            # Check for collisions with existing vehicles
            collision_free = True
            min_distance = 25.0  # Minimum safe distance
            
            for existing_vehicle in self.vehicles:
                distance = math.sqrt((existing_vehicle.x - x)**2 + (existing_vehicle.y - y)**2)
                if distance < min_distance:
                    collision_free = False
                    break
            
            if collision_free:
                vehicle = AutonomousVehicle(x, y, heading)
                # Give vehicles AUTONOMOUS starting speed - HIGHWAY LEVEL
                vehicle.speed = random.uniform(25.0, 35.0)  # True autonomous highway speeds
                
                # TEMPORARY DEBUG: Enable simple movement for first few episodes
                if hasattr(self, 'current_episode') and self.current_episode <= 2:
                    vehicle._debug_simple_movement = True
                    print(f"🔧 DEBUG: Vehicle {vehicle.id} using simple movement mode")
                
                # Wire the real policy network
                if self.use_advanced_ml:
                    vehicle.policy_net = self.policy_net
                    vehicle._all_vehicles = self.vehicles  # For GNN
                else:
                    vehicle.transformer = self.policy_net  # Fallback
                vehicle.target_speed = random.uniform(35, 42)  # HIGHWAY-LEVEL autonomous coordination
                vehicle._trainer = self
                vehicle.prev_exit_dist = vehicle.calculate_exit_distance()
                return vehicle
        
        # If we can't find a collision-free spot, don't spawn
        print(f"Warning: Could not find collision-free spawn point after {max_attempts} attempts")
        return None
    
    def calculate_reward(self, vehicle: AutonomousVehicle, dt: float) -> float:
        """Calculate reward for vehicle's actions"""
        reward = 0
        
        # MAJOR: Forward progress reward (HEAVILY encourages AUTONOMOUS SPEEDS)
        if vehicle.speed > 35.0:  # EXCELLENT autonomous speed - maximum reward
            reward += 12.0 * dt  # Huge reward for highway-level coordination
        elif vehicle.speed > 25.0:  # Good autonomous speed
            reward += 10.0 * dt
        elif vehicle.speed > 20.0:  # Moderate autonomous speed
            reward += 8.0 * dt
        elif vehicle.speed > 15.0:  # Acceptable speed
            reward += 6.0 * dt
        elif vehicle.speed > 5.0:  # Too slow for autonomous
            reward += 2.0 * dt
        else:  # VERY STRONG penalty for stopping/crawling
            reward -= 12.0 * dt  # Much stronger penalty for stopping
        
        # Delta progress reward - HUGE reward for actual progress toward exits
        current_exit_dist = vehicle.calculate_exit_distance()
        progress_delta = vehicle.prev_exit_dist - current_exit_dist
        reward += progress_delta / 10.0  # 5x higher reward for forward progress
        vehicle.prev_exit_dist = current_exit_dist
        
        # Strong exit bonus - encourage reaching boundaries
        if current_exit_dist < 30:  # Very close to exit
            reward += 10.0 * dt  # HUGE bonus for being near exit
        elif current_exit_dist < 60:  # Approaching exit
            reward += 5.0 * dt  # Big bonus
        
        # ANTI-STOPPING: Penalty for low speed when no traffic nearby
        nearby_vehicles = sum(1 for v in getattr(vehicle, '_all_vehicles', []) 
                             if v.id != vehicle.id and 
                             math.sqrt((v.x - vehicle.x)**2 + (v.y - vehicle.y)**2) < 50)
        
        if vehicle.speed < 5.0 and nearby_vehicles < 2:  # Slow with no reason
            reward -= 5.0 * dt  # Strong penalty for unnecessary slowing
        
        # Cooperation bonus
        if vehicle.behavior_mode == 'cooperative':
            reward += 0.2
        
        # Lane‑keeping penalty (much more lenient)
        deviation = abs(vehicle.lane_deviation())      # metres
        
        # Reduce lane-keeping penalties near exits to allow vehicles to exit
        near_exit = (vehicle.x < 150 or vehicle.x > 1050 or 
                    vehicle.y < 150 or vehicle.y > 650)
        
        # Reward proper lane keeping (staying within 2m of lane center)
        if deviation < 1.0:  # Very good lane keeping
            reward += 0.1 * dt
        elif deviation < 2.0:  # Acceptable lane keeping
            reward += 0.05 * dt  
        elif deviation > 3.0 and not near_exit:  # Poor lane keeping
            penalty_multiplier = 0.05 if near_exit else 0.1
            reward -= min(deviation, 3.0) * penalty_multiplier * dt
            # Only penalize if WAY off road
            if deviation > 15.0:
                reward -= 1.0
        
        # Safety penalty (REDUCED - only for extreme risk)
        if vehicle.collision_risk > 0.9:  # Only penalize very high risk
            reward -= 2.0 * vehicle.collision_risk
        elif vehicle.collision_risk > 0.7:  # Moderate penalty for high risk
            reward -= 0.5 * vehicle.collision_risk
        
        # Intersection coordination bonus
        if vehicle.has_reservation:
            reward += 0.5 * dt
        
        # THROUGHPUT BONUS: Big reward for vehicles that coordinate and keep moving
        if nearby_vehicles >= 2 and vehicle.speed > 10.0:  # Moving well despite traffic
            reward += 3.0 * dt  # Reward good coordination
        
        # Clamp total reward to prevent Q-value explosion (allow higher positive rewards)
        return np.clip(reward, -10.0, 20.0)
    
    def grpo_update(self):
        """Real GRPO update from collected trajectories with mixed precision"""
        if len(self.trajectory) < self.rollout_length:
            return {'policy_loss': 0.0, 'kl_div': 0.0, 'total_loss': 0.0}
        
        # Extract trajectory data
        states = torch.stack([t['state'] for t in self.trajectory])
        actions = torch.stack([t['action'] for t in self.trajectory])
        old_log_probs = torch.stack([t['log_prob'] for t in self.trajectory])
        rewards = torch.tensor([t['reward'] for t in self.trajectory])
        
        # Get device from model parameters
        device = self.device if hasattr(self, 'device') else next(self.policy_net.parameters()).device
        states = states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        rewards = rewards.to(device)
        
        # Perform GRPO update with mixed precision
        if self.grpo:
            loss_dict = self.grpo.update(states, actions, old_log_probs, rewards, self.trajectory, 
                                       use_mixed_precision=self.use_mixed_precision, scaler=self.scaler)
        else:
            # Fallback for discrete transformer with mixed precision
            self.policy_net.train()
            
            if self.use_mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    current_q_values, _ = self.policy_net(states)
                    current_q_values = current_q_values.gather(1, actions.long().unsqueeze(1))
                    
                    # Simple policy gradient for discrete case
                    returns = self.compute_returns(rewards)
                    loss = -torch.mean(current_q_values.squeeze() * returns)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                current_q_values, _ = self.policy_net(states)
                current_q_values = current_q_values.gather(1, actions.long().unsqueeze(1))
                
                # Simple policy gradient for discrete case
                returns = self.compute_returns(rewards)
                loss = -torch.mean(current_q_values.squeeze() * returns)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.optimizer.step()
            
            self.policy_net.eval()
            loss_dict = {'policy_loss': loss.item(), 'kl_div': 0.0, 'total_loss': loss.item()}
        
        # Clear trajectory
        self.trajectory.clear()
        return loss_dict
    
    def compute_returns(self, rewards, gamma=0.99):
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        for i in reversed(range(len(rewards))):
            running_return = rewards[i] + gamma * running_return
            returns[i] = running_return
        return returns
    
    def run_episode(self, viz: 'TrainingVisualization' = None, episode_length: int = 900, episode_num: int = 1) -> Dict:
        """Run GRPO training episode"""
        self.current_episode = episode_num  # Track current episode for debugging
        self.vehicles = []
        # Spawn MORE vehicles to fully utilize GPU processing power
        num_vehicles = 40 if torch.cuda.is_available() else 25  # More vehicles for GPU training
        for _ in range(num_vehicles):
            vehicle = self.spawn_vehicle()
            if vehicle:
                self.vehicles.append(vehicle)
        
        episode_rewards = []
        total_loss = 0
        training_steps = 0
        vehicles_passed = 0
        
        # Use consistent time step like simulation
        dt = 1/60  # Fixed 60 FPS to match simulation target
        
        for step in range(episode_length):
            # Handle visualization events to prevent GUI freezing
            if viz:
                if hasattr(viz, 'handle_events'):
                    # Panda3D visualization
                    if not viz.handle_events():
                        print("🛑 Training stopped by user")
                        return {'avg_reward': 0, 'avg_loss': 0, 'throughput': 0, 'trajectory_size': 0, 'training_steps': 0}
                else:
                    # Pygame visualization
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return {'avg_reward': 0, 'avg_loss': 0, 'throughput': 0, 'trajectory_size': 0, 'training_steps': 0}
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                return {'avg_reward': 0, 'avg_loss': 0, 'throughput': 0, 'trajectory_size': 0, 'training_steps': 0}
            
            # Update intersection managers
            for intersection in self.intersections:
                intersection['manager'].current_time += dt
                if not self.use_traffic_lights:
                    intersection['manager'].reservations = [
                        (vid, s, e) for vid, s, e in intersection['manager'].reservations 
                        if e > intersection['manager'].current_time
                    ]
                intersection['light'].update(dt)
            
            # Update vehicle physics ONCE
            for vehicle in self.vehicles:
                # Update physics only (no double observation collection)
                vehicle.kalman_filter.predict(dt)
                noisy_position = np.array([
                    vehicle.x + np.random.normal(0, 0.5),
                    vehicle.y + np.random.normal(0, 0.5)
                ])
                vehicle.kalman_filter.update(noisy_position)
                vehicle.handle_intersection_coordination([i['manager'] for i in self.intersections], dt)
                
                # Get observations ONCE
                observations = vehicle.get_observations(self.vehicles, [i['manager'] for i in self.intersections])
                vehicle.observation_history.append(observations)
                vehicle.calculate_collision_risk(self.vehicles)
            
            # Make decisions and collect trajectories
            for vehicle in self.vehicles:
                # Update all_vehicles reference for GNN
                if hasattr(vehicle, '_all_vehicles'):
                    vehicle._all_vehicles = self.vehicles
                
                # Make decision (returns action, log_prob)
                action, log_prob = vehicle.make_decision()
                vehicle.execute_action(action, dt)
                
                # Calculate reward
                reward = self.calculate_reward(vehicle, dt)
                episode_rewards.append(reward)
                
                # Collect trajectory data for GRPO
                if len(vehicle.observation_history) > 0:
                    self.trajectory.append({
                        'state': observations,  # Current observation
                        'action': action,
                        'log_prob': log_prob,
                        'reward': reward,
                        'vehicle': vehicle,  # Store vehicle reference for GNN
                        'all_vehicles': self.vehicles.copy()  # Store vehicle list for GNN
                    })
            
            # GRPO update every rollout_length steps
            if (step + 1) % self.rollout_length == 0:
                loss_dict = self.grpo_update()
                total_loss += loss_dict['total_loss']
                training_steps += 1
            
            # Remove vehicles that exit and count throughput
            initial_count = len(self.vehicles)
            self.vehicles = [v for v in self.vehicles 
                           if not (v.x < 50 or v.x > 1150 or v.y < 50 or v.y > 750)]
            vehicles_passed += initial_count - len(self.vehicles)
            
            # Maintain HIGH vehicle count for GPU utilization
            target_vehicles = 50 if torch.cuda.is_available() else 35
            if len(self.vehicles) < target_vehicles and step % 10 == 0:
                new_vehicle = self.spawn_vehicle()
                if new_vehicle:
                    self.vehicles.append(new_vehicle)
            
            # Debug output every 60 steps with detailed position info
            if step % 60 == 0 and step > 0:
                avg_speed = np.mean([v.speed for v in self.vehicles]) if self.vehicles else 0
                avg_exit_dist = np.mean([v.calculate_exit_distance() for v in self.vehicles]) if self.vehicles else 0
                avg_accel = np.mean([v.acceleration for v in self.vehicles]) if self.vehicles else 0
                
                # Detailed vehicle analysis
                if self.vehicles:
                    sample_vehicle = self.vehicles[0]
                    # Check distance to nearest intersection
                    nearest_intersection_dist = min([
                        math.sqrt((sample_vehicle.x - 400)**2 + (sample_vehicle.y - 300)**2),
                        math.sqrt((sample_vehicle.x - 800)**2 + (sample_vehicle.y - 300)**2),
                        math.sqrt((sample_vehicle.x - 400)**2 + (sample_vehicle.y - 500)**2),
                        math.sqrt((sample_vehicle.x - 800)**2 + (sample_vehicle.y - 500)**2)
                    ])
                    
                    # Sample an action to see what the network is producing
                    debug_action = "N/A"
                    try:
                        if hasattr(sample_vehicle, 'policy_net') and hasattr(sample_vehicle.policy_net, 'continuous_actions'):
                            comm_vector = torch.zeros(8)
                            mean, log_std = sample_vehicle.policy_net(self.vehicles, sample_vehicle, comm_vector)
                            debug_action = f"[{mean[0].item():.2f}, {mean[1].item():.2f}, {mean[2].item():.2f}]"
                    except Exception as e:
                        debug_action = f"Error: {e}"
                    
                    # Check if any vehicles are near intersections
                    vehicles_near_intersections = sum(1 for v in self.vehicles 
                                                     if min([math.sqrt((v.x - 400)**2 + (v.y - 300)**2),
                                                            math.sqrt((v.x - 800)**2 + (v.y - 300)**2),
                                                            math.sqrt((v.x - 400)**2 + (v.y - 500)**2),
                                                            math.sqrt((v.x - 800)**2 + (v.y - 500)**2)]) < 50)
                    
                    print(f"T+{step/60:.0f}s: {len(self.vehicles)} vehicles, avg_speed={avg_speed:.1f}, avg_accel={avg_accel:.2f}")
                    print(f"   Sample vehicle: pos=({sample_vehicle.x:.0f},{sample_vehicle.y:.0f}), speed={sample_vehicle.speed:.1f}, heading={sample_vehicle.heading:.2f}")
                    print(f"   Nearest intersection: {nearest_intersection_dist:.0f}m, exit_dist={sample_vehicle.calculate_exit_distance():.0f}m")
                    
                    print(f"   🚦 Vehicles near intersections (<50m): {vehicles_near_intersections}")
                    print(f"   Action output: {debug_action}, target_speed={sample_vehicle.target_speed:.1f}")
                else:
                    print(f"T+{step/60:.0f}s: No vehicles present")
                
            # Update visualization less frequently for better performance
            if viz and step % 30 == 0:  # Update every 30 steps (2 times per second) instead of 5
                viz.update_display(self.vehicles, self.intersections)
            
            # Handle events more frequently than visualization updates for responsiveness
            if viz and step % 3 == 0:  # Check events every 3 steps (20 times per second)
                if hasattr(viz, 'handle_events'):
                    # Panda3D visualization
                    if not viz.handle_events():
                        print("🛑 Training stopped by user")
                        return {'avg_reward': 0, 'avg_loss': 0, 'throughput': 0, 'trajectory_size': 0, 'training_steps': 0}
                else:
                    # Pygame visualization
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return {'avg_reward': 0, 'avg_loss': 0, 'throughput': 0, 'trajectory_size': 0, 'training_steps': 0}
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                return {'avg_reward': 0, 'avg_loss': 0, 'throughput': 0, 'trajectory_size': 0, 'training_steps': 0}
                
                # Smaller delay for better performance
                import time
                time.sleep(0.0001)  # 0.1ms delay instead of 1ms
        
        # Final GRPO update with remaining trajectory
        if self.trajectory:
            loss_dict = self.grpo_update()
            total_loss += loss_dict['total_loss']
            training_steps += 1
        
        return {
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'avg_loss': total_loss / max(training_steps, 1),
            'throughput': vehicles_passed,
            'trajectory_size': len(self.trajectory),
            'training_steps': training_steps
        }
    
    def train(self, episodes: int = 100, use_visualization: bool = True, use_3d: bool = None):
        """Train the model with optional visualization (2D pygame or 3D Panda3D)"""
        if use_visualization:
            # Auto-detect best visualization
            if use_3d is None:
                use_3d = PANDA3D_AVAILABLE
            
            if use_3d and PANDA3D_AVAILABLE:
                # Initialize Panda3D 3D visualization
                viz = Panda3DVisualization()
                viz.start_training(episodes)
                console.print("🎮 [bold green]Training with immersive 3D visualization[/bold green] - watch neural networks learn!")
            else:
                # Initialize pygame 2D visualization
                viz = TrainingVisualization()
                viz.start_training(episodes)
                console.print("🎮 [bold green]Training with live 2D visualization[/bold green] - watch the learning process!")
        else:
            viz = None
            console.print(Panel.fit(
                f"🧠 [bold blue]Training Transformer for {episodes} episodes[/bold blue]\n\n"
                f"• Each episode = 5 seconds of simulation time\n"
                f"• Vehicles learn intersection coordination through RL\n"
                f"• Target: 1-3 seconds per episode\n"
                f"• Expected total time: ~5-10 minutes",
                title="Training Configuration",
                border_style="blue"
            ))
        
        start_time = time.time()
        best_performance = -float('inf')
        best_episode = 1
        recent_rewards = deque(maxlen=10)
        
        # Progress indicators
        progress_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        
        episode_iterator = track(range(1, episodes + 1), description="Training Episodes...") if not viz else range(1, episodes + 1)
        
        for episode in episode_iterator:
            episode_start = time.time()
            
            # Visualization handling
            if viz:
                # Check for early exit
                if not viz.handle_events():
                    console.print(f"\n⏹️ [yellow]Training stopped early at episode {episode}[/yellow]")
                    break
                    
                # Update visualization during episode
                viz.update_display()
            else:
                # Rich progress bar handles display automatically
                pass
            
            results = self.run_episode(viz, episode_num=episode)
            last_results = results  # Safeguard for scope issues
            
            # Record metrics
            self.training_history['rewards'].append(results['avg_reward'])
            self.training_history['throughput'].append(results['throughput'])
            self.training_history['loss'].append(results['avg_loss'])
            recent_rewards.append(results['avg_reward'])
            
            # Update GRPO reference policy every 5 episodes
            if self.grpo and episode % 5 == 0:
                import copy
                self.grpo.set_reference_policy(copy.deepcopy(self.policy_net).eval())
                
            # Update visualization with results
            if viz:
                viz.update_episode(episode, results, 0.0)  # Remove epsilon reference
            
            # Calculate performance score
            performance = results['avg_reward'] + results['throughput'] * 0.1
            if performance > best_performance:
                best_performance = performance
                best_episode = episode
            
            episode_time = time.time() - episode_start
            
            # Detailed progress every episode for first 5, then every 10
            show_detail = episode <= 5 or episode % 10 == 0
            
            if show_detail and not viz:
                print(f"\r🎯 Episode {episode:3d}/{episodes} | "
                      f"Reward: {results['avg_reward']:6.2f} | "
                      f"Throughput: {results['throughput']:3d} | "
                      f"Loss: {results['avg_loss']:6.4f} | "
                      f"Traj: {results.get('trajectory_size', 0):4d} | "
                      f"Steps: {results['training_steps']:2d} | "
                      f"Time: {episode_time:.1f}s")
                
                # Learning progress indicators
                if episode >= 10:
                    recent_avg = np.mean(recent_rewards)
                    trend = "📈" if recent_avg > results['avg_reward'] - 1 else "📊" if recent_avg > results['avg_reward'] - 3 else "📉"
                    print(f"   Recent avg reward: {recent_avg:.2f} {trend}")
                
                # Trajectory status
                print(f"   Trajectory length: {results.get('trajectory_size', 0):5d} steps")
                
                # Learning phase description
                if episode <= 20:
                    print("   🌱 Early learning phase - vehicles exploring randomly")
                elif episode <= 50:
                    print("   🧠 Active learning phase - discovering coordination patterns")
                elif episode <= 80:
                    print("   ⚡ Optimization phase - refining coordination strategies")
                else:
                    print("   🎯 Final tuning phase - perfecting intersection coordination")
                
                print()
            elif not viz:
                # Simple progress for non-detailed episodes
                if episode % 5 == 0:
                    progress_pct = (episode / episodes) * 100
                    print(f"\r🔄 Progress: {episode}/{episodes} ({progress_pct:.0f}%) | "
                          f"Recent reward: {results['avg_reward']:.1f} | "
                          f"Time: {episode_time:.1f}s", end="", flush=True)
            
            # Progress milestones
            if episode in [10, 25, 50, 75] and not viz:
                elapsed = time.time() - start_time
                remaining = (elapsed / episode) * (episodes - episode)
                
                print(f"🏆 MILESTONE: {episode} episodes completed!")
                print(f"   Best performance so far: {best_performance:.2f} (episode {best_episode})")
                print(f"   Time elapsed: {elapsed/60:.1f}m | Estimated remaining: {remaining/60:.1f}m")
                print(f"   Learning trend: {'Improving' if len(recent_rewards) >= 5 and recent_rewards[-1] > recent_rewards[0] else 'Stabilizing'}")
                print()
                
            # Save checkpoint
            if episode in [10, 25, 50, 75]:
                checkpoint_path = f'checkpoint_ep{episode}.pth'
                torch.save({
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'training_history': self.training_history,
                    'episode': episode,
                    'use_advanced_ml': self.use_advanced_ml
                }, checkpoint_path)
                if not viz:
                    print(f"   💾 Checkpoint saved: {checkpoint_path}")
                    print()
        
        # Final results
        total_time = time.time() - start_time
        
        # Close visualization
        if viz:
            viz.close()
            
        print("\r" + " " * 80)  # Clear progress line
        print("🎉 TRAINING COMPLETED!")
        print("=" * 70)
        print(f"Total training time: {total_time/60:.1f} minutes")
        print(f"Best performance: {best_performance:.2f} (episode {best_episode})")
        final_reward = last_results['avg_reward'] if 'last_results' in locals() else 0.0
        print(f"Final reward: {final_reward:.2f}")
        if 'last_results' in locals():
            print(f"Total training steps: {last_results.get('training_steps', 0)}")
        else:
            print(f"Total training steps: 0")
        
        # Learning assessment
        if len(self.training_history['rewards']) >= 20:
            early_avg = np.mean(self.training_history['rewards'][:10])
            late_avg = np.mean(self.training_history['rewards'][-10:])
            improvement = ((late_avg - early_avg) / abs(early_avg)) * 100
            
            print(f"\n📊 Learning Analysis:")
            print(f"   Early episodes avg reward: {early_avg:.2f}")
            print(f"   Recent episodes avg reward: {late_avg:.2f}")
            print(f"   Overall improvement: {improvement:+.1f}%")
            
            if improvement > 50:
                print("   ✅ Excellent learning progress!")
            elif improvement > 20:
                print("   ✅ Good learning progress!")
            elif improvement > 0:
                print("   ⚠️  Modest learning progress - consider more episodes")
            else:
                print("   ❌ Poor learning - check hyperparameters")
        
        # Save final model with experiment tag
        experiment_tag = "lights" if self.use_traffic_lights else "coord"
        advanced_tag = "advanced" if self.use_advanced_ml else "basic"
        model_path = f'traffic_transformer_{experiment_tag}_{advanced_tag}.pth'
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'final_performance': best_performance,
            'training_time': total_time,
            'use_traffic_lights': self.use_traffic_lights,
            'use_advanced_ml': self.use_advanced_ml
        }, model_path)
        
        print(f"\n💾 Final model saved: {model_path}")
        print("🚗 Ready to run simulation with trained model!")
        print("   Run: python autocar.py → Choose option 2")
        print("=" * 70)
        
        # Plot training results
        self.plot_training_results()
        
        return self.training_history
    
    def plot_training_results(self):
        """Plot comprehensive training results"""
        if len(self.training_history['rewards']) < 2:
            print("Not enough data to plot")
            return
            
        try:
            import matplotlib.pyplot as plt
            
            # Suppress emoji warnings
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = 'DejaVu Sans'
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            episodes = range(1, len(self.training_history['rewards']) + 1)
            
            # 1. Rewards over time
            ax1.plot(episodes, self.training_history['rewards'], 'b-', alpha=0.7, label='Episode Reward')
            if len(self.training_history['rewards']) > 10:
                # Moving average
                window = min(10, len(self.training_history['rewards'])//4)
                rewards_ma = np.convolve(self.training_history['rewards'], 
                                       np.ones(window)/window, mode='valid')
                ax1.plot(range(window, len(episodes)+1), rewards_ma, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
            
            ax1.set_title('Training Rewards', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Average Reward')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. Throughput over time
            ax2.plot(episodes, self.training_history['throughput'], 'g-', alpha=0.7, label='Vehicles Passed')
            if len(self.training_history['throughput']) > 10:
                window = min(10, len(self.training_history['throughput'])//4)
                throughput_ma = np.convolve(self.training_history['throughput'], 
                                          np.ones(window)/window, mode='valid')
                ax2.plot(range(window, len(episodes)+1), throughput_ma, 'orange', linewidth=2, label=f'{window}-Episode Moving Avg')
            
            ax2.set_title('Traffic Throughput', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Vehicles per Episode')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 3. Training loss
            ax3.plot(episodes, self.training_history['loss'], 'purple', alpha=0.7, label='Training Loss')
            if len(self.training_history['loss']) > 10:
                window = min(10, len(self.training_history['loss'])//4)
                loss_ma = np.convolve(self.training_history['loss'], 
                                    np.ones(window)/window, mode='valid')
                ax3.plot(range(window, len(episodes)+1), loss_ma, 'red', linewidth=2, label=f'{window}-Episode Moving Avg')
            
            ax3.set_title('Neural Network Loss', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Loss Value')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 4. Learning progress summary
            ax4.axis('off')
            
            # Summary statistics
            final_reward = self.training_history['rewards'][-1]
            final_throughput = self.training_history['throughput'][-1]
            final_loss = self.training_history['loss'][-1]
            
            if len(self.training_history['rewards']) > 20:
                early_reward = np.mean(self.training_history['rewards'][:10])
                late_reward = np.mean(self.training_history['rewards'][-10:])
                reward_improvement = ((late_reward - early_reward) / abs(early_reward)) * 100
            else:
                reward_improvement = 0
            
            summary_text = f"""
TRAINING SUMMARY
═══════════════════════════

📊 Final Metrics:
   • Reward: {final_reward:.2f}
   • Throughput: {final_throughput} vehicles
   • Loss: {final_loss:.4f}

📈 Learning Progress:
   • Total Episodes: {len(episodes)}
   • Reward Improvement: {reward_improvement:+.1f}%
   
🎯 Performance Assessment:
"""
            
            if final_throughput > 5:
                assessment = "   ✅ EXCELLENT - Good vehicle coordination!"
            elif final_throughput > 2:
                assessment = "   ✅ GOOD - Decent coordination learning"  
            elif final_throughput > 0:
                assessment = "   ⚠️  MODERATE - Some learning observed"
            else:
                assessment = "   ❌ POOR - Needs more training"
            
            summary_text += assessment
            
            # Color-coded assessment
            if "EXCELLENT" in assessment:
                color = 'green'
            elif "GOOD" in assessment:
                color = 'blue'
            elif "MODERATE" in assessment:
                color = 'orange'
            else:
                color = 'red'
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.1))
            
            plt.suptitle('🚗 Autonomous Vehicle Training Results', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Training visualization saved: training_results.png")
            
            # Show the plot
            plt.show()
            
        except ImportError:
            print("📊 Install matplotlib to see training visualization: pip install matplotlib")
        except Exception as e:
            print(f"📊 Could not create visualization: {e}")

# ============================================================================
# SIMULATION SYSTEM
# ============================================================================

class TrafficSimulation:
    """Interactive traffic simulation with 2D pygame"""
    def __init__(self, model_path: str = 'traffic_transformer_coord_advanced.pth'):
        pygame.init()
        self.screen = pygame.display.set_mode((1600, 1000))  # Higher resolution for better quality
        pygame.display.set_caption("🚗 Autonomous Vehicle Coordination - High Speed Testing")
        self.clock = pygame.time.Clock()
        
        # Load trained model
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            # Try to determine model type from checkpoint
            if 'use_advanced_ml' in checkpoint and checkpoint['use_advanced_ml']:
                self.policy_net = AdvancedTrafficTransformer(continuous_actions=True)
                self.use_advanced_ml = True
            else:
                # Fallback to discrete transformer
                self.policy_net = TrafficTransformer()
                self.use_advanced_ml = False
            
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.policy_net.eval()
            print(f"✓ Loaded trained {'advanced' if self.use_advanced_ml else 'basic'} model from {model_path}")
        except FileNotFoundError:
            print(f"❌ No model found at {model_path}")
            print("Run training first!")
            self.running = False
            return
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Using basic model as fallback...")
            self.policy_net = TrafficTransformer()
            self.use_advanced_ml = False
        
        # Environment - scaled for higher resolution display
        self.vehicles = []
        self.intersection_managers = [
            IntersectionManager(530, 380),  # Scaled for 1600x1000
            IntersectionManager(1070, 380), 
            IntersectionManager(530, 620),
            IntersectionManager(1070, 620)
        ]
        
        # Display metadata for rendering (separate from logic)
        self.intersection_display = [
            {'x': 530, 'y': 380, 'occupancy': 0},
            {'x': 1070, 'y': 380, 'occupancy': 0},
            {'x': 530, 'y': 620, 'occupancy': 0},
            {'x': 1070, 'y': 620, 'occupancy': 0}
        ]
        
        self.running = True
        self.total_time = 0
        self.metrics = {'throughput': 0, 'cooperation': 0}
        
        # Spawn initial vehicles - HIGH DENSITY for "all vehicles automated" test
        for _ in range(25):
            self.spawn_vehicle()
    
    def spawn_vehicle(self):
        """Spawn vehicle with trained model - collision-free"""
        # Entry points scaled for higher resolution (offset from road center)
        lane_offset = 3.0  # 3 meters from center line
        entry_points = [
            # Horizontal roads - scaled for 1600x1000 resolution
            (200, 380 - lane_offset, 0),    # West entrance, heading East (right lane)
            (1400, 380 + lane_offset, math.pi),  # East entrance, heading West (right lane)
            (200, 620 - lane_offset, 0),    # West entrance, heading East (right lane) 
            (1400, 620 + lane_offset, math.pi),  # East entrance, heading West (right lane)
            # Vertical roads - scaled for 1600x1000 resolution
            (530 + lane_offset, 200, math.pi/2),   # North entrance, heading South (right lane)
            (530 - lane_offset, 800, -math.pi/2),  # South entrance, heading North (right lane)
            (1070 + lane_offset, 200, math.pi/2),   # North entrance, heading South (right lane)
            (1070 - lane_offset, 800, -math.pi/2),  # South entrance, heading North (right lane)
        ]
        
        max_attempts = 30  # Prevent infinite loops
        
        for attempt in range(max_attempts):
            x, y, heading = random.choice(entry_points)
            
            # Add variation to spread out spawning (but keep in lane)
            if heading == 0 or heading == math.pi:  # Horizontal roads
                x += random.uniform(-25, 25)  # Spread along road
                y += random.uniform(-1.5, 1.5)  # Small lane variation (stay in lane)
            else:  # Vertical roads
                x += random.uniform(-1.5, 1.5)  # Small lane variation (stay in lane)
                y += random.uniform(-25, 25)  # Spread along road
            
            # Check for collisions with existing vehicles
            collision_free = True
            min_distance = 25.0  # Minimum safe distance
            
            for existing_vehicle in self.vehicles:
                distance = math.sqrt((existing_vehicle.x - x)**2 + (existing_vehicle.y - y)**2)
                if distance < min_distance:
                    collision_free = False
                    break
            
            if collision_free:
                vehicle = AutonomousVehicle(x, y, heading)
                # Give vehicles AUTONOMOUS starting speed - HIGHWAY LEVEL
                vehicle.speed = random.uniform(25.0, 35.0)  # True autonomous highway speeds
                
                if self.use_advanced_ml:
                    vehicle.policy_net = self.policy_net
                    vehicle._all_vehicles = self.vehicles  # For GNN
                else:
                    vehicle.transformer = self.policy_net
                vehicle.target_speed = random.uniform(35, 42)  # HIGHWAY-LEVEL autonomous coordination
                self.vehicles.append(vehicle)
                return
        
        # If we can't find a collision-free spot, don't spawn
        if len(self.vehicles) < 5:  # Only warn if traffic is very light
            print(f"Warning: Could not find collision-free spawn point after {max_attempts} attempts")
    
    def update(self, dt: float):
        """Update simulation"""
        self.total_time += dt
        
        # Update intersection managers
        for manager in self.intersection_managers:
            manager.update(dt)
        
        # Update display metadata
        for i, intersection in enumerate(self.intersection_display):
            intersection['occupancy'] = sum(
                1 for v in self.vehicles 
                if math.sqrt((v.x - intersection['x'])**2 + (v.y - intersection['y'])**2) < 50
            )
        
        # Update vehicles with manager objects
        for vehicle in self.vehicles:
            vehicle.update(dt, self.vehicles, self.intersection_managers)
            
            if vehicle.behavior_mode == 'cooperative':
                self.metrics['cooperation'] += dt
        
        # Remove exited vehicles
        initial_count = len(self.vehicles)
        self.vehicles = [v for v in self.vehicles 
                        if not (v.x < 50 or v.x > 1550 or v.y < 50 or v.y > 950)]  # Scaled boundaries
        self.metrics['throughput'] += initial_count - len(self.vehicles)
        
        # Maintain traffic flow - HIGH DENSITY for "all vehicles automated" test
        if len(self.vehicles) < 35 and random.random() < 0.03:
            self.spawn_vehicle()
    
    def draw(self):
        """Draw simulation"""
        self.screen.fill((30, 30, 40))
        
        # Roads - scaled for higher resolution
        road_color = (60, 60, 75)
        pygame.draw.rect(self.screen, road_color, (0, 340, 1600, 80))  # Horizontal roads
        pygame.draw.rect(self.screen, road_color, (0, 580, 1600, 80))
        pygame.draw.rect(self.screen, road_color, (490, 0, 80, 1000))  # Vertical roads  
        pygame.draw.rect(self.screen, road_color, (1030, 0, 80, 1000))
        
        # Intersections - dark colors for better vehicle visibility
        for intersection in self.intersection_display:
            color = (60, 60, 60)      # Dark gray - clear
            if intersection['occupancy'] > 0:
                color = (80, 60, 40)  # Dark brown - occupied
            pygame.draw.rect(self.screen, color, 
                           (intersection['x'] - 40, intersection['y'] - 40, 80, 80))
        
        # Vehicles
        for vehicle in self.vehicles:
            vehicle.draw(self.screen)
        
        # UI
        font = pygame.font.Font(None, 24)
        
        stats = [
            f"🚗 Vehicles: {len(self.vehicles)}",
            f"📊 Throughput: {self.metrics['throughput']}",
            f"🤝 Cooperation: {self.metrics['cooperation']:.1f}s",
            f"⏱️ Time: {self.total_time:.1f}s"
        ]
        
        for i, stat in enumerate(stats):
            text = font.render(stat, True, (255, 255, 255))
            self.screen.blit(text, (20, 20 + i * 30))
        
        # Controls
        controls = font.render("SPACE: Add vehicles | ESC: Exit", True, (200, 200, 200))
        self.screen.blit(controls, (20, 750))
        
        pygame.display.flip()
    
    def run(self):
        """Run simulation with enhanced feedback"""
        print("🚗 Autonomous Vehicle Coordination Simulation")
        print("=" * 60)
        print("Watch vehicles coordinate at intersections without traffic lights!")
        print("Using trained transformer neural networks + Kalman filters")
        print("=" * 60)
        print("\n🎮 Controls:")
        print("   SPACE = Add 5 more vehicles")
        print("   ESC   = Exit simulation")
        print("\n⏱️  Starting simulation in 3 seconds...")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"   {i}...", end="", flush=True)
            time.sleep(1)
        print(" GO! 🚀\n")
        
        start_time = time.time()
        last_stats_time = 0
        performance_samples = deque(maxlen=12)  # 1 minute of 5-second samples
        
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        for _ in range(5):
                            self.spawn_vehicle()
                        print(f"➕ Added 5 vehicles (Total: {len(self.vehicles)})")
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
            
            self.update(dt)
            
            # Performance stats every 5 seconds
            if self.total_time - last_stats_time >= 5.0:
                throughput_rate = (self.metrics['throughput'] / max(1, self.total_time)) * 60
                cooperation_rate = self.metrics['cooperation'] / max(1, self.total_time)
                avg_speed = np.mean([v.speed for v in self.vehicles]) if self.vehicles else 0
                avg_collision_risk = np.mean([v.collision_risk for v in self.vehicles]) if self.vehicles else 0
                
                performance_samples.append(throughput_rate)
                
                # Performance assessment
                if throughput_rate > 40:
                    perf_status = "🔥 EXCELLENT"
                elif throughput_rate > 30:
                    perf_status = "✅ GOOD"
                elif throughput_rate > 20:
                    perf_status = "⚠️  LEARNING"
                else:
                    perf_status = "🐌 NEEDS TRAINING"
                
                print(f"📊 T+{self.total_time:3.0f}s | "
                      f"Vehicles: {len(self.vehicles):2d} | "
                      f"Throughput: {throughput_rate:4.1f}/min | "
                      f"Cooperation: {cooperation_rate:4.2f}/s | "
                      f"Safety: {(1-avg_collision_risk)*100:3.0f}% | "
                      f"{perf_status}")
                
                
                last_stats_time = self.total_time
                
                # Milestone celebrations
                if int(self.total_time) in [30, 60, 120, 300]:
                    print(f"\n🏆 MILESTONE: {int(self.total_time)} seconds of coordination!")
                    if len(performance_samples) >= 6:
                        avg_performance = np.mean(list(performance_samples)[-6:])
                        print(f"   Recent avg throughput: {avg_performance:.1f} vehicles/min")
                        if avg_performance > 35:
                            print("   🎉 Outstanding coordination performance!")
                        elif avg_performance > 25:
                            print("   👍 Good coordination learning!")
                    print()
            
            self.draw()
        
        # Final comprehensive results
        runtime = time.time() - start_time
        throughput_rate = (self.metrics['throughput'] / max(1, self.total_time)) * 60
        cooperation_rate = self.metrics['cooperation'] / max(1, self.total_time)
        
        print("\n" + "=" * 60)
        print("🏁 SIMULATION COMPLETED!")
        print("=" * 60)
        print(f"⏱️  Runtime: {runtime/60:.1f} minutes")
        print(f"🚗 Total vehicles processed: {self.metrics['throughput']}")
        print(f"📈 Average throughput: {throughput_rate:.1f} vehicles/minute")
        print(f"🤝 Cooperation rate: {cooperation_rate:.2f} events/second")
        print(f"🕐 Simulation time: {self.total_time:.1f} seconds")
        
        # Performance analysis
        print(f"\n📊 PERFORMANCE ANALYSIS:")
        if throughput_rate > 35:
            print("   ✅ EXCELLENT: Vehicles coordinate smoothly without traffic lights!")
            print("   🎯 Hypothesis SUPPORTED: Autonomous coordination works!")
        elif throughput_rate > 25:
            print("   ✅ GOOD: Strong coordination with room for improvement")
            print("   🎯 Hypothesis LIKELY SUPPORTED with more training")
        elif throughput_rate > 15:
            print("   ⚠️  MODERATE: Some coordination but not optimal")
            print("   🎯 Hypothesis PARTIALLY SUPPORTED - needs more training")
        else:
            print("   ❌ POOR: Limited coordination observed")
            print("   🎯 Hypothesis NOT SUPPORTED - model needs significant training")
        
        # Intersection analysis
        print(f"\n🚦 INTERSECTION ANALYSIS:")
        for i, intersection in enumerate(self.intersection_display):
            occupancy = intersection.get('occupancy', 0)
            status = "🔴 Congested" if occupancy > 3 else "🟡 Moderate" if occupancy > 1 else "🟢 Clear"
            print(f"   Intersection {i+1}: {status}")
        
        print(f"\n💡 INSIGHTS:")
        if cooperation_rate > 0.5:
            print("   • High cooperation rate indicates effective vehicle-to-vehicle coordination")
        if len(self.vehicles) > 15:
            print(f"   • Managed {len(self.vehicles)} simultaneous vehicles successfully")
        
        print(f"\n🔬 For better results:")
        print("   • Train for more episodes (200+ recommended)")
        print("   • Run longer simulations to see learning convergence")
        print("   • Add more vehicles to test scalability")
        
        pygame.quit()

class TrafficSimulation3D(ShowBase):
    """Interactive 3D traffic simulation using Panda3D"""
    def __init__(self, model_path: str = 'traffic_transformer_coord_advanced.pth'):
        if not PANDA3D_AVAILABLE:
            raise ImportError("Panda3D not available. Install with: pip install panda3d")
        
        # Initialize Panda3D ShowBase
        ShowBase.__init__(self)
        
        # Window configuration
        props = WindowProperties()
        props.setTitle("🚗 Autonomous Vehicle 3D Coordination Simulation")
        props.setSize(1600, 900)
        self.win.requestProperties(props)
        
        # Camera setup - bird's eye view for traffic monitoring
        self.camera.setPos(600, -800, 400)
        self.camera.lookAt(600, 400, 0)
        
        # Initialize simulation-specific attributes
        self.running = True
        self.total_time = 0
        self.metrics = {'throughput': 0, 'cooperation': 0}
        self.vehicle_nodes = {}
        self.intersection_nodes = []
        
        # Setup 3D environment
        self.setup_lighting()
        self.setup_environment()
        self.setup_ui()
        self.setup_camera_controls()
        
        # Load trained model
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            # Try to determine model type from checkpoint
            if 'use_advanced_ml' in checkpoint and checkpoint['use_advanced_ml']:
                self.policy_net = AdvancedTrafficTransformer(continuous_actions=True)
                self.use_advanced_ml = True
            else:
                # Fallback to discrete transformer
                self.policy_net = TrafficTransformer()
                self.use_advanced_ml = False
            
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.policy_net.eval()
            print(f"✓ Loaded trained {'advanced' if self.use_advanced_ml else 'basic'} model from {model_path}")
        except FileNotFoundError:
            print(f"❌ No model found at {model_path}")
            print("Run training first!")
            self.running = False
            return
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Using basic model as fallback...")
            self.policy_net = TrafficTransformer()
            self.use_advanced_ml = False
        
        # Replace the vehicles list with simulation vehicles
        self.vehicles = []
        
        # Initialize intersection managers for simulation
        self.intersection_managers = [
            IntersectionManager(400, 300),
            IntersectionManager(800, 300), 
            IntersectionManager(400, 500),
            IntersectionManager(800, 500)
        ]
        
        # Update display metadata for rendering (override parent's intersection_display)
        for i, manager in enumerate(self.intersection_managers):
            if i < len(self.intersection_nodes):
                self.intersection_nodes[i]['occupancy'] = 0
        
        # Override UI for simulation mode
        self.setup_simulation_ui()
        
        # Spawn initial vehicles - HIGH DENSITY for "all vehicles automated" test
        for _ in range(25):
            self.spawn_vehicle()
        
        # Add simulation controls
        self.accept("space", self.add_vehicles)
        self.accept("escape", self.exit_simulation)
        
        # Start simulation loop
        self.taskMgr.add(self.simulation_update, "simulation_update")
    
    def setup_lighting(self):
        """Create realistic lighting for the 3D scene"""
        # Ambient light for overall illumination
        ambient_light = AmbientLight('ambient')
        ambient_light.setColor((0.3, 0.3, 0.4, 1))
        ambient_np = self.render.attachNewNode(ambient_light)
        self.render.setLight(ambient_np)
        
        # Directional light (sun) for shadows and depth
        sun_light = DirectionalLight('sun')
        sun_light.setColor((0.8, 0.8, 0.7, 1))
        sun_light.setDirection(Vec3(-1, -1, -0.5))
        sun_np = self.render.attachNewNode(sun_light)
        sun_np.setPos(600, 400, 500)
        self.render.setLight(sun_np)
        
        # Additional point lights at intersections for better visibility
        for i, pos in enumerate([(400, 300), (800, 300), (400, 500), (800, 500)]):
            point_light = PointLight(f'intersection_light_{i}')
            point_light.setColor((0.4, 0.4, 0.5, 1))
            point_np = self.render.attachNewNode(point_light)
            point_np.setPos(pos[0], pos[1], 50)
            self.render.setLight(point_np)
        
        # Atmospheric fog for depth perception
        fog = Fog('atmosphere')
        fog.setColor(0.8, 0.8, 0.9)
        fog.setExpDensity(0.0008)
        self.render.setFog(fog)
    
    def setup_environment(self):
        """Create the 3D road environment"""
        # Ground plane
        ground = self.create_ground_plane()
        ground.reparentTo(self.render)
        
        # Road system
        self.create_road_system()
        
        # Intersections
        self.create_intersections()
        
        # Environmental elements
        self.create_environment_details()
    
    def create_ground_plane(self):
        """Create a textured ground plane"""
        cm = CardMaker('ground')
        cm.setFrame(-100, 1300, -100, 900)  # Large ground area
        ground = self.render.attachNewNode(cm.generate())
        ground.setP(-90)  # Rotate to be horizontal
        ground.setColor(0.2, 0.4, 0.2, 1)  # Dark green grass
        
        # Add subtle texture variation
        ground.setTexScale(TextureStage.getDefault(), 50, 50)
        return ground
    
    def create_road_system(self):
        """Create 3D road network"""
        # Road material properties
        road_color = (0.3, 0.3, 0.35, 1)
        lane_line_color = (0.9, 0.9, 0.9, 1)
        
        # Horizontal roads
        for y in [300, 500]:
            road = self.create_road_segment(0, y-30, 1200, 60, road_color)
            road.reparentTo(self.render)
            
            # Lane markings
            for x in range(50, 1200, 40):
                marking = self.create_lane_marking(x, y, 20, 2, lane_line_color)
                marking.reparentTo(self.render)
        
        # Vertical roads  
        for x in [400, 800]:
            road = self.create_road_segment(x-30, 0, 60, 800, road_color)
            road.reparentTo(self.render)
            
            # Lane markings
            for y in range(50, 800, 40):
                marking = self.create_lane_marking(x, y, 2, 20, lane_line_color)
                marking.reparentTo(self.render)
    
    def create_road_segment(self, x, y, width, height, color):
        """Create a single road segment"""
        cm = CardMaker('road')
        cm.setFrame(0, width, 0, height)
        road = self.render.attachNewNode(cm.generate())
        road.setPos(x, y, 0.1)  # Slightly above ground
        road.setColor(*color)
        road.setP(-90)  # Make horizontal
        return road
    
    def create_lane_marking(self, x, y, width, height, color):
        """Create lane markings"""
        cm = CardMaker('lane_marking')
        cm.setFrame(-width/2, width/2, -height/2, height/2)
        marking = self.render.attachNewNode(cm.generate())
        marking.setPos(x, y, 0.2)  # Above road surface
        marking.setColor(*color)
        marking.setP(-90)
        return marking
    
    def create_intersections(self):
        """Create 3D intersection visualizations"""
        intersection_positions = [(400, 300), (800, 300), (400, 500), (800, 500)]
        
        for i, (x, y) in enumerate(intersection_positions):
            # Intersection platform
            cm = CardMaker(f'intersection_{i}')
            cm.setFrame(-40, 40, -40, 40)
            intersection = self.render.attachNewNode(cm.generate())
            intersection.setPos(x, y, 0.15)
            intersection.setColor(0.4, 0.4, 0.4, 1)  # Darker intersection color
            intersection.setP(-90)
            
            # Traffic coordination indicator (will change color based on occupancy)
            indicator = self.create_traffic_indicator(x, y)
            indicator.reparentTo(self.render)
            
            self.intersection_nodes.append({
                'node': intersection,
                'indicator': indicator,
                'position': (x, y),
                'occupancy': 0
            })
    
    def create_traffic_indicator(self, x, y):
        """Create traffic coordination status indicator"""
        cm = CardMaker('indicator')
        cm.setFrame(-5, 5, -5, 5)
        indicator = self.render.attachNewNode(cm.generate())
        indicator.setPos(x, y, 20)  # Elevated above intersection
        indicator.setColor(0, 1, 0, 0.7)  # Green = clear
        indicator.setTransparency(TransparencyAttrib.MAlpha)
        indicator.setP(-90)
        indicator.setBillboardAxis()  # Always face camera
        return indicator
    
    def create_environment_details(self):
        """Add environmental details for immersion"""
        # Simple buildings/structures around the road network
        building_positions = [
            (200, 150, 30), (600, 150, 25), (1000, 150, 35),
            (200, 650, 28), (600, 650, 32), (1000, 650, 30),
            (150, 400, 20), (1050, 400, 25)
        ]
        
        for i, (x, y, height) in enumerate(building_positions):
            building = self.create_building(x, y, height)
            building.reparentTo(self.render)
    
    def create_building(self, x, y, height):
        """Create simple building structures"""
        cm = CardMaker('building')
        cm.setFrame(-15, 15, 0, height)
        building = self.render.attachNewNode(cm.generate())
        building.setPos(x, y, 0)
        building.setColor(0.6, 0.6, 0.7, 1)
        building.lookAt(600, 400, 0)  # Face toward center
        return building
    
    def create_vehicle_model(self, vehicle_id: int):
        """Create 3D vehicle model"""
        # Create vehicle body
        cm = CardMaker(f'vehicle_{vehicle_id}')
        cm.setFrame(-6, 6, -3, 3)  # 12x6 meter vehicle
        vehicle_node = self.render.attachNewNode(cm.generate())
        vehicle_node.setColor(0.8, 0.2, 0.2, 1)  # Red vehicle
        vehicle_node.setP(-90)  # Make horizontal
        vehicle_node.setZ(1)  # Slightly above road
        
        # Add vehicle details
        # Windshield
        windshield = CardMaker('windshield')
        windshield.setFrame(-4, 4, -2, 2)
        windshield_node = vehicle_node.attachNewNode(windshield.generate())
        windshield_node.setPos(0, 0, 2)
        windshield_node.setColor(0.2, 0.2, 0.4, 0.8)
        windshield_node.setTransparency(TransparencyAttrib.MAlpha)
        
        # Direction indicator
        indicator = CardMaker('direction')
        indicator.setFrame(-1, 1, -1, 1)
        indicator_node = vehicle_node.attachNewNode(indicator.generate())
        indicator_node.setPos(5, 0, 1)  # Front of vehicle
        indicator_node.setColor(1, 1, 1, 1)  # White
        
        return vehicle_node
    
    def setup_ui(self):
        """Setup 3D UI elements"""
        # Title
        self.title_text = OnscreenText(
            text="🚗 Autonomous Vehicle 3D Coordination Simulation",
            pos=(0, 0.9), scale=0.08, fg=(1, 1, 1, 1),
            align=TextNode.ACenter, mayChange=True
        )
        
        # Controls help
        self.controls_text = OnscreenText(
            text="Controls: Mouse = Rotate | Wheel = Zoom | SPACE = Add vehicles | ESC = Exit",
            pos=(0, -0.9), scale=0.04, fg=(0.8, 0.8, 0.8, 1),
            align=TextNode.ACenter, mayChange=False
        )
        
        # Add simulation-specific stats
        self.simulation_text = OnscreenText(
            text="", pos=(0.6, 0.8), scale=0.05, fg=(0.8, 1, 1, 1),
            align=TextNode.ALeft, mayChange=True
        )
    
    def setup_camera_controls(self):
        """Setup interactive camera controls"""
        # Mouse camera control
        self.camera_pivot = self.render.attachNewNode("camera_pivot")
        self.camera_pivot.setPos(600, 400, 0)  # Center of scene
        
        # Accept mouse events
        self.accept("mouse1", self.start_camera_rotate)
        self.accept("mouse1-up", self.stop_camera_rotate)
        self.accept("wheel_up", self.zoom_in)
        self.accept("wheel_down", self.zoom_out)
        
        self.camera_rotating = False
        self.last_mouse_pos = None
    
    def start_camera_rotate(self):
        """Start camera rotation"""
        self.camera_rotating = True
        self.last_mouse_pos = None
    
    def stop_camera_rotate(self):
        """Stop camera rotation"""
        self.camera_rotating = False
    
    def zoom_in(self):
        """Zoom camera in"""
        current_pos = self.camera.getPos()
        center = Vec3(600, 400, 0)
        direction = current_pos - center
        direction.normalize()
        new_pos = current_pos - direction * 50
        self.camera.setPos(new_pos)
    
    def zoom_out(self):
        """Zoom camera out"""
        current_pos = self.camera.getPos()
        center = Vec3(600, 400, 0)
        direction = current_pos - center
        direction.normalize()
        new_pos = current_pos + direction * 50
        self.camera.setPos(new_pos)
    
    def update_vehicles(self, vehicles: List):
        """Update 3D vehicle positions and states"""
        current_vehicle_ids = set(v.id for v in vehicles)
        
        # Remove vehicles that no longer exist
        for vehicle_id in list(self.vehicle_nodes.keys()):
            if vehicle_id not in current_vehicle_ids:
                self.vehicle_nodes[vehicle_id].removeNode()
                del self.vehicle_nodes[vehicle_id]
        
        # Update existing vehicles and create new ones
        for vehicle in vehicles:
            if vehicle.id not in self.vehicle_nodes:
                # Create new vehicle
                self.vehicle_nodes[vehicle.id] = self.create_vehicle_model(vehicle.id)
            
            # Update vehicle position and orientation
            vehicle_node = self.vehicle_nodes[vehicle.id]
            vehicle_node.setPos(vehicle.x, vehicle.y, 1)
            vehicle_node.setH(math.degrees(-vehicle.heading) + 90)  # Convert to Panda3D heading
            
            # Color based on behavior and state
            if vehicle.behavior_mode == 'cooperative':
                color = (0.2, 0.8, 0.2, 1)  # Green for cooperative
            elif vehicle.collision_risk > 0.5:
                color = (1, 0.5, 0, 1)  # Orange for high risk
            elif vehicle.speed < 5:
                color = (0.8, 0.8, 0.2, 1)  # Yellow for slow
            else:
                color = (0.2, 0.5, 1, 1)  # Blue for normal
            
            vehicle_node.setColor(*color)
            
            # Scale based on speed (visual feedback)
            speed_scale = 1.0 + (vehicle.speed / 45.0) * 0.3
            vehicle_node.setScale(speed_scale, 1, 1)
    
    def update_intersections(self, intersections: List):
        """Update intersection status indicators"""
        for i, intersection_data in enumerate(intersections):
            if i < len(self.intersection_nodes):
                # Calculate occupancy
                occupancy = intersection_data.get('occupancy', 0)
                
                # Update indicator color based on traffic
                indicator = self.intersection_nodes[i]['indicator']
                if occupancy > 3:
                    indicator.setColor(1, 0, 0, 0.8)  # Red - congested
                elif occupancy > 1:
                    indicator.setColor(1, 0.5, 0, 0.8)  # Orange - moderate
                elif occupancy > 0:
                    indicator.setColor(1, 1, 0, 0.8)  # Yellow - light traffic
                else:
                    indicator.setColor(0, 1, 0, 0.7)  # Green - clear
                
                # Animate indicator based on activity
                scale = 1.0 + occupancy * 0.2
                indicator.setScale(scale)
    
    def setup_simulation_ui(self):
        """Override UI for simulation mode"""
        # Update title for simulation
        self.title_text.setText("🚗 Autonomous Vehicle 3D Coordination Simulation")
        
        # Update controls text
        self.controls_text.setText("Controls: Mouse = Rotate | Wheel = Zoom | SPACE = Add vehicles | ESC = Exit")
        
        # Add simulation-specific stats
        self.simulation_text = OnscreenText(
            text="", pos=(0.6, 0.8), scale=0.05, fg=(0.8, 1, 1, 1),
            align=TextNode.ALeft, mayChange=True
        )
    
    def spawn_vehicle(self):
        """Spawn vehicle with trained model - collision-free"""
        # Entry points (adjusted for the simulation coordinate system)
        lane_offset = 3.0
        entry_points = [
            (150, 300 - lane_offset, 0),    # West entrance, heading East
            (1050, 300 + lane_offset, math.pi),  # East entrance, heading West
            (150, 500 - lane_offset, 0),    # West entrance, heading East
            (1050, 500 + lane_offset, math.pi),  # East entrance, heading West
            (400 + lane_offset, 150, math.pi/2),   # North entrance, heading South
            (400 - lane_offset, 650, -math.pi/2),  # South entrance, heading North
            (800 + lane_offset, 150, math.pi/2),   # North entrance, heading South
            (800 - lane_offset, 650, -math.pi/2),  # South entrance, heading North
        ]
        
        max_attempts = 30
        
        for attempt in range(max_attempts):
            x, y, heading = random.choice(entry_points)
            
            # Add variation to spread out spawning
            if heading == 0 or heading == math.pi:
                x += random.uniform(-25, 25)
                y += random.uniform(-1.5, 1.5)
            else:
                x += random.uniform(-1.5, 1.5)
                y += random.uniform(-25, 25)
            
            # Check for collisions
            collision_free = True
            min_distance = 25.0
            
            for existing_vehicle in self.vehicles:
                distance = math.sqrt((existing_vehicle.x - x)**2 + (existing_vehicle.y - y)**2)
                if distance < min_distance:
                    collision_free = False
                    break
            
            if collision_free:
                vehicle = AutonomousVehicle(x, y, heading)
                vehicle.speed = random.uniform(25.0, 35.0)
                
                if self.use_advanced_ml:
                    vehicle.policy_net = self.policy_net
                    vehicle._all_vehicles = self.vehicles
                else:
                    vehicle.transformer = self.policy_net
                vehicle.target_speed = random.uniform(35, 42)
                self.vehicles.append(vehicle)
                return
        
        if len(self.vehicles) < 5:
            print(f"Warning: Could not find collision-free spawn point after {max_attempts} attempts")
    
    def simulation_update(self, task):
        """Main simulation update loop"""
        dt = task.time - getattr(self, '_last_time', 0)
        self._last_time = task.time
        dt = min(dt, 1/30.0)  # Cap at 30 FPS minimum
        self.total_time += dt
        
        # Update intersection managers
        for manager in self.intersection_managers:
            manager.update(dt)
        
        # Update intersection occupancy for visualization
        for i, intersection_node in enumerate(self.intersection_nodes):
            if i < len(self.intersection_managers):
                occupancy = sum(
                    1 for v in self.vehicles 
                    if math.sqrt((v.x - intersection_node['position'][0])**2 + 
                               (v.y - intersection_node['position'][1])**2) < 50
                )
                intersection_node['occupancy'] = occupancy
        
        # Update vehicles with manager objects
        for vehicle in self.vehicles:
            vehicle.update(dt, self.vehicles, self.intersection_managers)
            
            if vehicle.behavior_mode == 'cooperative':
                self.metrics['cooperation'] += dt
        
        # Remove exited vehicles
        initial_count = len(self.vehicles)
        self.vehicles = [v for v in self.vehicles 
                        if not (v.x < 50 or v.x > 1150 or v.y < 50 or v.y > 750)]
        self.metrics['throughput'] += initial_count - len(self.vehicles)
        
        # Maintain traffic flow
        if len(self.vehicles) < 35 and random.random() < 0.03:
            self.spawn_vehicle()
        
        # Update 3D visualization
        self.update_vehicles(self.vehicles)
        self.update_intersections(self.intersection_nodes)
        
        # Update simulation stats
        self.update_simulation_stats()
        
        return task.cont
    
    def update_simulation_stats(self):
        """Update simulation-specific statistics"""
        throughput_rate = (self.metrics['throughput'] / max(1, self.total_time)) * 60
        cooperation_rate = self.metrics['cooperation'] / max(1, self.total_time)
        avg_speed = np.mean([v.speed for v in self.vehicles]) if self.vehicles else 0
        avg_collision_risk = np.mean([v.collision_risk for v in self.vehicles]) if self.vehicles else 0
        
        simulation_text = f"""🚗 Active Vehicles: {len(self.vehicles)}
📊 Throughput: {throughput_rate:.1f}/min
🤝 Cooperation: {cooperation_rate:.2f}/s
⚡ Avg Speed: {avg_speed:.1f} m/s
⚠️ Collision Risk: {avg_collision_risk:.1%}
⏱️ Runtime: {self.total_time:.1f}s"""
        
        self.simulation_text.setText(simulation_text)
        
        # Performance assessment for metrics display
        self.training_metrics.update({
            'vehicles': len(self.vehicles),
            'throughput': int(throughput_rate),
            'cooperation': cooperation_rate,
            'collision_risk': avg_collision_risk,
            'reward': throughput_rate / 10.0  # Convert to reward-like metric
        })
    
    def add_vehicles(self):
        """Add 5 more vehicles (SPACE key)"""
        for _ in range(5):
            self.spawn_vehicle()
        print(f"➕ Added 5 vehicles (Total: {len(self.vehicles)})")
    
    def exit_simulation(self):
        """Exit simulation (ESC key)"""
        self.running = False
        self.userExit = True
    
    def run_simulation(self):
        """Run the 3D simulation"""
        print("🚗 3D Autonomous Vehicle Coordination Simulation")
        print("=" * 60)
        print("Immersive 3D visualization with Panda3D!")
        print("Watch vehicles coordinate at intersections without traffic lights!")
        print("Using trained transformer neural networks + Kalman filters")
        print("=" * 60)
        print("\n🎮 Controls:")
        print("   Mouse Drag = Rotate camera")
        print("   Mouse Wheel = Zoom in/out")
        print("   SPACE = Add 5 more vehicles")
        print("   ESC   = Exit simulation")
        print("\n🚀 Starting 3D simulation...")
        
        # Start Panda3D main loop
        self.run()

# ============================================================================
# GRAPH NEURAL NETWORK FOR VEHICLE RELATIONSHIPS
# ============================================================================

class GraphSAGEVehicleNet(nn.Module):
    """GraphSAGE for vehicle relationship modeling - 2024-2025 optimized
    - 15-25 neighbors per node
    - Hidden dimensions 128-256
    - Mean aggregation (best for AV scenarios)
    """
    def __init__(self, node_features: int = 32, hidden_dim: int = 128, 
                 num_neighbors: int = 20, num_layers: int = 3):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors  # Optimized 15-25 range
        self.edge_features = 7  # Number of edge features we're using
        
        # Node feature encoding with batch norm
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GraphSAGE layers with mean aggregation (optimal for AV)
        self.sage_layers = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim, aggr='mean', normalize=True)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each SAGE layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Communication network
        self.comm_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 16)  # Communication embedding
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim + 16, hidden_dim),  # Include communication
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)  # Final vehicle representation
        )
        
    def create_vehicle_graph(self, vehicles: List, ego_vehicle) -> Data:
        """Create graph representation of vehicle interactions"""
        if not vehicles:
            return None
            
        # Filter nearby vehicles
        nearby_vehicles = []
        for vehicle in vehicles:
            if vehicle.id != ego_vehicle.id:
                distance = math.sqrt((vehicle.x - ego_vehicle.x)**2 + (vehicle.y - ego_vehicle.y)**2)
                if distance <= 150.0:  # Extended range for GNN
                    nearby_vehicles.append(vehicle)
        
        if not nearby_vehicles:
            return None
            
        # Node features: [x, y, vx, vy, heading, speed, accel, lane_dev, collision_risk, ...]
        node_features = []
        vehicle_list = [ego_vehicle] + nearby_vehicles
        
        for vehicle in vehicle_list:
            features = [
                vehicle.x / 1200.0,  # Normalized position
                vehicle.y / 800.0,
                vehicle.speed * math.cos(vehicle.heading) / 45.0,  # Velocity components (autonomous speeds)
                vehicle.speed * math.sin(vehicle.heading) / 45.0,
                (vehicle.heading % (2 * math.pi)) / (2 * math.pi),  # Normalized heading
                vehicle.speed / 45.0,
                vehicle.acceleration / 8.0,
                vehicle.lane_deviation() / 10.0,
                vehicle.collision_risk,
                1.0 if hasattr(vehicle, 'has_reservation') and vehicle.has_reservation else 0.0,
                # Behavioral encoding
                {'normal': 0.25, 'aggressive': 0.5, 'conservative': 0.75, 'cooperative': 1.0}.get(vehicle.behavior_mode, 0.25),
                vehicle.target_speed / 45.0,
                vehicle.waiting_time / 10.0 if hasattr(vehicle, 'waiting_time') else 0.0,
                # Pad to node_features size
                *([0.0] * (self.node_features - 13))
            ]
            node_features.append(features[:self.node_features])
        
        # Create edges based on spatial proximity and interaction potential
        edge_indices = []
        edge_attrs = []
        
        for i, vehicle_i in enumerate(vehicle_list):
            for j, vehicle_j in enumerate(vehicle_list):
                if i != j:
                    distance = math.sqrt((vehicle_i.x - vehicle_j.x)**2 + (vehicle_i.y - vehicle_j.y)**2)
                    if distance <= 100.0:  # Connection threshold
                        edge_indices.append([i, j])
                        
                        # Edge features: [distance, relative_velocity, relative_heading, interaction_strength]
                        rel_vel_x = (vehicle_j.speed * math.cos(vehicle_j.heading) - 
                                   vehicle_i.speed * math.cos(vehicle_i.heading)) / 45.0
                        rel_vel_y = (vehicle_j.speed * math.sin(vehicle_j.heading) - 
                                   vehicle_i.speed * math.sin(vehicle_i.heading)) / 45.0
                        rel_heading = ((vehicle_j.heading - vehicle_i.heading) % (2 * math.pi)) / (2 * math.pi)
                        interaction_strength = max(0, 1 - distance / 100.0)  # Closer = stronger interaction
                        
                        edge_attr = [
                            distance / 100.0,  # Normalized distance
                            rel_vel_x, rel_vel_y,
                            rel_heading,
                            interaction_strength,
                            # Collision potential
                            1.0 if distance < 20 and abs(rel_vel_x) + abs(rel_vel_y) > 0.1 else 0.0,
                            # Same road indicator
                            1.0 if abs(vehicle_i.lane_center()[0] - vehicle_j.lane_center()[0]) < 50 or 
                                   abs(vehicle_i.lane_center()[1] - vehicle_j.lane_center()[1]) < 50 else 0.0,
                            *([0.0] * (self.edge_features - 7))
                        ]
                        edge_attrs.append(edge_attr[:self.edge_features])
        
        if not edge_indices:
            return None
            
        # Get device from model parameters and ensure all tensors are on same device
        device = next(self.node_encoder.parameters()).device
        x = torch.FloatTensor(node_features).to(device)
        edge_index = torch.LongTensor(edge_indices).t().contiguous().to(device)
        edge_attr = torch.FloatTensor(edge_attrs).to(device)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def forward(self, graph_data: Data) -> torch.Tensor:
        """Forward pass through graph network"""
        if graph_data is None:
            device = next(self.node_encoder.parameters()).device
            return torch.zeros(1, 64, device=device)  # Return default if no graph
            
        x = self.node_encoder(graph_data.x)
        
        # Apply GraphSAGE layers
        for i, sage_layer in enumerate(self.sage_layers):
            x = F.relu(sage_layer(x, graph_data.edge_index))
            x = self.layer_norms[i](x)
            x = F.dropout(x, training=self.training)
        
        # Generate communication embeddings
        comm_embeddings = self.comm_encoder(x)
        
        # Aggregate communication (simple mean for now)
        global_comm = torch.mean(comm_embeddings, dim=0, keepdim=True)
        
        # Combine local features with global communication
        ego_features = x[0:1]  # Ego vehicle is always first node
        combined = torch.cat([ego_features, global_comm], dim=1)
        
        output = self.output_proj(combined)
        return output

# ============================================================================
# GRPO (GROUP RELATIVE POLICY OPTIMIZATION)
# ============================================================================

class GRPO:
    """Group Relative Policy Optimization - Advanced RL without critic
    Optimized parameters based on 2024-2025 advances:
    - Group sizes: 4-8 responses per query
    - Clip epsilon: 0.2
    - Learning rates: 1e-5 to 5e-5
    """
    def __init__(self, policy_net, optimizer, group_size: int = 6, 
                 kl_coeff: float = 0.015, clip_ratio: float = 0.2):
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.group_size = group_size  # Optimized 4-8 range
        self.kl_coeff = kl_coeff
        self.clip_ratio = clip_ratio  # Standard 0.2 clip epsilon
        self.ref_policy = None
        self.group_stats_buffer = deque(maxlen=1000)  # Track group statistics
        
    def set_reference_policy(self, ref_policy):
        """Set reference policy for KL divergence"""
        self.ref_policy = ref_policy
        
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-relative advantages with improved stability"""
        # Enhanced group normalization with running statistics
        group_mean = torch.mean(rewards)
        group_std = torch.std(rewards) + 1e-8
        
        # Store group statistics for analysis
        self.group_stats_buffer.append({
            'mean': group_mean.item(),
            'std': group_std.item(),
            'size': len(rewards)
        })
        
        # Improved advantage computation with clipping
        advantages = (rewards - group_mean) / group_std
        advantages = torch.clamp(advantages, -5.0, 5.0)  # Prevent extreme values
        return advantages
        
    def update(self, states, actions, old_log_probs, rewards, trajectory=None, use_mixed_precision=False, scaler=None):
        """GRPO update step with mixed precision support"""
        # Compute advantages using group statistics
        advantages = self.compute_advantages(rewards)
        
        # For continuous actions, we need to do a proper forward pass to get gradients
        if hasattr(self.policy_net, 'continuous_actions') and self.policy_net.continuous_actions and trajectory:
            if use_mixed_precision and scaler:
                with torch.cuda.amp.autocast():
                    # Get the current log probabilities by re-running the network
                    current_log_probs = []
                    
                    for i, traj_data in enumerate(trajectory):
                        vehicle = traj_data['vehicle']
                        all_vehicles = traj_data['all_vehicles']
                        action = traj_data['action']
                        
                        # Re-run the policy network
                        comm_vector = torch.zeros(8).to(actions.device)
                        mean, log_std = self.policy_net(all_vehicles, vehicle, comm_vector)
                        std = torch.exp(log_std.clamp(-5, 2))
                        dist = torch.distributions.Normal(mean, std)
                        log_prob = dist.log_prob(action).sum()
                        current_log_probs.append(log_prob)
                    
                    if current_log_probs:
                        current_log_probs = torch.stack(current_log_probs)
                        
                        # Compute ratio
                        ratio = torch.exp(current_log_probs - old_log_probs.detach())
                        
                        # Compute clipped objective
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        kl_div = torch.tensor(0.0)
                        total_loss = policy_loss
                    else:
                        # Fallback if no valid trajectories
                        policy_loss = torch.tensor(0.0, requires_grad=True)
                        kl_div = torch.tensor(0.0)
                        total_loss = policy_loss
                
                # Mixed precision backward pass
                if total_loss.requires_grad:
                    self.optimizer.zero_grad()
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
            else:
                # Regular precision for continuous actions
                current_log_probs = []
                
                for i, traj_data in enumerate(trajectory):
                    vehicle = traj_data['vehicle']
                    all_vehicles = traj_data['all_vehicles']
                    action = traj_data['action']
                    
                    # Re-run the policy network
                    comm_vector = torch.zeros(8).to(actions.device)
                    mean, log_std = self.policy_net(all_vehicles, vehicle, comm_vector)
                    std = torch.exp(log_std.clamp(-5, 2))
                    dist = torch.distributions.Normal(mean, std)
                    log_prob = dist.log_prob(action).sum()
                    current_log_probs.append(log_prob)
                
                if current_log_probs:
                    current_log_probs = torch.stack(current_log_probs)
                    
                    # Compute ratio
                    ratio = torch.exp(current_log_probs - old_log_probs.detach())
                    
                    # Compute clipped objective
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    kl_div = torch.tensor(0.0)
                    total_loss = policy_loss
                else:
                    # Fallback if no valid trajectories
                    policy_loss = torch.tensor(0.0, requires_grad=True)
                    kl_div = torch.tensor(0.0)
                    total_loss = policy_loss
                
                # Regular backward pass
                if total_loss.requires_grad:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                    self.optimizer.step()
        else:
            # Get current policy outputs for discrete actions
            if use_mixed_precision and scaler:
                with torch.cuda.amp.autocast():
                    current_outputs = self.policy_net(states)
                    current_log_probs = self.get_log_probs(current_outputs, actions)
                    
                    # Compute ratio
                    ratio = torch.exp(current_log_probs - old_log_probs.detach())
                    
                    # Compute clipped objective
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # KL divergence regularization
                    if self.ref_policy is not None:
                        ref_outputs = self.ref_policy(states)
                        ref_log_probs = self.get_log_probs(ref_outputs, actions)
                        kl_div = torch.mean(current_log_probs - ref_log_probs.detach())
                        total_loss = policy_loss + self.kl_coeff * kl_div
                    else:
                        total_loss = policy_loss
                        kl_div = torch.tensor(0.0)
                
                # Mixed precision backward pass
                if total_loss.requires_grad:
                    self.optimizer.zero_grad()
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
            else:
                # Regular precision for discrete actions
                current_outputs = self.policy_net(states)
                current_log_probs = self.get_log_probs(current_outputs, actions)
                
                # Compute ratio
                ratio = torch.exp(current_log_probs - old_log_probs.detach())
                
                # Compute clipped objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # KL divergence regularization
                if self.ref_policy is not None:
                    ref_outputs = self.ref_policy(states)
                    ref_log_probs = self.get_log_probs(ref_outputs, actions)
                    kl_div = torch.mean(current_log_probs - ref_log_probs.detach())
                    total_loss = policy_loss + self.kl_coeff * kl_div
                else:
                    total_loss = policy_loss
                    kl_div = torch.tensor(0.0)
                
                # Regular backward pass
                if total_loss.requires_grad:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                    self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item() if hasattr(policy_loss, 'item') else 0.0,
            'kl_div': kl_div.item() if hasattr(kl_div, 'item') else 0.0,
            'total_loss': total_loss.item() if hasattr(total_loss, 'item') else 0.0
        }
    
    def get_log_probs(self, outputs, actions):
        """Extract log probabilities from network outputs"""
        # Assuming outputs are logits for continuous actions
        mean, log_std = outputs.chunk(2, dim=-1)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(-1)

# ============================================================================
# COMMUNICATION PROTOCOL
# ============================================================================

class VehicleCommunication:
    """Learned communication protocol between vehicles"""
    def __init__(self, message_dim: int = 8, max_range: float = 100.0):
        self.message_dim = message_dim
        self.max_range = max_range
        
        # Message encoder/decoder
        self.message_encoder = nn.Sequential(
            nn.Linear(32, 16),  # Vehicle state -> message
            nn.Tanh(),
            nn.Linear(16, message_dim)
        )
        
        self.message_decoder = nn.Sequential(
            nn.Linear(message_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 8)  # Message -> action influence
        )
        
    def encode_message(self, vehicle_state: torch.Tensor) -> torch.Tensor:
        """Encode vehicle state into communication message"""
        return self.message_encoder(vehicle_state)
        
    def decode_message(self, message: torch.Tensor) -> torch.Tensor:
        """Decode received message into action influence"""
        return self.message_decoder(message)
        
    def broadcast_message(self, sender, vehicles: List, message: torch.Tensor):
        """Broadcast message to nearby vehicles"""
        for vehicle in vehicles:
            if vehicle.id != sender.id:
                distance = math.sqrt((vehicle.x - sender.x)**2 + (vehicle.y - sender.y)**2)
                if distance <= self.max_range:
                    # Attenuate message by distance
                    attenuation = max(0, 1 - distance / self.max_range)
                    attenuated_message = message * attenuation
                    
                    # Store in vehicle's message buffer
                    if not hasattr(vehicle, 'received_messages'):
                        vehicle.received_messages = []
                    vehicle.received_messages.append(attenuated_message)
        
    def aggregate_messages(self, vehicle) -> torch.Tensor:
        """Aggregate all received messages for a vehicle"""
        if not hasattr(vehicle, 'received_messages') or not vehicle.received_messages:
            device = next(self.message_encoder.parameters()).device
            return torch.zeros(self.message_dim, device=device)
            
        # Simple aggregation: mean of all messages
        messages = torch.stack(vehicle.received_messages)
        vehicle.received_messages = []  # Clear buffer
        return torch.mean(messages, dim=0)

# ============================================================================
# ADVANCED TRAFFIC TRANSFORMER WITH GNN + COMMUNICATION
# ============================================================================

class AdvancedTrafficTransformer(nn.Module):
    """Advanced transformer with GNN and communication"""
    def __init__(self, gnn_features: int = 64, d_model: int = 256, 
                 n_heads: int = 8, n_layers: int = 4, continuous_actions: bool = True):
        super().__init__()
        
        self.continuous_actions = continuous_actions
        self.gnn = GraphSAGEVehicleNet(hidden_dim=128, num_neighbors=20)  # Optimized GraphSAGE
        
        # Communication networks as proper submodules
        self.comm_message_encoder = nn.Sequential(
            nn.Linear(32, 16),  # Vehicle state -> message
            nn.Tanh(),
            nn.Linear(16, 8)
        )
        
        self.comm_message_decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 8)  # Message -> action influence
        )
        
        # Input projection
        self.input_projection = nn.Linear(gnn_features + 8, d_model)  # GNN + communication
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4) 
            for _ in range(n_layers)
        ])
        
        if continuous_actions:
            # Continuous action outputs: [acceleration, steering, communication_intensity]
            self.action_mean = nn.Linear(d_model, 3)
            self.action_log_std = nn.Linear(d_model, 3)
        else:
            # Discrete action outputs
            self.action_head = nn.Linear(d_model, 4)
            
    def forward(self, vehicles: List, ego_vehicle, received_messages: torch.Tensor = None):
        """Forward pass with GNN and communication"""
        # Get GNN representation
        graph_data = self.gnn.create_vehicle_graph(vehicles, ego_vehicle)
        gnn_output = self.gnn(graph_data)
        
        # Process communication
        if received_messages is None:
            device = gnn_output.device
            received_messages = torch.zeros(8, device=device)
        
        # Ensure communication is on the right device
        device = gnn_output.device
        received_messages = received_messages.to(device)
        comm_influence = self.comm_message_decoder(received_messages)
        comm_influence = comm_influence.to(device)  # Ensure same device
        
        # Combine GNN + communication
        combined_input = torch.cat([gnn_output.squeeze(), comm_influence], dim=-1)
        x = self.input_projection(combined_input)
        
        # For transformer blocks, we need a sequence dimension - create a dummy sequence of length 1
        x = x.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions: [1, 1, d_model]
        
        # Apply transformer layers
        for block in self.transformer_blocks:
            x, _ = block(x)
        
        # Remove sequence dimension and get final features
        x = x.squeeze(1).squeeze(0)  # Back to [d_model]
        
        # Generate actions
        if self.continuous_actions:
            mean = self.action_mean(x)
            log_std = self.action_log_std(x)
            return mean, log_std
        else:
            return self.action_head(x)
    
    def get_communication_message(self, vehicles: List, ego_vehicle) -> torch.Tensor:
        """Generate communication message based on current state"""
        # Encode ego vehicle state for communication
        device = next(self.comm_message_encoder.parameters()).device
        ego_state = torch.FloatTensor([
            ego_vehicle.x / 1200.0, ego_vehicle.y / 800.0,
            ego_vehicle.speed / 45.0, ego_vehicle.acceleration / 8.0,
            ego_vehicle.heading / (2 * math.pi), ego_vehicle.collision_risk,
            ego_vehicle.lane_deviation() / 10.0, ego_vehicle.target_speed / 45.0,
            *[0.0] * 24  # Pad to 32 features
        ]).to(device)
        return self.comm_message_encoder(ego_state)

# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Main interface with advanced ML and 3D visualization options"""
    print("🚗 Modern Autonomous Vehicle Traffic Simulation with Panda3D")
    print("=" * 60)
    print("Testing: Can autonomous vehicles eliminate traffic lights?")
    print("Technology: Advanced ML with GRPO + GNN + Communication + 3D Visualization")
    if PANDA3D_AVAILABLE:
        print("🎮 Panda3D 3D Graphics: AVAILABLE")
    else:
        print("📦 Panda3D 3D Graphics: Not installed (pip install panda3d)")
    print("💡 Note: Trained models work for both 2D and 3D visualization")
    print("=" * 60)
    
    
    choice = input("""
Choose mode:
1. 🧠 Train AI model (with visualization - slower)
2. ⚡ Fast train AI model (no visualization - faster)
3. 🚗 Run 2D simulation (requires trained model)
4. 🎮 Run 3D simulation (requires trained model)

Enter choice (1-4): """).strip()
    
    if choice == '1':
        episodes = int(input("Training episodes (50-150 recommended): ") or "100")
        
        # Choose visualization type for training
        if PANDA3D_AVAILABLE:
            viz_choice = input("Training visualization: (2)D pygame or (3)D Panda3D? [3]: ").strip() or "3"
            use_3d = viz_choice == "3"
        else:
            use_3d = False
            print("Using 2D pygame visualization (Panda3D not available)")
        
        trainer = TrafficTrainer(use_advanced_ml=True)
        trainer.train(episodes, use_visualization=True, use_3d=use_3d)
        
    elif choice == '2':
        episodes = int(input("Training episodes (50-150 recommended): ") or "100")
        
        print("\n🧠 Fast training (no visualization)...")
        trainer = TrafficTrainer(use_advanced_ml=True)
        trainer.train(episodes, use_visualization=False)
        print("✅ Training completed! Model saved as 'traffic_transformer_coord_advanced.pth'")
        
    elif choice == '3':
        sim = TrafficSimulation()
        if sim.running:
            sim.run()
            
    elif choice == '4':
        if not PANDA3D_AVAILABLE:
            print("❌ Panda3D not available! Install with: pip install panda3d")
            print("Falling back to 2D simulation...")
            sim = TrafficSimulation()
            if sim.running:
                sim.run()
        else:
            sim = TrafficSimulation3D()
            if sim.running:
                sim.run_simulation()
    
    else:
        print("❌ Invalid choice! Please run again and choose 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()