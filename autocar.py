"""
Modern Autonomous Vehicle Traffic Simulation
Testing hypothesis: Can autonomous vehicles eliminate traffic lights through coordination?

Architecture:
- Transformer neural networks for decision making
- Kalman filters for state estimation
- Realistic vehicle physics and sensors
- Multi-agent reinforcement learning
"""

import torch
import torch.nn as nn   
import torch.nn.functional as F
import torch.optim as optim
import pygame
import numpy as np
import math
import random
from typing import List, Dict, Tuple
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
    
    def get_next_available_slot(self, duration: float = 2.0) -> float:
        """Get the next available slot for reservation"""
        if not self.reservations:
            return self.current_time
        
        # Sort by start time
        sorted_res = sorted(self.reservations, key=lambda x: x[1])
        
        # Check if we can fit before first reservation
        if sorted_res[0][1] - self.current_time >= duration:
            return self.current_time
        
        # Find gap between reservations
        for i in range(len(sorted_res) - 1):
            gap_start = sorted_res[i][2]
            gap_end = sorted_res[i + 1][1]
            if gap_end - gap_start >= duration:
                return gap_start
        
        # Use slot after last reservation
        return sorted_res[-1][2]
    
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
        
    def get_state(self) -> Dict[str, float]:
        return {
            'x': self.x[0], 'y': self.x[1], 
            'vx': self.x[2], 'vy': self.x[3],
            'speed': math.sqrt(self.x[2]**2 + self.x[3]**2),
            'heading': math.atan2(self.x[3], self.x[2])
        }

# ============================================================================
# AUTONOMOUS VEHICLE
# ============================================================================

class AutonomousVehicle:
    """Modern autonomous vehicle with transformer + Kalman filter"""
    def __init__(self, x: float, y: float, heading: float):
        self.id = random.randint(10000, 99999)
        
        # Physical state
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = random.uniform(1.0, 3.0)  # Start with small initial speed instead of 0
        self.acceleration = 0.0
        self.max_speed = 25.0
        self.max_acceleration = 3.0
        self.max_deceleration = -8.0
        
        # Autonomous systems
        self.kalman_filter = KalmanFilter()
        self.transformer = None  # Set externally
        
        # Sensors
        self.sensor_range = 100.0
        self.sensor_noise_std = 0.1
        
        # State tracking
        self.target_speed = random.uniform(20, 25)
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
        
        return torch.FloatTensor(features)
    
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
        Returns the (x,y) coordinate of the centre‑line the vehicle
        is *supposed* to be on, based on its original heading.
        """
        if abs(self.heading) < 0.2:                  # east‑bound
            return self.x, 300 if self.y < 400 else 500
        if abs(self.heading - math.pi) < 0.2:        # west‑bound
            return self.x, 300 if self.y < 400 else 500
        if abs(self.heading - math.pi/2) < 0.2:      # south‑bound
            return 400 if self.x < 600 else 800, self.y
        return 400 if self.x < 600 else 800, self.y # north‑bound

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
            return torch.zeros(seq_len, 64)
            
        if len(self.observation_history) < seq_len:
            sequence = [torch.zeros_like(self.observation_history[0]) for _ in range(seq_len - len(self.observation_history))]
            sequence.extend(list(self.observation_history))
        else:
            sequence = list(self.observation_history)[-seq_len:]
        return torch.stack(sequence)
    
    def make_decision(self, observations: torch.Tensor = None, q_values: torch.Tensor = None) -> int:
        """Make driving decision - supports both individual and batched inference"""
        if self.transformer is None:
            return random.randint(0, 3)
        
        if observations is not None:
            self.observation_history.append(observations)
        
        # Epsilon-greedy exploration during training with bias toward acceleration
        if hasattr(self, '_epsilon') and random.random() < self._epsilon:
            # Bias exploration toward acceleration (60% chance) to encourage movement
            if random.random() < 0.6:
                return 0  # Accelerate
            else:
                return random.randint(1, 3)  # Other actions
        
        # If q_values provided (from batch inference), use them
        if q_values is not None:
            action = torch.argmax(q_values).item()
            self.action_history.append(action)
            return action
        
        # Fallback to individual inference (slower)
        if observations is None:
            return random.randint(0, 3)
            
        input_sequence = self.prepare_sequence().unsqueeze(0)
        
        with torch.no_grad():
            q_vals, _ = self.transformer(input_sequence)
            action = torch.argmax(q_vals).item()
        
        self.action_history.append(action)
        return action
    
    def execute_action(self, action: int, dt: float):
        """Execute control action with collision avoidance"""
        # Action mapping: 0=accelerate, 1=maintain, 2=brake, 3=cooperate
        base_acceleration = 0
        
        if action == 0:  # Accelerate
            base_acceleration = min(2.0, self.max_acceleration)
        elif action == 1:  # Maintain
            base_acceleration = 0.0
        elif action == 2:  # Brake
            base_acceleration = max(-3.0, self.max_deceleration)
        else:  # Cooperate (yield)
            base_acceleration = max(-1.0, self.max_deceleration)
            self.behavior_mode = 'cooperative'
        
        # Override acceleration if collision risk is high
        if self.collision_risk > 0.6:
            # Emergency braking to avoid collision
            base_acceleration = min(base_acceleration, -2.0)
        elif self.collision_risk > 0.3:
            # Gentle deceleration for safety
            base_acceleration = min(base_acceleration, -0.5)
        
        self.acceleration = base_acceleration
        
        # Update speed
        self.speed = max(0, min(self.max_speed, self.speed + self.acceleration * dt))
        
        # Enhanced lane following with collision avoidance
        # Force vehicles to maintain their spawn direction and avoid clustering
        
        # Determine which road and direction vehicle should be on
        if abs(self.heading) < 0.2:  # Heading East (0°)
            target_y = 300 if abs(self.y - 300) < abs(self.y - 500) else 500
            # Lane discipline: keep slightly offset from center for opposing traffic
            lane_offset = 8 if target_y == 300 else -8
            target_y += lane_offset
            
        elif abs(self.heading - math.pi) < 0.2:  # Heading West (180°)
            target_y = 300 if abs(self.y - 300) < abs(self.y - 500) else 500
            # Lane discipline: opposite offset from eastbound traffic
            lane_offset = -8 if target_y == 300 else 8
            target_y += lane_offset
            
        elif abs(self.heading - math.pi/2) < 0.2:  # Heading South (90°)
            target_x = 400 if abs(self.x - 400) < abs(self.x - 800) else 800
            lane_offset = 8 if target_x == 400 else -8
            target_x += lane_offset
            
        elif abs(self.heading + math.pi/2) < 0.2:  # Heading North (-90°)
            target_x = 400 if abs(self.x - 400) < abs(self.x - 800) else 800
            lane_offset = -8 if target_x == 400 else 8
            target_x += lane_offset
        
        # Gentle lane correction to prevent off-road driving
        if abs(self.heading) < 0.2 or abs(self.heading - math.pi) < 0.2:  # Horizontal
            if abs(self.y - target_y) > 15:
                self.y += 0.8 * dt * (target_y - self.y)
        else:  # Vertical
            if abs(self.x - target_x) > 15:
                self.x += 0.8 * dt * (target_x - self.x)
        
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
        
        # Update collision risk
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
                    if min_distance < 30 and not nearest_intersection.is_vehicle_cleared(self.id):
                        self.waiting_time += dt
                        self.target_speed = min(self.target_speed, 10.0)
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
                    
                    if dist_cpa < 10.0:  # Safety distance
                        min_ttc = min(min_ttc, t_cpa)
        
        if min_ttc < float('inf'):
            self.collision_risk = max(0, 1 - min_ttc / 5.0)
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
        
        # Color based on behavior
        colors = {
            'normal': (100, 255, 150),
            'aggressive': (255, 150, 100),
            'conservative': (150, 150, 255),
            'cooperative': (255, 255, 100)
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
                color = (255, 100, 100)  # Red - congested
            elif occupancy > 1:
                color = (255, 200, 100)  # Orange - moderate
            elif occupancy > 0:
                color = (255, 255, 100)  # Yellow - light traffic
            else:
                color = (100, 255, 100)  # Green - clear
                
            pygame.draw.rect(self.screen, color, 
                           (intersection['x'] - 30, intersection['y'] - 30, 60, 60))
            
            # Occupancy number
            if occupancy > 0:
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
                
                # Vehicle color based on behavior
                colors = {
                    'normal': (100, 255, 150),
                    'aggressive': (255, 150, 100),
                    'conservative': (150, 150, 255),
                    'cooperative': (255, 255, 100)
                }
                color = colors.get(vehicle.behavior_mode, (200, 200, 200))
                
                # Override color if badly off-lane
                if abs(vehicle.lane_deviation()) > 3.0:
                    color = (255, 80, 80)  # Red for poor lane keeping
                
                # Draw as small circle
                pygame.draw.circle(self.screen, color, (int(scaled_x), int(scaled_y)), 4)
                
                # Collision risk indicator
                if vehicle.collision_risk > 0.3:
                    risk_color = (255, int(255 * (1 - vehicle.collision_risk)), 0)
                    pygame.draw.circle(self.screen, risk_color, 
                                     (int(scaled_x), int(scaled_y)), 
                                     int(4 + vehicle.collision_risk * 6), 1)
    
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
        self.clock.tick(30)  # 30 FPS for training visualization
    
    def close(self):
        """Clean up pygame"""
        pygame.quit()

# ============================================================================
# TRAINING SYSTEM (UPDATED)
# ============================================================================

class TrafficTrainer:
    """Training system for transformer models"""
    def __init__(self):
        self.transformer = TrafficTransformer()
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=0.0001)
        self.memory = deque(maxlen=50000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        
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
        """Spawn vehicle at random entry point - properly aligned to roads"""
        # Entry points aligned to road centers
        entry_points = [
            # Horizontal roads (y=300, y=500)
            (100, 300, 0),    # West entrance, heading East
            (1100, 300, math.pi),  # East entrance, heading West  
            (100, 500, 0),    # West entrance, heading East
            (1100, 500, math.pi),  # East entrance, heading West
            # Vertical roads (x=400, x=800)  
            (400, 100, math.pi/2),   # North entrance, heading South
            (400, 700, -math.pi/2),  # South entrance, heading North
            (800, 100, math.pi/2),   # North entrance, heading South
            (800, 700, -math.pi/2),  # South entrance, heading North
        ]
        
        x, y, heading = random.choice(entry_points)
        
        # Add variation to spread out spawning and reduce clustering
        if heading == 0 or heading == math.pi:  # Horizontal roads
            x += random.uniform(-50, 50)  # More variation along road to spread vehicles
            y += random.uniform(-12, 12)  # Slight perpendicular variation for lanes
        else:  # Vertical roads
            x += random.uniform(-12, 12)  # Slight perpendicular variation for lanes  
            y += random.uniform(-50, 50)  # More variation along road to spread vehicles
        
        vehicle = AutonomousVehicle(x, y, heading)
        vehicle.transformer = self.transformer
        vehicle.target_speed = random.uniform(18, 25)  # Match simulation range
        vehicle._epsilon = self.epsilon  # Pass exploration rate for training
        vehicle._trainer = self  # Reference for traffic light mode
        
        # Reset progress tracking for fresh spawn
        vehicle.prev_exit_dist = vehicle.calculate_exit_distance()
        
        return vehicle
    
    def calculate_reward(self, vehicle: AutonomousVehicle, dt: float) -> float:
        """Calculate reward for vehicle's actions"""
        reward = 0
        
        # MAJOR: Forward progress reward (encourages movement toward exits)
        if vehicle.speed > 2.0:  # Reward for moving
            reward += 2.0 * dt
        elif vehicle.speed > 0.5:  # Small reward for slow movement
            reward += 0.5 * dt
        else:  # Penalty for stopping
            reward -= 1.0 * dt
        
        # Delta progress reward - only reward actual progress toward exits
        current_exit_dist = vehicle.calculate_exit_distance()
        progress_delta = vehicle.prev_exit_dist - current_exit_dist
        reward += progress_delta / 50.0  # Doubled reward for forward progress
        vehicle.prev_exit_dist = current_exit_dist
        
        # Strong exit bonus - encourage reaching boundaries
        if current_exit_dist < 30:  # Very close to exit
            reward += 5.0 * dt  # Big bonus for being near exit
        elif current_exit_dist < 60:  # Approaching exit
            reward += 2.0 * dt  # Medium bonus
        
        # Speed efficiency (maintain reasonable speed)
        if vehicle.speed > 15.0:  # Encourage decent speed
            reward += 1.0 * dt
        
        # Cooperation bonus
        if vehicle.behavior_mode == 'cooperative':
            reward += 0.2
        
        # Lane‑keeping penalty  ---------------------------
        deviation = abs(vehicle.lane_deviation())      # metres
        
        # Reduce lane-keeping penalties near exits to allow vehicles to exit
        near_exit = (vehicle.x < 100 or vehicle.x > 1100 or 
                    vehicle.y < 100 or vehicle.y > 700)
        
        if deviation > 1.0:                            # small tolerance
            penalty_multiplier = 0.1 if near_exit else 0.3  # Much lower penalty near exits
            reward -= min(deviation, 5.0) * penalty_multiplier * dt
            # big smack if they fully drift off the road (but not near exits)
            if deviation > 8.0 and not near_exit:
                reward -= 2.0
        else:
            reward += 0.1 * dt
        
        # Safety penalty (but not too harsh)
        if vehicle.collision_risk > 0.5:
            reward -= 2.0 * vehicle.collision_risk
        
        # Intersection coordination bonus
        if vehicle.has_reservation:
            reward += 0.5 * dt
        
        # Clamp total reward to prevent Q-value explosion
        return np.clip(reward, -5.0, 10.0)
    
    def train_step(self):
        """Single training step"""
        if len(self.memory) < 64:
            return 0
        
        # Set model to training mode
        self.transformer.train()
        
        # Filter memory for entries with sufficient history (avoid alignment issues)
        valid_memory = [exp for exp in self.memory if len(exp) == 5 and 
                       exp[0].shape[0] == 5 and exp[3].shape[0] == 5]
        if len(valid_memory) < 32:  # Use smaller batch if needed
            return 0
        
        batch_size = min(64, len(valid_memory))
        batch = random.sample(valid_memory, batch_size)
        
        # Get device from model parameters
        device = next(self.transformer.parameters()).device
        
        # Decompress compressed states with proper mapping
        states = []
        next_states = []
        for exp in batch:
            # Decompress: uint8 [0,255] → float32 [-1,1] 
            prev_state_decompressed = torch.from_numpy(exp[0]).float() / 255.0 * 2.0 - 1.0
            curr_state_decompressed = torch.from_numpy(exp[3]).float() / 255.0 * 2.0 - 1.0
            states.append(prev_state_decompressed)
            next_states.append(curr_state_decompressed)
        
        states = torch.stack(states).to(device)
        actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long).to(device)
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor([exp[4] for exp in batch], dtype=torch.bool).to(device)
        
        current_q_values, _ = self.transformer(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values, _ = self.transformer(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (0.95 * max_next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
        self.optimizer.step()
        
        # Set back to eval mode for inference
        self.transformer.eval()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def run_episode(self, viz: 'TrainingVisualization' = None, episode_length: int = 300) -> Dict:
        """Run training episode (5 seconds simulated time, optimized)"""
        self.vehicles = [self.spawn_vehicle() for _ in range(12)]  # Match simulation initial count
        
        episode_rewards = []
        total_loss = 0
        training_steps = 0
        vehicles_passed = 0
        
        # Use consistent time step like simulation
        dt = 1/60  # Fixed 60 FPS to match simulation target
        
        for step in range(episode_length):
            # Update intersection managers (always update time, conditionally handle reservations)
            for intersection in self.intersections:
                # Always update time for ETA calculations
                intersection['manager'].current_time += dt
                
                # Only process reservations in coordination mode
                if not self.use_traffic_lights:
                    # Clean expired reservations when in coordination mode
                    intersection['manager'].reservations = [
                        (vid, s, e) for vid, s, e in intersection['manager'].reservations 
                        if e > intersection['manager'].current_time
                    ]
                
                intersection['light'].update(dt)
            
            # Count vehicles before update
            initial_count = len(self.vehicles)
            
            # Batch inference for all vehicles (GPU optimized)
            valid_vehicles = []
            sequences = []
            
            for vehicle in self.vehicles:
                vehicle._epsilon = self.epsilon
                
                # Update physics first but don't trim history yet
                vehicle.kalman_filter.predict(dt)
                
                # Simulate noisy GPS measurement
                noisy_position = np.array([
                    vehicle.x + np.random.normal(0, 0.5),
                    vehicle.y + np.random.normal(0, 0.5)
                ])
                vehicle.kalman_filter.update(noisy_position)
                
                # Handle intersection coordination
                vehicle.handle_intersection_coordination([i['manager'] for i in self.intersections], dt)
                
                # Get observations and add to history (keep growing for experience collection)
                observations = vehicle.get_observations(self.vehicles, [i['manager'] for i in self.intersections])
                vehicle.observation_history.append(observations)
                
                # Update collision risk
                vehicle.calculate_collision_risk(self.vehicles)
                
                # Collect sequences for batch inference
                if len(vehicle.observation_history) >= 5:
                    sequences.append(vehicle.prepare_sequence())
                    valid_vehicles.append(vehicle)
            
            # Batch transformer inference
            if sequences:
                device = next(self.transformer.parameters()).device
                batch_sequences = torch.stack(sequences).to(device)
                
                with torch.no_grad():
                    batch_q_values, _ = self.transformer(batch_sequences)
                
                # Distribute actions back to vehicles
                for vehicle, q_values in zip(valid_vehicles, batch_q_values):
                    action = vehicle.make_decision(q_values=q_values.cpu())
                    vehicle.execute_action(action, dt)
                    
                    # Collect experience when enough history available
                    if step % 3 == 0 and len(vehicle.observation_history) >= 6:
                        # Create proper state transition: prev 5 frames -> action -> next 5 frames
                        prev_state = list(vehicle.observation_history)[-6:-1]  # frames t-5 to t-1
                        current_state = list(vehicle.observation_history)[-5:]  # frames t-4 to t
                        
                        reward = self.calculate_reward(vehicle, dt)
                        episode_rewards.append(reward)
                        done = vehicle.collision_risk > 0.8 or step > 250
                        
                        if len(vehicle.action_history) > 0:
                            # Compress memory storage with proper quantization
                            prev_tensor = torch.stack(prev_state)
                            curr_tensor = torch.stack(current_state)
                            
                            # float32 [-1,1] → uint8 [0,255] with proper mapping
                            prev_compressed = (((prev_tensor.clamp(-1,1)+1)*0.5)*255).round().to(torch.uint8).cpu().numpy()
                            curr_compressed = (((curr_tensor.clamp(-1,1)+1)*0.5)*255).round().to(torch.uint8).cpu().numpy()
                            
                            self.memory.append((
                                prev_compressed, vehicle.action_history[-1], 
                                reward, curr_compressed, done
                            ))
            
            # Handle vehicles without enough history
            for vehicle in self.vehicles:
                if vehicle not in valid_vehicles:
                    action = random.randint(0, 3)  # Random until enough history
                    vehicle.execute_action(action, dt)
            
            # Remove vehicles that exit and count throughput
            exited_vehicles = [v for v in self.vehicles 
                             if (v.x < 50 or v.x > 1150 or v.y < 50 or v.y > 750)]  # Expanded exit boundaries
            
            # Optional debug logging (disabled by default)
            debug_logging = False
            if debug_logging and exited_vehicles and step % 30 == 0:
                console.print(f"[green]Episode {step//30}: {len(exited_vehicles)} vehicles exited![/green]")
            
            vehicles_passed += len(exited_vehicles)
            
            # Debug output every 60 steps (1 second) to track progress
            if step % 60 == 0 and step > 0:
                avg_speed = np.mean([v.speed for v in self.vehicles]) if self.vehicles else 0
                avg_exit_dist = np.mean([v.calculate_exit_distance() for v in self.vehicles]) if self.vehicles else 0
                print(f"T+{step/60:.0f}s: {len(self.vehicles)} vehicles, avg_speed={avg_speed:.1f}, avg_exit_dist={avg_exit_dist:.0f}, passed={vehicles_passed}")
            
            self.vehicles = [v for v in self.vehicles 
                           if not (v.x < 50 or v.x > 1150 or v.y < 50 or v.y > 750)]  # Use same expanded boundaries
            
            # Maintain vehicle count (spawn new ones) - match simulation density
            if len(self.vehicles) < 20 and step % 15 == 0:
                self.vehicles.append(self.spawn_vehicle())
            
            # Training (start early and more frequent)
            if step % 10 == 0 and len(self.memory) > 64:
                loss = self.train_step()
                total_loss += loss
                training_steps += 1
                
            # Update visualization periodically
            if viz and step % 10 == 0:
                viz.update_display(self.vehicles, self.intersections)
        
        # Ensure epsilon decay happens even without training
        if self.epsilon > self.epsilon_min:
            # Decay epsilon based on episode progress
            episode_decay = 0.995  # More aggressive decay per episode
            self.epsilon = max(self.epsilon_min, self.epsilon * episode_decay)
        
        return {
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'avg_loss': total_loss / max(training_steps, 1),
            'throughput': vehicles_passed,
            'memory_size': len(self.memory),
            'training_steps': training_steps
        }
    
    def train(self, episodes: int = 100, use_visualization: bool = True):
        """Train the model with optional pygame visualization"""
        if use_visualization:
            # Initialize pygame visualization
            viz = TrainingVisualization()
            viz.start_training(episodes)
            console.print("🎮 [bold green]Training with live visualization[/bold green] - watch the learning process!")
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
            
            results = self.run_episode(viz)
            
            # Record metrics
            self.training_history['rewards'].append(results['avg_reward'])
            self.training_history['throughput'].append(results['throughput'])
            self.training_history['loss'].append(results['avg_loss'])
            recent_rewards.append(results['avg_reward'])
            
            # Update visualization with results
            if viz:
                viz.update_episode(episode, results, self.epsilon)
            
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
                      f"ε: {self.epsilon:.3f} | "
                      f"Mem: {results['memory_size']:4d} | "
                      f"Steps: {results['training_steps']:2d} | "
                      f"Time: {episode_time:.1f}s")
                
                # Learning progress indicators
                if episode >= 10:
                    recent_avg = np.mean(recent_rewards)
                    trend = "📈" if recent_avg > results['avg_reward'] - 1 else "📊" if recent_avg > results['avg_reward'] - 3 else "📉"
                    print(f"   Recent avg reward: {recent_avg:.2f} {trend}")
                
                # Memory and exploration status
                memory_pct = len(self.memory) / 50000 * 100
                print(f"   Memory: {len(self.memory):5d}/50000 ({memory_pct:.1f}%) | "
                      f"Exploration: {self.epsilon:.1%}")
                
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
                    'model_state_dict': self.transformer.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'training_history': self.training_history,
                    'episode': episode
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
        print(f"Final reward: {results['avg_reward']:.2f}")
        print(f"Final exploration rate: {self.epsilon:.1%}")
        print(f"Memory utilization: {len(self.memory):,}/50,000 experiences")
        print(f"Total training steps: {sum(r.get('training_steps', 0) for r in [results])}")
        
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
        model_path = f'traffic_transformer_{experiment_tag}.pth'
        torch.save({
            'model_state_dict': self.transformer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'final_performance': best_performance,
            'training_time': total_time,
            'use_traffic_lights': self.use_traffic_lights
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
    """Interactive traffic simulation"""
    def __init__(self, model_path: str = 'traffic_transformer_coord.pth'):
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption("Autonomous Vehicle Coordination")
        self.clock = pygame.time.Clock()
        
        # Load trained model
        self.transformer = TrafficTransformer()
        try:
            checkpoint = torch.load(model_path)
            self.transformer.load_state_dict(checkpoint['model_state_dict'])
            self.transformer.eval()
            print(f"✓ Loaded trained model from {model_path}")
        except FileNotFoundError:
            print(f"❌ No model found at {model_path}")
            print("Run training first!")
            self.running = False
            return
        
        # Environment - use consistent IntersectionManager objects
        self.vehicles = []
        self.intersection_managers = [
            IntersectionManager(400, 300),
            IntersectionManager(800, 300), 
            IntersectionManager(400, 500),
            IntersectionManager(800, 500)
        ]
        
        # Display metadata for rendering (separate from logic)
        self.intersection_display = [
            {'x': 400, 'y': 300, 'occupancy': 0},
            {'x': 800, 'y': 300, 'occupancy': 0},
            {'x': 400, 'y': 500, 'occupancy': 0},
            {'x': 800, 'y': 500, 'occupancy': 0}
        ]
        
        self.running = True
        self.total_time = 0
        self.metrics = {'throughput': 0, 'cooperation': 0}
        
        # Spawn initial vehicles
        for _ in range(12):
            self.spawn_vehicle()
    
    def spawn_vehicle(self):
        """Spawn vehicle with trained model"""
        entry_points = [
            (100, 300, 0), (1100, 300, math.pi),
            (400, 100, math.pi/2), (400, 700, -math.pi/2),
            (800, 100, math.pi/2), (800, 700, -math.pi/2)
        ]
        
        x, y, heading = random.choice(entry_points)
        x += random.uniform(-30, 30)
        y += random.uniform(-30, 30)
        
        vehicle = AutonomousVehicle(x, y, heading)
        vehicle.transformer = self.transformer
        vehicle.target_speed = random.uniform(18, 25)
        
        self.vehicles.append(vehicle)
    
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
                        if not (v.x < 50 or v.x > 1150 or v.y < 50 or v.y > 750)]
        self.metrics['throughput'] += initial_count - len(self.vehicles)
        
        # Maintain traffic flow
        if len(self.vehicles) < 20 and random.random() < 0.03:
            self.spawn_vehicle()
    
    def draw(self):
        """Draw simulation"""
        self.screen.fill((30, 30, 40))
        
        # Roads
        road_color = (60, 60, 75)
        pygame.draw.rect(self.screen, road_color, (0, 270, 1200, 60))
        pygame.draw.rect(self.screen, road_color, (0, 470, 1200, 60))
        pygame.draw.rect(self.screen, road_color, (370, 0, 60, 800))
        pygame.draw.rect(self.screen, road_color, (770, 0, 60, 800))
        
        # Intersections
        for intersection in self.intersection_display:
            color = (100, 100, 100)
            if intersection['occupancy'] > 0:
                color = (150, 150, 100)
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

# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Main interface"""
    print("🚗 Modern Autonomous Vehicle Traffic Simulation")
    print("=" * 50)
    print("Testing: Can autonomous vehicles eliminate traffic lights?")
    print("Technology: Transformer neural networks + Kalman filters")
    print("=" * 50)
    
    choice = input("""
Choose mode:
1. Train new model (with live visualization)
2. Train new model (text only)  
3. Run simulation (requires trained model)
4. Both (train then simulate)
5. 🆚 BASELINE COMPARISON: Train with traffic lights vs no lights

Enter choice (1-5): """).strip()
    
    if choice == '1':
        episodes = int(input("Training episodes (100-200 recommended): ") or "100")
        trainer = TrafficTrainer()
        trainer.train(episodes, use_visualization=True)
        
    elif choice == '2':
        episodes = int(input("Training episodes (100-200 recommended): ") or "100")
        trainer = TrafficTrainer()
        trainer.train(episodes, use_visualization=False)
        
    elif choice == '3':
        sim = TrafficSimulation()
        if sim.running:
            sim.run()
            
    elif choice == '4':
        episodes = int(input("Training episodes (100-200 recommended): ") or "100")
        use_viz = input("Use visualization for training? (y/N): ").strip().lower() == 'y'
        
        print("\n🧠 Training phase...")
        trainer = TrafficTrainer()
        trainer.train(episodes, use_visualization=use_viz)
        
        print("\n🚗 Simulation phase...")
        sim = TrafficSimulation()
        if sim.running:
            sim.run()
    
    elif choice == '5':
        episodes = int(input("Training episodes per mode (50-100 recommended): ") or "50")
        
        console.print(Panel.fit(
            "🆚 [bold blue]BASELINE COMPARISON EXPERIMENT[/bold blue]\n\n"
            f"Training {episodes} episodes with traffic lights, then {episodes} without\n"
            f"This will generate statistical comparison data",
            title="Baseline Comparison",
            border_style="blue"
        ))
        
        # Train with traffic lights
        console.print("\n🚦 [red]Training WITH traffic lights (baseline)...[/red]")
        trainer_lights = TrafficTrainer()
        trainer_lights.use_traffic_lights = True
        lights_results = trainer_lights.train(episodes, use_visualization=False)
        
        # Train without traffic lights
        console.print("\n🚗 [green]Training WITHOUT traffic lights (coordination)...[/green]")
        trainer_coord = TrafficTrainer()
        trainer_coord.use_traffic_lights = False
        coord_results = trainer_coord.train(episodes, use_visualization=False)
        
        # Compare results
        console.print("\n📊 [bold yellow]COMPARISON RESULTS:[/bold yellow]")
        lights_avg = np.mean(lights_results['throughput'][-10:]) if lights_results['throughput'] else 0
        coord_avg = np.mean(coord_results['throughput'][-10:]) if coord_results['throughput'] else 0
        
        table = Table(title="Traffic Light vs Coordination")
        table.add_column("Method", style="cyan")
        table.add_column("Avg Throughput", style="magenta")
        table.add_column("Best Episode", style="green")
        
        table.add_row("Traffic Lights", f"{lights_avg:.2f}", f"{max(lights_results['throughput']) if lights_results['throughput'] else 0:.2f}")
        table.add_row("Coordination", f"{coord_avg:.2f}", f"{max(coord_results['throughput']) if coord_results['throughput'] else 0:.2f}")
        
        console.print(table)
        
        if coord_avg > lights_avg:
            console.print("🎉 [bold green]Coordination WINS! Traffic lights can be eliminated![/bold green]")
        else:
            console.print("⚠️ [bold red]Traffic lights still needed. More training required.[/bold red]")
    
    else:
        console.print("[bold red]Invalid choice![/bold red]")

if __name__ == "__main__":
    main()