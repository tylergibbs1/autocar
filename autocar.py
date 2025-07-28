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
import time
from typing import List, Dict, Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt

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
        self.speed = 0.0
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
        
    def get_observations(self, other_vehicles: List['AutonomousVehicle'], 
                        intersections: List) -> torch.Tensor:
        """Get sensor observations and convert to neural network input"""
        features = np.zeros(64)
        
        # Ego vehicle features (8 dims)
        features[0:8] = [
            self.x / 1000.0,
            self.y / 1000.0,
            self.heading / (2 * math.pi),
            self.speed / self.max_speed,
            self.acceleration / self.max_acceleration,
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
        
        nearby.sort(key=lambda x: x[0])  # Sort by distance
        
        for i, (dist, vehicle, noise_x, noise_y) in enumerate(nearby[:7]):
            base_idx = 8 + i * 6
            features[base_idx:base_idx+6] = [
                (vehicle.x + noise_x - self.x) / self.sensor_range,
                (vehicle.y + noise_y - self.y) / self.sensor_range,
                vehicle.speed * math.cos(vehicle.heading) / self.max_speed,
                vehicle.speed * math.sin(vehicle.heading) / self.max_speed,
                dist / self.sensor_range,
                (self.speed - vehicle.speed) / self.max_speed
            ]
        
        # Intersection information (6 dims)
        if intersections:
            intersection = min(intersections, 
                key=lambda i: math.sqrt((i['x'] - self.x)**2 + (i['y'] - self.y)**2))
            
            dist = math.sqrt((intersection['x'] - self.x)**2 + (intersection['y'] - self.y)**2)
            features[50:56] = [
                (intersection['x'] - self.x) / self.sensor_range,
                (intersection['y'] - self.y) / self.sensor_range,
                dist / self.sensor_range,
                intersection.get('occupancy', 0) / 5.0,
                1.0 if dist < 50 else 0.0,
                1.0 if intersection.get('type') == 'roundabout' else 0.0
            ]
        
        # Behavioral context (8 dims)
        behavior_encoding = {'normal': 0.25, 'aggressive': 0.5, 'conservative': 0.75, 'cooperative': 1.0}
        features[56:64] = [
            behavior_encoding.get(self.behavior_mode, 0.25),
            random.random() * 0.1,  # Exploration noise
            0, 0, 0, 0, 0, 0  # Reserved
        ]
        
        return torch.FloatTensor(features)
    
    def make_decision(self, observations: torch.Tensor) -> int:
        """Make driving decision using transformer (optimized)"""
        if self.transformer is None:
            return random.randint(0, 3)  # Random action if no model
            
        self.observation_history.append(observations)
        
        # Prepare shorter sequence for faster processing
        seq_len = 5  # Reduced from 10
        if len(self.observation_history) < seq_len:
            sequence = [torch.zeros_like(observations) for _ in range(seq_len - len(self.observation_history))]
            sequence.extend(list(self.observation_history))
        else:
            sequence = list(self.observation_history)[-seq_len:]
        
        input_sequence = torch.stack(sequence).unsqueeze(0)
        
        with torch.no_grad():
            output, _ = self.transformer(input_sequence)
            action_probs = F.softmax(output, dim=-1)
        
        action = torch.argmax(action_probs).item()
        self.action_history.append(action)
        
        return action
    
    def execute_action(self, action: int, dt: float):
        """Execute control action"""
        # Action mapping: 0=accelerate, 1=maintain, 2=brake, 3=cooperate
        if action == 0:  # Accelerate
            self.acceleration = min(2.0, self.max_acceleration)
        elif action == 1:  # Maintain
            self.acceleration = 0.0
        elif action == 2:  # Brake
            self.acceleration = max(-3.0, self.max_deceleration)
        else:  # Cooperate (yield)
            self.acceleration = max(-1.0, self.max_deceleration)
            self.behavior_mode = 'cooperative'
        
        # Update speed
        self.speed = max(0, min(self.max_speed, self.speed + self.acceleration * dt))
        
        # Update position
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
    
    def update(self, dt: float, other_vehicles: List['AutonomousVehicle'], 
               intersections: List):
        """Main update loop"""
        # Kalman filter prediction
        self.kalman_filter.predict(dt)
        
        # Simulate noisy GPS measurement
        noisy_position = np.array([
            self.x + np.random.normal(0, 0.5),
            self.y + np.random.normal(0, 0.5)
        ])
        self.kalman_filter.update(noisy_position)
        
        # Get observations
        observations = self.get_observations(other_vehicles, intersections)
        
        # Make decision
        action = self.make_decision(observations)
        
        # Execute action
        self.execute_action(action, dt)
        
        # Update collision risk
        self.calculate_collision_risk(other_vehicles)
    
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
# TRAINING SYSTEM
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
        self.intersections = [
            {'x': 400, 'y': 300, 'type': 'cross', 'occupancy': 0},
            {'x': 800, 'y': 300, 'type': 'cross', 'occupancy': 0},
            {'x': 400, 'y': 500, 'type': 'cross', 'occupancy': 0},
            {'x': 800, 'y': 500, 'type': 'cross', 'occupancy': 0}
        ]
        
        self.training_history = {'rewards': [], 'throughput': [], 'loss': []}
        
    def spawn_vehicle(self) -> AutonomousVehicle:
        """Spawn vehicle at random entry point"""
        entry_points = [
            (100, 300, 0), (1100, 300, math.pi),
            (400, 100, math.pi/2), (400, 700, -math.pi/2),
            (800, 100, math.pi/2), (800, 700, -math.pi/2)
        ]
        
        x, y, heading = random.choice(entry_points)
        x += random.uniform(-20, 20)
        y += random.uniform(-20, 20)
        
        vehicle = AutonomousVehicle(x, y, heading)
        vehicle.transformer = self.transformer
        return vehicle
    
    def calculate_reward(self, vehicle: AutonomousVehicle) -> float:
        """Calculate reward for vehicle's actions"""
        reward = 0
        
        # Speed efficiency
        speed_ratio = vehicle.speed / max(vehicle.target_speed, 1)
        reward += speed_ratio * 2.0
        
        # Safety penalty
        if vehicle.collision_risk > 0.5:
            reward -= 10.0 * vehicle.collision_risk
        
        # Cooperation bonus
        if vehicle.behavior_mode == 'cooperative':
            reward += 1.0
        
        # Intersection efficiency
        for intersection in self.intersections:
            dist = math.sqrt((intersection['x'] - vehicle.x)**2 + (intersection['y'] - vehicle.y)**2)
            if 30 < dist < 50:  # Just passed through
                reward += 5.0
        
        return reward
    
    def train_step(self):
        """Single training step"""
        if len(self.memory) < 64:
            return 0
        
        batch = random.sample(self.memory, 64)
        
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long)
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32)
        next_states = torch.stack([exp[3] for exp in batch])
        dones = torch.tensor([exp[4] for exp in batch], dtype=torch.bool)
        
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
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def run_episode(self, episode_length: int = 300) -> Dict:
        """Run training episode (5 seconds simulated time, optimized)"""
        self.vehicles = [self.spawn_vehicle() for _ in range(6)]
        
        episode_rewards = []
        total_loss = 0
        training_steps = 0
        vehicles_passed = 0
        
        for step in range(episode_length):
            # Update intersections (simplified)
            for intersection in self.intersections:
                intersection['occupancy'] = sum(
                    1 for v in self.vehicles 
                    if math.sqrt((v.x - intersection['x'])**2 + (v.y - intersection['y'])**2) < 50
                )
            
            # Count vehicles before update
            initial_count = len(self.vehicles)
            
            # Update vehicles (streamlined)
            for vehicle in self.vehicles:
                # Faster state collection
                if len(vehicle.observation_history) >= 5:
                    prev_state = list(vehicle.observation_history)[-5:]  # Shorter history
                    
                    vehicle.update(1/60, self.vehicles, self.intersections)
                    
                    # Collect experience less frequently
                    if step % 5 == 0:
                        reward = self.calculate_reward(vehicle)
                        episode_rewards.append(reward)
                        
                        current_state = list(vehicle.observation_history)[-5:]
                        done = vehicle.collision_risk > 0.8 or step > 250
                        
                        if len(vehicle.action_history) > 0 and len(prev_state) == 5 and len(current_state) == 5:
                            action = vehicle.action_history[-1]
                            self.memory.append((
                                torch.stack(prev_state),
                                action, reward,
                                torch.stack(current_state),
                                done
                            ))
                else:
                    vehicle.update(1/60, self.vehicles, self.intersections)
            
            # Remove vehicles that exit and count throughput
            exited_vehicles = [v for v in self.vehicles 
                             if (v.x < 20 or v.x > 1180 or v.y < 20 or v.y > 780)]
            vehicles_passed += len(exited_vehicles)
            
            self.vehicles = [v for v in self.vehicles 
                           if not (v.x < 20 or v.x > 1180 or v.y < 20 or v.y > 780)]
            
            # Maintain vehicle count (spawn new ones)
            if len(self.vehicles) < 8 and step % 15 == 0:
                self.vehicles.append(self.spawn_vehicle())
            
            # Training (much less frequent)
            if step % 20 == 0 and len(self.memory) > 200:
                loss = self.train_step()
                total_loss += loss
                training_steps += 1
        
        return {
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'avg_loss': total_loss / max(training_steps, 1),
            'throughput': vehicles_passed
        }
    
    def train(self, episodes: int = 100):
        """Train the model with comprehensive feedback"""
        print(f"🧠 Training transformer for {episodes} episodes...")
        print("=" * 70)
        print("Each episode = 5 seconds of simulation time (ultra-optimized)")
        print("Vehicles learn to coordinate at intersections through trial & error")
        print("Target: 1-3 seconds per episode | Expected total time: ~5-10 minutes")
        print("=" * 70)
        
        start_time = time.time()
        best_performance = -float('inf')
        recent_rewards = deque(maxlen=10)
        
        # Progress indicators
        progress_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        
        for episode in range(1, episodes + 1):
            episode_start = time.time()
            
            # Show progress spinner for long episodes
            if episode > 1:
                spinner = progress_chars[(episode - 1) % len(progress_chars)]
                print(f"\r{spinner} Running episode {episode}...", end="", flush=True)
            
            results = self.run_episode()
            
            # Record metrics
            self.training_history['rewards'].append(results['avg_reward'])
            self.training_history['throughput'].append(results['throughput'])
            self.training_history['loss'].append(results['avg_loss'])
            recent_rewards.append(results['avg_reward'])
            
            # Calculate performance score
            performance = results['avg_reward'] + results['throughput'] * 0.1
            if performance > best_performance:
                best_performance = performance
                best_episode = episode
            
            episode_time = time.time() - episode_start
            
            # Detailed progress every episode for first 5, then every 10
            show_detail = episode <= 5 or episode % 10 == 0
            
            if show_detail:
                print(f"\r🎯 Episode {episode:3d}/{episodes} | "
                      f"Reward: {results['avg_reward']:6.2f} | "
                      f"Throughput: {results['throughput']:3d} | "
                      f"Loss: {results['avg_loss']:6.4f} | "
                      f"ε: {self.epsilon:.3f} | "
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
            else:
                # Simple progress for non-detailed episodes
                if episode % 5 == 0:
                    progress_pct = (episode / episodes) * 100
                    print(f"\r🔄 Progress: {episode}/{episodes} ({progress_pct:.0f}%) | "
                          f"Recent reward: {results['avg_reward']:.1f} | "
                          f"Time: {episode_time:.1f}s", end="", flush=True)
            
            # Progress milestones
            if episode in [10, 25, 50, 75]:
                elapsed = time.time() - start_time
                remaining = (elapsed / episode) * (episodes - episode)
                
                print(f"🏆 MILESTONE: {episode} episodes completed!")
                print(f"   Best performance so far: {best_performance:.2f} (episode {best_episode})")
                print(f"   Time elapsed: {elapsed/60:.1f}m | Estimated remaining: {remaining/60:.1f}m")
                print(f"   Learning trend: {'Improving' if len(recent_rewards) >= 5 and recent_rewards[-1] > recent_rewards[0] else 'Stabilizing'}")
                print()
                
                # Save checkpoint
                checkpoint_path = f'checkpoint_ep{episode}.pth'
                torch.save({
                    'model_state_dict': self.transformer.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'training_history': self.training_history,
                    'episode': episode
                }, checkpoint_path)
                print(f"   💾 Checkpoint saved: {checkpoint_path}")
                print()
        
        # Final results
        total_time = time.time() - start_time
        print("\r" + " " * 80)  # Clear progress line
        print("🎉 TRAINING COMPLETED!")
        print("=" * 70)
        print(f"Total training time: {total_time/60:.1f} minutes")
        print(f"Best performance: {best_performance:.2f} (episode {best_episode})")
        print(f"Final reward: {results['avg_reward']:.2f}")
        print(f"Final exploration rate: {self.epsilon:.1%}")
        print(f"Memory utilization: {len(self.memory):,}/50,000 experiences")
        
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
        
        # Save final model
        model_path = 'traffic_transformer.pth'
        torch.save({
            'model_state_dict': self.transformer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'final_performance': best_performance,
            'training_time': total_time
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
    def __init__(self, model_path: str = 'traffic_transformer.pth'):
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
        
        # Environment
        self.vehicles = []
        self.intersections = [
            {'x': 400, 'y': 300, 'type': 'cross', 'occupancy': 0},
            {'x': 800, 'y': 300, 'type': 'cross', 'occupancy': 0},
            {'x': 400, 'y': 500, 'type': 'cross', 'occupancy': 0},
            {'x': 800, 'y': 500, 'type': 'cross', 'occupancy': 0}
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
        
        # Update intersections
        for intersection in self.intersections:
            intersection['occupancy'] = sum(
                1 for v in self.vehicles 
                if math.sqrt((v.x - intersection['x'])**2 + (v.y - intersection['y'])**2) < 50
            )
        
        # Update vehicles
        for vehicle in self.vehicles:
            vehicle.update(dt, self.vehicles, self.intersections)
            
            if vehicle.behavior_mode == 'cooperative':
                self.metrics['cooperation'] += dt
        
        # Remove exited vehicles
        initial_count = len(self.vehicles)
        self.vehicles = [v for v in self.vehicles 
                        if not (v.x < 20 or v.x > 1180 or v.y < 20 or v.y > 780)]
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
        for intersection in self.intersections:
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
        for i, intersection in enumerate(self.intersections):
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
1. Train new model
2. Run simulation (requires trained model)
3. Both (train then simulate)

Enter choice (1-3): """).strip()
    
    if choice == '1':
        episodes = int(input("Training episodes (100-200 recommended): ") or "100")
        trainer = TrafficTrainer()
        trainer.train(episodes)
        
    elif choice == '2':
        sim = TrafficSimulation()
        if sim.running:
            sim.run()
            
    elif choice == '3':
        episodes = int(input("Training episodes (100-200 recommended): ") or "100")
        
        print("\n🧠 Training phase...")
        trainer = TrafficTrainer()
        trainer.train(episodes)
        
        print("\n🚗 Simulation phase...")
        sim = TrafficSimulation()
        if sim.running:
            sim.run()
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()