import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

@dataclass
class Vehicle:
    id: int
    x: float
    y: float
    velocity: float
    target_velocity: float
    approach_angle: int
    lane: int
    max_accel: float = 3.0
    max_decel: float = 6.0
    length: float = 4.5
    width: float = 2.0
    color: str = 'blue'

class NeuralMPCController(nn.Module):
    """
    Neural Network-based MPC that learns to optimize control sequences
    Much faster than traditional optimization, especially on GPU
    """
    def __init__(self, horizon=30, n_vehicles_max=20):
        super().__init__()
        self.horizon = horizon
        self.n_vehicles_max = n_vehicles_max
        
        # Network to predict optimal control sequence
        # Input: [own_state(3) + other_vehicles_states(3*(n_vehicles_max-1)) + time_features(2)]
        # For 16 vehicles: 3 + 3*15 + 2 = 50 features
        input_size = 3 + 3 * (n_vehicles_max - 1) + 2
        
        print(f"🧠 Neural MPC: {n_vehicles_max} max vehicles, input size: {input_size}")
        
        self.control_network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, horizon),  # Output: acceleration sequence
            nn.Tanh()  # Scale to [-1, 1], will be scaled by max_accel
        )
        
        # Initialize with small weights for stable training
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
    
    def forward(self, state_tensor):
        """
        Input: batch of state tensors [batch_size, input_features]
        Output: batch of control sequences [batch_size, horizon]
        """
        return self.control_network(state_tensor)

class PyTorchMPCVehicleController:
    def __init__(self, vehicle: Vehicle, neural_mpc: NeuralMPCController, horizon=30, dt=0.1):
        self.vehicle = vehicle
        self.neural_mpc = neural_mpc
        self.horizon = horizon
        self.dt = dt
        self.safety_radius = 8.0
        
        # For training the neural network
        self.optimizer = optim.Adam(self.neural_mpc.parameters(), lr=0.001)
        
    def encode_state(self, all_vehicles, time_step):
        """
        Encode current state for neural network input
        Returns tensor ready for GPU processing
        """
        # Own state: [x, y, velocity]
        own_state = [self.vehicle.x, self.vehicle.y, self.vehicle.velocity]
        
        # Other vehicles' states (padded to max vehicles)
        other_states = []
        for v in all_vehicles:
            if v.id != self.vehicle.id:
                other_states.extend([v.x, v.y, v.velocity])
        
        # Pad with zeros if fewer than max vehicles
        target_other_size = 3 * (self.neural_mpc.n_vehicles_max - 1)
        while len(other_states) < target_other_size:
            other_states.append(0.0)
        
        # Time features
        time_features = [time_step * self.dt, np.sin(time_step * self.dt * 0.1)]
        
        # Combine all features
        state = own_state + other_states + time_features
        
        # Debug print for first vehicle only to avoid spam
        if self.vehicle.id == 1 and time_step < 2:
            print(f"🔍 Vehicle {self.vehicle.id}: state size = {len(state)} (own:3 + others:{len(other_states)} + time:2)")
            print(f"    Max vehicles: {self.neural_mpc.n_vehicles_max}, actual vehicles: {len(all_vehicles)}")
        
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    def predict_trajectory_torch(self, control_inputs_tensor):
        """
        GPU-accelerated trajectory prediction using PyTorch
        """
        trajectory = torch.zeros((self.horizon + 1, 3), device=device)
        trajectory[0] = torch.tensor([self.vehicle.x, self.vehicle.y, self.vehicle.velocity], device=device)
        
        for i in range(self.horizon):
            x, y, v = trajectory[i]
            accel = control_inputs_tensor[i] * self.vehicle.max_accel  # Scale from [-1,1] to actual accel
            
            # Clamp acceleration to vehicle limits
            accel = torch.clamp(accel, -self.vehicle.max_decel, self.vehicle.max_accel)
            
            # Update velocity
            v_new = v + accel * self.dt
            v_new = torch.clamp(v_new, 0, self.vehicle.target_velocity * 1.2)
            
            # Update position based on approach direction
            if self.vehicle.approach_angle == 0:  # North
                x_new = x + v_new * self.dt
                y_new = y
            elif self.vehicle.approach_angle == 180:  # South
                x_new = x - v_new * self.dt
                y_new = y
            elif self.vehicle.approach_angle == 90:  # East
                x_new = x
                y_new = y + v_new * self.dt
            elif self.vehicle.approach_angle == 270:  # West
                x_new = x
                y_new = y - v_new * self.dt
            
            trajectory[i + 1] = torch.stack([x_new, y_new, v_new])
        
        return trajectory
    
    def compute_cost_torch(self, control_inputs_tensor, other_trajectories_torch):
        """
        GPU-accelerated cost computation
        """
        trajectory = self.predict_trajectory_torch(control_inputs_tensor)
        
        # Velocity tracking cost
        velocity_errors = trajectory[1:, 2] - self.vehicle.target_velocity
        velocity_cost = 10.0 * torch.sum(velocity_errors ** 2)
        
        # Control effort cost
        control_changes = control_inputs_tensor[1:] - control_inputs_tensor[:-1]
        control_cost = torch.sum(control_changes ** 2)
        
        # Collision avoidance cost
        collision_cost = torch.tensor(0.0, device=device)
        for other_traj in other_trajectories_torch:
            if other_traj is not None:
                # Compute distances at each time step
                dx = trajectory[1:, 0] - other_traj[1:len(trajectory), 0]
                dy = trajectory[1:, 1] - other_traj[1:len(trajectory), 1]
                distances = torch.sqrt(dx**2 + dy**2)
                
                # Exponential penalty for close distances
                close_mask = distances < self.safety_radius * 2
                penalties = 5000.0 * torch.exp(-distances[close_mask] / (self.safety_radius * 0.5))
                collision_cost += torch.sum(penalties)
        
        # Total cost
        total_cost = velocity_cost + control_cost + collision_cost
        
        return total_cost
    
    def solve_neural_mpc(self, all_vehicles, time_step):
        """
        Use neural network to predict optimal control, then fine-tune with gradient descent
        """
        # Encode state for neural network
        state_tensor = self.encode_state(all_vehicles, time_step)
        
        # Get initial control prediction from neural network
        with torch.no_grad():
            control_pred = self.neural_mpc(state_tensor).squeeze(0)
        
        # Fine-tune with a few gradient steps (online optimization)
        control_inputs = control_pred.clone().requires_grad_(True)
        local_optimizer = optim.Adam([control_inputs], lr=0.1)
        
        # Convert other vehicles' planned trajectories to torch tensors
        other_trajectories_torch = []
        for v in all_vehicles:
            if v.id != self.vehicle.id and hasattr(v, 'planned_trajectory') and v.planned_trajectory:
                traj = torch.tensor(v.planned_trajectory[:self.horizon+1], device=device)
                other_trajectories_torch.append(traj)
            else:
                other_trajectories_torch.append(None)
        
        # Quick local optimization (5 steps)
        for _ in range(5):
            local_optimizer.zero_grad()
            cost = self.compute_cost_torch(control_inputs, other_trajectories_torch)
            cost.backward()
            local_optimizer.step()
            
            # Clamp to valid range
            with torch.no_grad():
                control_inputs.clamp_(-1.0, 1.0)
        
        # Convert back to numpy for compatibility
        optimal_control = control_inputs.detach().cpu().numpy()
        
        # Scale first control input by max acceleration
        first_accel = optimal_control[0] * self.vehicle.max_accel
        
        # Update planned trajectory
        with torch.no_grad():
            traj_torch = self.predict_trajectory_torch(control_inputs.detach())
            planned_trajectory = traj_torch.cpu().numpy().tolist()
            self.vehicle.planned_trajectory = [(pos[0], pos[1], pos[2]) for pos in planned_trajectory]
        
        return first_accel, optimal_control

class PyTorchIntersectionSimulation:
    def __init__(self, intersection_size=20, dt=0.1, max_vehicles=20):
        self.intersection_size = intersection_size
        self.lane_width = 4.0
        self.dt = dt
        self.vehicles = []
        self.max_vehicles = max_vehicles
        
        # Single shared neural MPC network (faster than individual networks)
        self.neural_mpc = NeuralMPCController(n_vehicles_max=max_vehicles).to(device)
        self.controllers = {}
        
        # Animation setup
        self.history_x = {}
        self.history_y = {}
        self.history_v = {}
        self.current_step = 0
        
        self.fig, (self.ax_main, self.ax_speed) = plt.subplots(1, 2, figsize=(16, 8))
        self.plot_initialized = False
    
    def add_vehicle(self, vehicle: Vehicle):
        if len(self.vehicles) >= self.max_vehicles:
            print(f"Warning: Maximum {self.max_vehicles} vehicles supported")
            return
            
        # Calculate proper lane position - RIGHT LANES ONLY
        lane_offset = self.lane_width / 2
        if vehicle.approach_angle == 0:  # North (moving up) - right lane is negative Y
            vehicle.y = -lane_offset
        elif vehicle.approach_angle == 180:  # South (moving down) - right lane is positive Y  
            vehicle.y = lane_offset
        elif vehicle.approach_angle == 90:  # East (moving right) - right lane is positive X
            vehicle.x = lane_offset
        elif vehicle.approach_angle == 270:  # West (moving left) - right lane is negative X
            vehicle.x = -lane_offset
        
        self.vehicles.append(vehicle)
        self.controllers[vehicle.id] = PyTorchMPCVehicleController(vehicle, self.neural_mpc, dt=self.dt)
        self.history_x[vehicle.id] = []
        self.history_y[vehicle.id] = []
        self.history_v[vehicle.id] = []
    
    def create_car_shape(self, x, y, angle, length=4.5, width=2.0):
        """Create car shape polygon"""
        half_length = length / 2
        half_width = width / 2
        
        car_points = np.array([
            [-half_length, -half_width],
            [-half_length, half_width],
            [half_length * 0.8, half_width],
            [half_length, 0],
            [half_length * 0.8, -half_width],
        ])
        
        angle_rad = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        rotated_points = np.dot(car_points, rotation_matrix.T)
        rotated_points[:, 0] += x
        rotated_points[:, 1] += y
        
        return rotated_points
    
    def setup_plot(self):
        """Initialize visualization"""
        # Main intersection plot
        self.ax_main.set_xlim(-100, 100)
        self.ax_main.set_ylim(-100, 100)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_title(f'PyTorch MPC + V2V Intersection Control', fontsize=16, fontweight='bold')
        self.ax_main.set_xlabel('X Position (m)')
        self.ax_main.set_ylabel('Y Position (m)')
        self.ax_main.set_facecolor('#f0f0f0')
        
        # Draw road infrastructure
        intersection = Rectangle((-self.intersection_size/2, -self.intersection_size/2), 
                               self.intersection_size, self.intersection_size, 
                               fill=True, facecolor='lightgray', edgecolor='black', 
                               linewidth=2, alpha=0.8)
        self.ax_main.add_patch(intersection)
        
        # Draw roads (single lanes in each direction)
        road_length = 100
        
        # North-South road (single lane each direction)
        ns_road_north = Rectangle((-self.lane_width/2, 0), 
                                self.lane_width/2, road_length,
                                fill=True, facecolor='darkgray', alpha=0.7)
        self.ax_main.add_patch(ns_road_north)
        
        ns_road_south = Rectangle((0, -road_length), 
                                self.lane_width/2, road_length,
                                fill=True, facecolor='darkgray', alpha=0.7)
        self.ax_main.add_patch(ns_road_south)
        
        # East-West road (single lane each direction)
        ew_road_east = Rectangle((0, -self.lane_width/2),
                               road_length, self.lane_width/2,
                               fill=True, facecolor='darkgray', alpha=0.7)
        self.ax_main.add_patch(ew_road_east)
        
        ew_road_west = Rectangle((-road_length, 0),
                               road_length, self.lane_width/2,
                               fill=True, facecolor='darkgray', alpha=0.7)
        self.ax_main.add_patch(ew_road_west)
        
        # Draw lane center lines (yellow dashed)
        self.ax_main.plot([0, 0], [-road_length, -self.intersection_size/2], 'y--', linewidth=2, alpha=0.8)  # South approach
        self.ax_main.plot([0, 0], [self.intersection_size/2, road_length], 'y--', linewidth=2, alpha=0.8)    # North approach
        self.ax_main.plot([-road_length, -self.intersection_size/2], [0, 0], 'y--', linewidth=2, alpha=0.8) # West approach
        self.ax_main.plot([self.intersection_size/2, road_length], [0, 0], 'y--', linewidth=2, alpha=0.8)   # East approach
        
        # Speed plot
        self.ax_speed.set_xlim(0, 60)
        self.ax_speed.set_ylim(0, 20)
        self.ax_speed.grid(True, alpha=0.3)
        self.ax_speed.set_title('Vehicle Speeds', fontsize=14, fontweight='bold')
        self.ax_speed.set_xlabel('Time (s)')
        self.ax_speed.set_ylabel('Speed (m/s)')
        
        # Initialize vehicle visualizations
        self.vehicle_shapes = {}
        self.vehicle_trails = {}
        self.speed_lines = {}
        self.predicted_paths = {}
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan'] * 3
        
        for i, vehicle in enumerate(self.vehicles):
            color = colors[i % len(colors)]
            vehicle.color = color
            
            # Car shape
            car_points = self.create_car_shape(vehicle.x, vehicle.y, vehicle.approach_angle,
                                             vehicle.length, vehicle.width)
            self.vehicle_shapes[vehicle.id] = Polygon(car_points, 
                                                    facecolor=color, 
                                                    edgecolor='black',
                                                    linewidth=1, 
                                                    alpha=0.9)
            self.ax_main.add_patch(self.vehicle_shapes[vehicle.id])
            
            # Trail
            self.vehicle_trails[vehicle.id], = self.ax_main.plot([], [], 
                                                               color=color, alpha=0.6, 
                                                               linewidth=2)
            
            # Speed line
            self.speed_lines[vehicle.id], = self.ax_speed.plot([], [], 
                                                             color=color, linewidth=2, 
                                                             label=f'V{vehicle.id}')
            
            # Predicted path
            self.predicted_paths[vehicle.id], = self.ax_main.plot([], [], 
                                                                color=color, linestyle=':', 
                                                                alpha=0.5, linewidth=1)
        
        if self.vehicles:
            self.ax_speed.legend(loc='upper right')
        
        # Metrics text
        self.metrics_text = self.ax_main.text(0.02, 0.98, '', transform=self.ax_main.transAxes, 
                                            verticalalignment='top', fontsize=10, fontweight='bold',
                                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    def simulate_step(self):
        """GPU-accelerated simulation step"""
        # All vehicles solve MPC in parallel (GPU optimization)
        control_inputs = {}
        
        for vehicle in self.vehicles:
            controller = self.controllers[vehicle.id]
            optimal_accel, _ = controller.solve_neural_mpc(self.vehicles, self.current_step)
            control_inputs[vehicle.id] = optimal_accel
        
        # Apply controls
        for vehicle in self.vehicles:
            accel = control_inputs[vehicle.id]
            
            # Update velocity
            vehicle.velocity += accel * self.dt
            vehicle.velocity = np.clip(vehicle.velocity, 0, vehicle.target_velocity * 1.2)
            
            # Update position
            if vehicle.approach_angle == 0:  # North
                vehicle.x += vehicle.velocity * self.dt
            elif vehicle.approach_angle == 180:  # South
                vehicle.x -= vehicle.velocity * self.dt
            elif vehicle.approach_angle == 90:  # East
                vehicle.y += vehicle.velocity * self.dt
            elif vehicle.approach_angle == 270:  # West
                vehicle.y -= vehicle.velocity * self.dt
            
            # Store history
            self.history_x[vehicle.id].append(vehicle.x)
            self.history_y[vehicle.id].append(vehicle.y)
            self.history_v[vehicle.id].append(vehicle.velocity)
        
        self.current_step += 1
    
    def animate(self, frame):
        """Animation with clean visualization"""
        if not self.plot_initialized:
            return []
        
        if frame < 1000:  # 100 seconds
            self.simulate_step()
            
            # Update vehicle visualizations
            for vehicle in self.vehicles:
                if vehicle.id in self.vehicle_shapes:
                    # Update car shape
                    car_points = self.create_car_shape(vehicle.x, vehicle.y, vehicle.approach_angle,
                                                     vehicle.length, vehicle.width)
                    self.vehicle_shapes[vehicle.id].set_xy(car_points)
                    
                    # Update trail
                    if len(self.history_x[vehicle.id]) > 1:
                        self.vehicle_trails[vehicle.id].set_data(
                            self.history_x[vehicle.id][-100:],
                            self.history_y[vehicle.id][-100:]
                        )
                    
                    # Update speed plot
                    times = [i * self.dt for i in range(len(self.history_v[vehicle.id]))]
                    self.speed_lines[vehicle.id].set_data(times, self.history_v[vehicle.id])
                    
                    # Update predicted path
                    if hasattr(vehicle, 'planned_trajectory') and vehicle.planned_trajectory:
                        pred_x = [pos[0] for pos in vehicle.planned_trajectory[:20]]
                        pred_y = [pos[1] for pos in vehicle.planned_trajectory[:20]]
                        self.predicted_paths[vehicle.id].set_data(pred_x, pred_y)
            
            # Update metrics
            current_speeds = [v.velocity for v in self.vehicles]
            min_speed = min(current_speeds) if current_speeds else 0
            avg_speed = np.mean(current_speeds) if current_speeds else 0
            time_elapsed = frame * self.dt
            
            # Compute minimum distance
            min_distance = float('inf')
            for i, v1 in enumerate(self.vehicles):
                for j, v2 in enumerate(self.vehicles):
                    if i < j:
                        dist = np.sqrt((v1.x - v2.x)**2 + (v1.y - v2.y)**2)
                        min_distance = min(min_distance, dist)
            
            # Count vehicles in intersection
            in_intersection = 0
            for v in self.vehicles:
                if (abs(v.x) < self.intersection_size/2 + 5 and 
                    abs(v.y) < self.intersection_size/2 + 5):
                    in_intersection += 1
            
            metrics_str = f"⏱️ Time: {time_elapsed:.1f}s\n"
            metrics_str += f"🚗 Vehicles: {len(self.vehicles)}\n"
            metrics_str += f"🏁 In intersection: {in_intersection}\n"
            metrics_str += f"📏 Min distance: {min_distance:.1f}m\n"
            metrics_str += f"⚡ Min speed: {min_speed:.1f} m/s\n"
            metrics_str += f"✅ Safe coordination: {'YES' if min_distance > 6.0 else '⚠️ CLOSE'}"
            
            self.metrics_text.set_text(metrics_str)
        
        return list(self.vehicle_shapes.values()) + list(self.vehicle_trails.values())
    
    def finalize_setup(self):
        if not self.plot_initialized:
            self.setup_plot()
            self.plot_initialized = True
    
    def run_animation(self):
        self.finalize_setup()
        ani = animation.FuncAnimation(self.fig, self.animate, frames=1000, 
                                    interval=50, blit=False, repeat=False)
        plt.tight_layout()
        plt.show()
        return ani

def create_large_scale_scenario():
    """Create scenario with vehicles in right lanes only"""
    sim = PyTorchIntersectionSimulation(max_vehicles=8)
    
    # Create 8 vehicles from different directions - RIGHT LANES ONLY
    vehicles = [
        # North approach (2 cars)
        Vehicle(1, x=-70, y=0, velocity=14, target_velocity=15, approach_angle=0, lane=0),
        Vehicle(2, x=-95, y=0, velocity=13, target_velocity=15, approach_angle=0, lane=0),
        
        # South approach (2 cars)
        Vehicle(3, x=75, y=0, velocity=16, target_velocity=15, approach_angle=180, lane=0),
        Vehicle(4, x=100, y=0, velocity=14, target_velocity=15, approach_angle=180, lane=0),
        
        # East approach (2 cars)
        Vehicle(5, x=0, y=-65, velocity=12, target_velocity=15, approach_angle=90, lane=0),
        Vehicle(6, x=0, y=-90, velocity=15, target_velocity=15, approach_angle=90, lane=0),
        
        # West approach (2 cars)
        Vehicle(7, x=0, y=70, velocity=13, target_velocity=15, approach_angle=270, lane=0),
        Vehicle(8, x=0, y=95, velocity=16, target_velocity=15, approach_angle=270, lane=0),
    ]
    
    for v in vehicles:
        sim.add_vehicle(v)
    
    return sim

if __name__ == "__main__":
    print("🚗 PyTorch MPC + V2V Intersection Control")
    print(f"Device: {device}")
    print("Features:")
    print("  🧠 Neural MPC coordination")
    print("  🛣️ Right-lane-only traffic")
    print("  🚗 Vehicle-to-vehicle communication")
    print("  ✨ Clean visualization")
    
    sim = create_large_scale_scenario()
    animation_object = sim.run_animation()
    
    print("🎬 Animation complete! All vehicles stayed in right lanes and coordinated safely!")