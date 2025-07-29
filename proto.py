"""
Modern Autonomous Vehicle Traffic Simulation 
Testing hypothesis: Can autonomous vehicles eliminate traffic lights through coordination?
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Reproducibility seeds
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

@dataclass
class VehicleState:
    """Optimized vehicle state representation with priority system"""
    id: int
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [x, y]
    velocity: float = 0.0
    target_velocity: float = 20.0  # Increased from 15.0
    approach_angle: int = 0  # 0=North, 90=East, 180=South, 270=West
    lane: int = 0
    priority: float = 0.0  # Higher priority vehicles have right of way
    
    # Physical properties
    length: float = 4.5
    width: float = 2.0
    max_accel: float = 3.0
    max_decel: float = 6.0
    
    # Visualization
    color: str = '#00aaff'
    
    # Trajectory storage
    planned_trajectory: Optional[np.ndarray] = None  # [horizon+1, 3] array
    
    @property
    def x(self) -> float:
        return self.position[0]
    
    @property
    def y(self) -> float:
        return self.position[1]
    
    def update_position(self, dt: float):
        """Update position based on velocity and approach angle"""
        if self.approach_angle == 0:  # North
            self.position[0] += self.velocity * dt
        elif self.approach_angle == 180:  # South
            self.position[0] -= self.velocity * dt
        elif self.approach_angle == 90:  # East
            self.position[1] += self.velocity * dt
        elif self.approach_angle == 270:  # West
            self.position[1] -= self.velocity * dt


class NeuralMPCController(nn.Module):
    """Predictive neural MPC controller with collision avoidance"""
    def __init__(self, horizon: int = 25, max_vehicles: int = 20):  # Increased horizon for better prediction
        super().__init__()
        self.horizon = horizon
        self.max_vehicles = max_vehicles
        
        input_size = 3 + 3 * (max_vehicles - 1) + 2  # own + others + time
        
        self.control_network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, horizon),
            nn.Tanh()
        )
        
        # Initialize with small weights using @torch.no_grad() for efficiency
        self._initialize_weights()
    
    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.control_network(state_tensor)
    
    @torch.no_grad()
    def _initialize_weights(self):
        """Initialize weights efficiently without tracking gradients"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class BatchedMPCController:
    """High-performance batched MPC controller"""
    def __init__(self, neural_mpc: NeuralMPCController, dt: float = 0.1):
        self.neural_mpc = neural_mpc
        self.dt = dt
        self.horizon = neural_mpc.horizon
        self.max_vehicles = neural_mpc.max_vehicles
        
        # Pre-allocated GPU tensors with optimal memory management
        self.state_batch = torch.zeros((self.max_vehicles, neural_mpc.control_network[0].in_features), 
                                      device=device, dtype=torch.float32)
        self.trajectory_batch = torch.zeros((self.max_vehicles, self.horizon + 1, 3), 
                                           device=device, dtype=torch.float32)
        self.control_batch = torch.zeros((self.max_vehicles, self.horizon), 
                                        device=device, dtype=torch.float32)
        
        # Initialize CUDA streams for concurrent execution
        if device.type == 'cuda':
            self.compute_stream = torch.cuda.Stream()
            self.transfer_stream = torch.cuda.Stream()
        else:
            self.compute_stream = None
            self.transfer_stream = None
    
    def encode_states_batch(self, vehicles: List[VehicleState], time_step: int) -> torch.Tensor:
        """Vectorized state encoding with single CPU→GPU transfer"""
        n_vehicles = len(vehicles)
        batch_size = min(n_vehicles, self.max_vehicles)
        feat = self.neural_mpc.control_network[0].in_features  # input_size
        
        # Build on CPU
        state_cpu = np.zeros((batch_size, feat), dtype=np.float32)

        t = time_step * self.dt
        time_feats = np.array([t, np.sin(t * 0.1)], dtype=np.float32)

        # Pre-extract (x,y,v) for all vehicles to avoid attribute lookups in inner loop
        xyv = np.array([[v.x, v.y, v.velocity] for v in vehicles], dtype=np.float32)

        max_others = self.max_vehicles - 1
        for i in range(batch_size):
            # own state
            state_cpu[i, :3] = xyv[i]
            # others (nearest-first for more informative context, same input size)
            fill, idx = 0, 3
            deltas = xyv[:, :2] - xyv[i, :2]              # [N,2]
            dists = np.linalg.norm(deltas, axis=1)        # [N]
            order = np.argsort(dists)                     # nearest → farthest
            for j in order:
                if j == i or fill >= max_others:
                    continue
                state_cpu[i, idx:idx+3] = xyv[j]
                idx += 3
                fill += 1
            # time features
            start = 3 + 3 * (self.max_vehicles - 1)
            state_cpu[i, start:start+2] = time_feats

        # Single nonblocking copy into prealloc GPU buffer
        if device.type == 'cuda':
            state_gpu = torch.from_numpy(state_cpu).pin_memory().to(device, non_blocking=True)
        else:
            state_gpu = torch.from_numpy(state_cpu).to(device)
        # Put into our pre-allocated batch to keep shapes stable for TorchScript etc.
        with torch.no_grad():
            self.state_batch[:batch_size].copy_(state_gpu)
        return self.state_batch[:batch_size]
    
    def predict_trajectories_batch(self, control_batch: torch.Tensor, vehicles: List[VehicleState]) -> torch.Tensor:
        """Vectorized trajectory prediction"""
        batch_size = control_batch.shape[0]
        self.trajectory_batch[:batch_size].zero_()
        
        # Initialize with current positions (vectorized copy)
        xyv0 = np.array([[v.x, v.y, v.velocity] for v in vehicles[:batch_size]], dtype=np.float32)
        xyv0 = torch.from_numpy(xyv0).to(device=device)
        self.trajectory_batch[:batch_size, 0].copy_(xyv0)
        
        # Precompute direction vectors and target velocities once
        dir_map = {0: (1.0, 0.0), 180: (-1.0, 0.0), 90: (0.0, 1.0), 270: (0.0, -1.0)}
        dirs = torch.tensor([dir_map[v.approach_angle] for v in vehicles[:batch_size]],
                           device=device, dtype=torch.float32)  # [B,2]
        target_vels = torch.tensor([v.target_velocity * 1.2 for v in vehicles[:batch_size]],
                                  device=device, dtype=torch.float32)  # [B]
        
        # Predict trajectories
        for t in range(self.horizon):
            current_states = self.trajectory_batch[:batch_size, t]
            accels = torch.clamp(control_batch[:, t] * 3.0, -6.0, 3.0)
            
            # Update velocities
            new_velocities = current_states[:, 2] + accels * self.dt
            new_velocities = torch.clamp(new_velocities, min=0.0)  # lower bound 0
            new_velocities = torch.minimum(new_velocities, target_vels)
            
            # Positions advance along each vehicle's direction (vectorized)
            new_positions = current_states[:, :2] + dirs * new_velocities.unsqueeze(1) * self.dt
            
            self.trajectory_batch[:batch_size, t + 1, :2] = new_positions
            self.trajectory_batch[:batch_size, t + 1, 2] = new_velocities
        
        return self.trajectory_batch[:batch_size]
    
    def compute_costs_batch(self, control_batch: torch.Tensor, trajectory_batch: torch.Tensor, vehicles: List[VehicleState], reserved_ids: Optional[set] = None, center_half: float = 10.0, soon_steps: int = 10) -> torch.Tensor:
        """Vectorized cost computation"""
        batch_size = control_batch.shape[0]
        
        # soft reservation penalty
        if reserved_ids is None:
            reserved_mask = torch.zeros(batch_size, device=device)
        else:
            reserved_mask = torch.tensor([1.0 if v.id in reserved_ids else 0.0 for v in vehicles[:batch_size]],
                                         device=device)
        
        # Enhanced velocity tracking with adaptive targets
        velocity_costs = torch.zeros(batch_size, device=device)
        
        # Vectorized look-ahead proximity computation (avoid in-place ops for autograd)
        K = min(6, self.horizon + 1)
        # pos: [K,B,2] - transpose for easier indexing
        pos = trajectory_batch[:batch_size, :K, :2].permute(1, 0, 2)
        # pairwise diffs across K: [K,B,B,2]
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        # distances: [K,B,B]
        dist = torch.linalg.norm(diff, dim=-1)
        # mask self-pairs without in-place edits
        eye = torch.eye(batch_size, device=device, dtype=torch.bool)
        mask = eye.unsqueeze(0).expand(K, -1, -1)                    # [K,B,B]
        dist = dist.masked_fill(mask, float('inf'))
        # per-vehicle min distance over others and over K steps: [B]
        min_distance_to_others = dist.min(dim=2).values.min(dim=0).values
        
        for i in range(batch_size):
            vehicle = vehicles[i]
            target_vel = vehicle.target_velocity
            actual_vels = trajectory_batch[i, 1:, 2]  # Future velocities

            # Adaptive target velocity & weight (no GPU→CPU sync)
            adaptive_target = torch.as_tensor(target_vel, device=device, dtype=torch.float32)
            near_mask = (min_distance_to_others[i] < 20.0)
            adaptive_target = torch.where(
                near_mask,
                adaptive_target * (0.3 + 0.7 * (min_distance_to_others[i] / 20.0)),
                adaptive_target
            )
            velocity_weight = torch.where(
                near_mask,
                torch.tensor(5.0, device=device),
                torch.tensor(15.0, device=device)
            )
            
            # Smooth velocity profile - penalize rapid changes
            vel_errors = actual_vels - adaptive_target
            velocity_costs[i] = velocity_weight * torch.sum(vel_errors ** 2)
            
            # Penalty for velocity changes (jerk minimization)
            if len(actual_vels) > 1:
                vel_changes = actual_vels[1:] - actual_vels[:-1]
                velocity_costs[i] += 2.0 * torch.sum(vel_changes ** 2)
        
        # Enhanced control effort cost with smoothness
        control_costs = torch.zeros(batch_size, device=device)
        
        # Penalize large control inputs
        control_magnitude = torch.sum(control_batch ** 2, dim=1)
        control_costs += 0.5 * control_magnitude
        
        # Penalize rapid control changes (smoothness)
        if self.horizon > 1:
            control_changes = control_batch[:, 1:] - control_batch[:, :-1]
            control_costs += 2.0 * torch.sum(control_changes ** 2, dim=1)
            
            # Extra penalty for extreme control changes
            extreme_changes = torch.abs(control_changes) > 0.5
            if extreme_changes.any():
                control_costs += 10.0 * torch.sum(extreme_changes.float(), dim=1)
        
        # Encourage only the reserved vehicle to enter the center soon
        # Heuristic: if predicted position crosses the center bounds within ~1s, penalize unreserved
        K_check = min(soon_steps, self.horizon + 1)
        pos_check = trajectory_batch[:batch_size, :K_check, :2]          # [B,K,2]
        soon = ((pos_check[..., 0].abs() < center_half + 2.0) &
                (pos_check[..., 1].abs() < center_half + 2.0)).any(dim=1).float()  # [B]
        # penalty only if soon and not reserved
        not_reserved = (1.0 - reserved_mask)
        reservation_costs = 200.0 * soon * not_reserved
        
        # Advanced predictive collision avoidance cost using CPA
        collision_costs = torch.zeros(batch_size, device=device)

        if batch_size > 1:
            # states at first predicted step
            pos1 = trajectory_batch[:batch_size, 1, :2]    # [B,2]
            vel1 = trajectory_batch[:batch_size, 1, 2]     # [B]

            # Precompute direction and extent tensors
            dir_map = {0: (1.0, 0.0), 180: (-1.0, 0.0), 90: (0.0, 1.0), 270: (0.0, -1.0)}
            dirs = torch.tensor([dir_map[v.approach_angle] for v in vehicles[:batch_size]],
                                device=device, dtype=torch.float32)                        # [B,2]

            half_exts = torch.tensor(
                [[(veh.length*0.5 if veh.approach_angle in (0,180) else veh.width*0.5),
                  (veh.width*0.5  if veh.approach_angle in (0,180) else veh.length*0.5)]
                 for veh in vehicles[:batch_size]],
                device=device, dtype=torch.float32)                                        # [B,2]

            # Pairwise relative position/velocity (broadcasted)
            r0 = pos1[:, None, :] - pos1[None, :, :]                                       # [B,B,2]
            v0 = vel1[:, None, None] * dirs[:, None, :] - vel1[None, :, None] * dirs[None, :, :]  # [B,B,2]

            vv = (v0 * v0).sum(-1) + 1e-6                                                  # [B,B]
            H_dt = self.horizon * self.dt
            t_star = torch.clamp(-(r0 * v0).sum(-1) / vv, min=0.0, max=H_dt)               # [B,B]

            r_star = r0 + v0 * t_star[..., None]                                           # [B,B,2]
            combined_half = half_exts[:, None, :] + half_exts[None, :, :]                  # [B,B,2]
            sep_x = torch.abs(r_star[..., 0]) - combined_half[..., 0]
            sep_y = torch.abs(r_star[..., 1]) - combined_half[..., 1]
            d_min = torch.maximum(sep_x, sep_y)                                            # [B,B]

            margin, tau = 1.0, 2.0
            gap = margin - d_min
            dist_cost = 10000.0 * torch.relu(gap) ** 2                                     # [B,B]
            time_weight = torch.exp(-t_star / tau)                                         # [B,B]
            base_pair_cost = dist_cost * time_weight                                       # [B,B]

            # Zero self-pairs and (optionally) keep only upper triangle to avoid double work
            eye = torch.eye(batch_size, device=device, dtype=torch.bool)
            base_pair_cost = base_pair_cost.masked_fill(eye, 0.0)

            # Priority-based weights (for "cost applied to i" and to j)
            pr = torch.tensor([veh.priority for veh in vehicles[:batch_size]],
                              device=device, dtype=torch.float32)                           # [B]
            wi = torch.where(pr[:, None] > pr[None, :], 0.3,
                 torch.where(pr[:, None] < pr[None, :], 1.0, 0.7))                          # [B,B]

            # Row-sum gives per-vehicle contribution (same behavior as your loop)
            collision_costs = (base_pair_cost * wi).sum(dim=1)                              # [B]
        
        return velocity_costs + control_costs + reservation_costs + collision_costs
    
    def solve_batch_mpc(self, vehicles: List[VehicleState], time_step: int, reserved_ids: Optional[set] = None) -> List[float]:
        """High-performance MPC solver with CUDA streams"""
        if not vehicles:
            return []
        
        batch_size = len(vehicles)
        
        # Debug assertions to catch shape mismatches
        assert batch_size <= self.max_vehicles, f"Batch size {batch_size} exceeds max {self.max_vehicles}"
        
        # Use compute stream for main computation
        if self.compute_stream is not None:
            with torch.cuda.stream(self.compute_stream):
                state_batch = self.encode_states_batch(vehicles, time_step)
                
                # Neural network prediction - inference only, no gradients needed
                with torch.no_grad():
                    control_pred_batch = self.neural_mpc(state_batch)
        else:
            state_batch = self.encode_states_batch(vehicles, time_step)
            # Neural network prediction - inference only, no gradients needed
            with torch.no_grad():
                control_pred_batch = self.neural_mpc(state_batch)
        
        # Neural network prediction handled above in stream context
        
        # Predictive gradient descent for collision avoidance
        control_batch = control_pred_batch.clone().detach()
        learning_rate = 0.15  # Balanced learning rate for stability
        
        # Clear any existing gradients for optimal performance  
        for param in self.neural_mpc.parameters():
            param.grad = None
        
        # More iterations for better collision avoidance planning
        max_iterations = 15  # Increased for better collision prediction
        best_cost = float('inf')
        best_controls = control_batch.clone()
        
        # Shape assertions before optimization loop
        assert control_batch.shape == (batch_size, self.horizon), f"Control shape {control_batch.shape} != ({batch_size}, {self.horizon})"
        
        for iteration in range(max_iterations):
            # Create fresh control tensor for this iteration
            control_var = control_batch.clone().requires_grad_(True)
            
            # Forward pass
            trajectory_batch = self.predict_trajectories_batch(control_var, vehicles)
            
            # Shape assertions during optimization
            assert trajectory_batch.shape[:2] == (batch_size, self.horizon + 1), f"Trajectory shape {trajectory_batch.shape[:2]} != ({batch_size}, {self.horizon + 1})"
            
            costs = self.compute_costs_batch(control_var, trajectory_batch, vehicles, reserved_ids,
                                            center_half=getattr(self, 'center_half', 10.0),
                                            soon_steps=getattr(self, 'soon_steps', min(10, self.horizon + 1)))
            
            # Ensure costs has correct shape
            assert costs.shape == (batch_size,), f"Costs shape {costs.shape} != ({batch_size},)"
            total_cost = torch.sum(costs)
            
            # Compute gradients efficiently
            grad = torch.autograd.grad(total_cost, control_var, 
                                     create_graph=False, retain_graph=False)[0]
            
            # Manual gradient descent step with improved efficiency
            with torch.no_grad():
                # More aggressive gradient clipping for faster convergence
                grad_norm = torch.norm(grad)
                if grad_norm > 2.0:  # Increased threshold
                    grad = grad * (2.0 / grad_norm)
                
                # Update controls with momentum-like step
                control_batch = control_batch - learning_rate * grad
                
                # Clamp controls
                control_batch = torch.clamp(control_batch, -1.0, 1.0)
                
                # Track best solution for collision avoidance
                if iteration > 0:
                    current_cost = total_cost.item()
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_controls = control_batch.clone()
                
                # Early stopping if converged (more conservative)
                if iteration > 5 and grad_norm < 0.005:
                    break
        
        # Use best solution found
        control_batch = best_controls
        
        # Update planned trajectories (final forward pass) - no gradients needed
        with torch.no_grad():
            final_trajectories = self.predict_trajectories_batch(control_batch, vehicles)
            # Keep trajectories on GPU, only transfer when needed for visualization
            for i, vehicle in enumerate(vehicles):
                if hasattr(vehicle, '_gpu_trajectory'):
                    vehicle._gpu_trajectory = final_trajectories[i]
                else:
                    vehicle.planned_trajectory = final_trajectories[i].cpu().numpy()
        
        # Synchronize streams before final output
        if self.compute_stream is not None:
            torch.cuda.synchronize()
        
        # Return first control inputs - avoid CPU transfer bottleneck
        with torch.no_grad():
            # Extract first control step directly on GPU, then single CPU transfer
            first_controls = control_batch[:batch_size, 0] * 3.0
            return first_controls.cpu().tolist()


class IntersectionSimulation:
    """Main simulation class with optimal data structures"""
    def __init__(self, intersection_size: int = 20, dt: float = 0.1, max_vehicles: int = 20):
        self.intersection_size = intersection_size
        self.lane_width = 8.0
        self.dt = dt
        self.max_vehicles = max_vehicles
        
        # Vehicle storage
        self.vehicles: List[VehicleState] = []
        
        # Scene parameters for consistent calculations (define before controller)
        self.center_half = intersection_size / 2
        self.soon_steps = max(1, int(round(1.0 / dt)))  # 1s look-ahead
        
        # High-performance controller with optimized initialization
        neural_mpc = NeuralMPCController(max_vehicles=max_vehicles)
        neural_mpc = neural_mpc.to(device=device, dtype=torch.float32)
        neural_mpc.eval()  # inference-only
        self.controller = BatchedMPCController(neural_mpc, dt=dt)
        
        # Propagate scene parameters to controller
        self.controller.center_half = self.center_half
        self.controller.soon_steps = self.soon_steps
        
        # Simulation state
        self.current_step = 0
        self.collision_detected = False
        self.collision_pairs: List[Tuple[int, int]] = []
        self.collision_alert_time = 0
        self._warn_cooldown = 0.0  # Print throttling
        
        # Metrics for validation
        self.metrics = dict(unreserved_attempts=0, completed=0)
        self._inside_box_prev = {}
        self._has_completed = {}
        
        # Intersection scheduling
        self.conflict_zones = {
            'center': {'bounds': (-intersection_size/2, intersection_size/2, -intersection_size/2, intersection_size/2),
                      'reservations': []},  # [(vehicle_id, start_time, end_time)]
        }
        self.headway_time = 2.0  # Minimum time between vehicles in conflict zone
        
        # History storage (numpy arrays for efficiency)
        self.history_x = {}
        self.history_y = {}
        self.history_v = {}
        
        # Optimized visualization with performance settings
        plt.ioff()  # Turn off interactive mode for stable animation
        self.fig, (self.ax_main, self.ax_speed) = plt.subplots(1, 2, figsize=(16, 8))
        if hasattr(self.fig.canvas, "toolbar_visible"):
            self.fig.canvas.toolbar_visible = False  # Disable toolbar for speed
        
        self.plot_initialized = False
        self.vehicle_shapes = {}
        self.vehicle_trails = {}
        self.speed_lines = {}
        self.predicted_paths = {}
        
        # Performance optimization flags
        self._skip_frames = 0
        self._frame_skip_count = 2  # Skip every N frames for smoother animation
    
    def add_vehicle(self, vehicle: VehicleState):
        """Add vehicle with proper lane positioning"""
        if len(self.vehicles) >= self.max_vehicles:
            print(f"Warning: Maximum {self.max_vehicles} vehicles supported")
            return
        
        # Position vehicle in center of lane
        lane_offset = self.lane_width / 4
        if vehicle.approach_angle == 0:  # North
            vehicle.position[1] = -lane_offset
        elif vehicle.approach_angle == 180:  # South
            vehicle.position[1] = lane_offset
        elif vehicle.approach_angle == 90:  # East
            vehicle.position[0] = lane_offset
        elif vehicle.approach_angle == 270:  # West
            vehicle.position[0] = -lane_offset
        
        self.vehicles.append(vehicle)
        self.history_x[vehicle.id] = []
        self.history_y[vehicle.id] = []
        self.history_v[vehicle.id] = []
        self._inside_box_prev[vehicle.id] = False
        self._has_completed[vehicle.id] = False
    
    def _inside_box(self, x, y, margin=5.0):
        """Check if position is inside intersection box with margin"""
        return (abs(x) < self.center_half + margin) and (abs(y) < self.center_half + margin)

    def _add_candidates_Linf(self, r, v, ext, T, cands):
        """Add all critical times for L∞ separation function"""
        # Axis zero-crossings
        for ax in (0, 1):
            if abs(v[ax]) > 1e-9:
                t0 = -r[ax] / v[ax]
                if 0.0 <= t0 <= T:
                    cands.append(t0)

        # Boundary-touch times |s_ax| = ext_ax
        for ax in (0, 1):
            if abs(v[ax]) > 1e-9:
                for sgn in (-1.0, 1.0):
                    t_hit = (sgn * ext[ax] - r[ax]) / v[ax]
                    if 0.0 <= t_hit <= T:
                        cands.append(t_hit)

        # Equality of axis penalties: |sx|-ex = |sy|-ey
        # Enumerate sign combos to avoid abs(); solve linear t
        rx, ry = r
        vx, vy = v
        ex, ey = ext
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                denom = (sx * vx - sy * vy)
                if abs(denom) > 1e-9:
                    t_eq = (sy * ry - sx * rx + (ex - ey)) / denom
                    if 0.0 <= t_eq <= T:
                        cands.append(t_eq)

    def cpa_rect(self, p_i, v_i, p_j, v_j, half_extents, H, dt):
        """Time of closest approach (TCA/CPA) + Minkowski rectangles with complete candidate set"""
        r = p_i - p_j
        v = v_i - v_j
        T = H * dt
        vv = np.dot(v, v)
        t_star = 0.0 if vv < 1e-9 else np.clip(-np.dot(r, v) / vv, 0.0, T)

        def sep_Linf(t):
            s = r + v * t
            return max(abs(s[0]) - half_extents[0], abs(s[1]) - half_extents[1])

        cands = [0.0, T, t_star]
        self._add_candidates_Linf(r, v, half_extents, T, cands)
        
        # De-dup and clamp
        cands = sorted(set(x for x in cands if 0.0 <= x <= T))
        d_vals = [sep_Linf(t) for t in cands]
        t_cpa = cands[int(np.argmin(d_vals))]
        d_min = min(d_vals)
        return t_cpa, d_min

    def half_extents_world_np(self, v):
        """Get world-aligned half-extents based on vehicle orientation"""
        return (np.array([v.length*0.5, v.width*0.5]) if v.approach_angle in (0, 180)
                else np.array([v.width*0.5, v.length*0.5]))

    def check_collisions_continuous(self):
        """Proper CPA-based collision detection with orientation-aware extents"""
        if len(self.vehicles) < 2:
            return
        
        self.collision_detected = False
        self.collision_pairs = []
        H, dt = 8, self.dt   # lookahead steps

        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                vi, vj = self.vehicles[i], self.vehicles[j]
                p_i = np.array([vi.x, vi.y])
                p_j = np.array([vj.x, vj.y])
                v_i = self._get_velocity_vector(vi)
                v_j = self._get_velocity_vector(vj)
                
                # Proper orientation-aware extents
                ext = self.half_extents_world_np(vi) + self.half_extents_world_np(vj)
                t_cpa, d_min = self.cpa_rect(p_i, v_i, p_j, v_j, ext, H, dt)
                
                if d_min <= 0.0:
                    self.collision_detected = True
                    self.collision_pairs.append((vi.id, vj.id))
                    self.collision_alert_time = self.current_step  # Update collision counter
                    if self._warn_cooldown <= 0.0:
                        print(f"PREDICTED COLLISION! V{vi.id}–V{vj.id} in ~{t_cpa:.2f}s (d_min={d_min:.2f}m)")
                        self._warn_cooldown = 0.5  # seconds
                elif d_min < 2.0 and t_cpa < 3.0:  # Close call warning
                    if self._warn_cooldown <= 0.0:
                        print(f"CLOSE CALL! V{vi.id}–V{vj.id} will be {d_min:.1f}m apart in {t_cpa:.2f}s")
                        self._warn_cooldown = 0.5  # seconds
    
    def arrival_window(self, distance, v, a_max, a_min):
        """Earliest/latest arrival using accel bounds. a_min ≤ 0."""
        eps = 1e-6

        # Earliest: accelerate (or coast) as hard as allowed
        if a_max > eps:
            disc = max(v*v + 2*a_max*distance, eps)
            t_min = (-v + np.sqrt(disc)) / max(a_max, eps)
        else:
            t_min = distance / max(v, eps)

        # Latest (conservative): brake at a_min until either stop or reach center,
        # then creep at a small v_min if we had to stop before reaching the center.
        v_min = max(0.5, 0.1 * max(v, 1.0))  # small crawl speed (m/s)
        if a_min < -eps and v > eps:
            t_stop = -v / a_min  # a_min is negative
            d_stop = v * t_stop + 0.5 * a_min * t_stop * t_stop  # = v^2/(-2 a_min)
            if d_stop >= distance:
                # reach center before (fully) stopping: solve v*t + 0.5*a_min*t^2 = distance
                disc2 = max(v*v + 2*a_min*distance, eps)
                t_max = (-v + np.sqrt(disc2)) / max(a_min, -eps)  # denominator negative
                t_max = abs(t_max)
            else:
                # stop first, then creep remaining distance
                d_left = distance - d_stop
                t_max = t_stop + d_left / v_min
        else:
            # no braking capability—coast slow
            t_max = distance / max(v_min, eps)

        # sanity: ensure t_max ≥ t_min
        t_min = max(t_min, 0.0)
        t_max = max(t_max, t_min + 0.1)
        return t_min, t_max

    def update_intersection_schedule(self):
        """Intersection scheduling with acceleration-bounded arrival windows"""
        current_time = self.current_step * self.dt
        
        # Clean up old reservations
        for zone in self.conflict_zones.values():
            zone['reservations'] = [(vid, start, end) for vid, start, end in zone['reservations'] 
                                  if end > current_time]
        
        # Check which vehicles need reservations
        for vehicle in self.vehicles:
            # Estimate distance to intersection center
            if vehicle.approach_angle in (0, 180):  # North-South
                distance_to_center = abs(vehicle.x)
            else:  # East-West
                distance_to_center = abs(vehicle.y)
            
            if distance_to_center < 30.0 and vehicle.velocity > 1.0:  # Approaching intersection
                # Calculate arrival window with acceleration bounds
                t_earliest, t_latest = self.arrival_window(
                    distance_to_center, vehicle.velocity, 
                    vehicle.max_accel, -vehicle.max_decel * 0.3  # Conservative decel
                )
                
                arrival_time = current_time + t_earliest
                # consistent crawl speed with arrival_window()
                v_exit = max(vehicle.velocity, 0.5)
                exit_time = current_time + t_latest + self.intersection_size / v_exit + self.headway_time
                
                # Check if window is available
                zone = self.conflict_zones['center']
                conflict = False
                for other_id, start, end in zone['reservations']:
                    if other_id != vehicle.id and not (exit_time <= start or arrival_time >= end):
                        conflict = True
                        break
                
                # Reserve window if available
                if not conflict:
                    # Remove any existing reservation for this vehicle
                    zone['reservations'] = [(vid, start, end) for vid, start, end in zone['reservations'] 
                                          if vid != vehicle.id]
                    # Add new reservation
                    zone['reservations'].append((vehicle.id, arrival_time, exit_time))
                    zone['reservations'].sort(key=lambda x: x[1])  # Sort by arrival time
    
    def _axis_proj(self, vehicle, ego_angle):
        """Get 1D projection for ego axis to avoid coordinate confusion"""
        if ego_angle in (0, 180):   # use x-axis motion
            p = vehicle.x
            vv = vehicle.velocity if vehicle.approach_angle == 0 else (-vehicle.velocity if vehicle.approach_angle == 180 else 0.0)
        else:                       # use y-axis motion
            p = vehicle.y
            vv = vehicle.velocity if vehicle.approach_angle == 90 else (-vehicle.velocity if vehicle.approach_angle == 270 else 0.0)
        return p, vv

    def cbf_clip(self, u_nom, i, vehicles, dt, d_safe=6.0, alpha=0.2):
        """Proper discrete-time CBF safety filter with numerical safeguards and cross-traffic guard"""
        veh = vehicles[i]
        u_min, u_max = -veh.max_decel, veh.max_accel

        # Cross-traffic guard: check if any vehicle will reach intersection center within 1.0s
        intersection_eta_i = self._time_to_intersection_center(veh)
        
        for j, oth in enumerate(vehicles):
            if j == i:
                continue
            
            # Check for perpendicular cross-traffic conflict
            intersection_eta_j = self._time_to_intersection_center(oth)
            if (abs(intersection_eta_i - intersection_eta_j) < 1.0 and 
                intersection_eta_i < 5.0 and intersection_eta_j < 5.0):
                # Perpendicular conflict detected - increase safety margin
                alpha = min(0.5, alpha * 2.0)  # More conservative CBF
                u_max = min(u_max, 0.5)        # also cap aggressive acceleration
            
            p_i, v_i = self._axis_proj(veh, veh.approach_angle)
            p_j, v_j = self._axis_proj(oth, veh.approach_angle)  # project other onto ego axis
            s = p_i - p_j
            v_rel = v_i - v_j
            h = s * s - d_safe * d_safe

            # One-step forward: s+ = s + v_rel*dt + 0.5*u*dt^2
            # Linearize h+ around u=0: h+ ≈ (s + v_rel*dt)^2 - d^2 + (s + v_rel*dt)*dt^2 * u
            s_bar = s + v_rel * dt
            A = s_bar * (dt * dt)  # coefficient of u
            c = (s_bar * s_bar) - (1.0 - alpha) * (s * s) - alpha * (d_safe * d_safe)

            # Numerical floor: skip if coefficient is too small
            if abs(A) <= 1e-6:
                continue

            # ZCBF condition: h+ - (1-α)h ≥ 0  ⇒  A*u ≥ -c
            bound = float(np.clip((-c) / A, -veh.max_decel, veh.max_accel))  # Clamp to physical limits
            if A > 0:
                u_min = max(u_min, bound)  # lower bound
            else:
                u_max = min(u_max, bound)  # upper bound

        # Handle infeasible case gracefully
        if u_min > u_max:
            # Pick nearest bound to u_nom
            return u_min if abs(u_nom - u_min) < abs(u_nom - u_max) else u_max
        
        return float(np.clip(u_nom, u_min, u_max))

    def _time_to_intersection_center(self, vehicle):
        """Estimate time for vehicle to reach intersection center"""
        if vehicle.approach_angle in (0, 180):  # North-South
            distance_to_center = abs(vehicle.x)
        else:  # East-West
            distance_to_center = abs(vehicle.y)
        
        if vehicle.velocity < 0.1:
            return float('inf')
        
        return distance_to_center / vehicle.velocity
    
    def distance_based_safety_filter(self, control_inputs: List[float]) -> List[float]:
        """DEPRECATED: Use proper CBF safety filter instead"""
        # Apply proper CBF safety filter
        safe_controls = []
        for i, u_nom in enumerate(control_inputs):
            u_safe = self.cbf_clip(u_nom, i, self.vehicles, self.dt, d_safe=6.0, alpha=0.2)
            safe_controls.append(u_safe)
        return safe_controls
    
    def _get_velocity_vector(self, vehicle: VehicleState) -> np.ndarray:
        """Get 2D velocity vector based on vehicle's approach angle"""
        speed = vehicle.velocity
        if vehicle.approach_angle == 0:  # North
            return np.array([speed, 0.0])
        elif vehicle.approach_angle == 180:  # South
            return np.array([-speed, 0.0])
        elif vehicle.approach_angle == 90:  # East
            return np.array([0.0, speed])
        else:  # West (270)
            return np.array([0.0, -speed])
    
    def cbf_safety_filter(self, control_inputs: List[float]) -> List[float]:
        """DEPRECATED: Use distance_based_safety_filter instead"""
        return self.distance_based_safety_filter(control_inputs)
    
    def simulate_step(self):
        """Main simulation step"""
        if not self.vehicles:
            return
        
        # Solve MPC for all vehicles - performance optimization
        zone = self.conflict_zones['center']
        now = self.current_step * self.dt
        H_soon = 1.0  # match the ~1s "soon" horizon used in compute_costs_batch
        reserved_ids = {
            vid for (vid, start, end) in zone['reservations']
            if not (end <= now or start >= now + H_soon)   # overlaps [now, now+H_soon]
        }
        optimal_accels = self.controller.solve_batch_mpc(self.vehicles, self.current_step, reserved_ids=reserved_ids)
        
        # Track metrics (event-based, pre-motion)
        if self.vehicles:
            for veh in self.vehicles:
                inside_now = self._inside_box(veh.x, veh.y)
                if (not self._inside_box_prev[veh.id]) and inside_now:
                    # just entered; count if not reserved for the upcoming 1s window
                    if veh.id not in reserved_ids:
                        self.metrics['unreserved_attempts'] += 1
                self._inside_box_prev[veh.id] = inside_now
        
        # Apply CBF safety filter - hard safety guarantee
        safe_accels = self.distance_based_safety_filter(optimal_accels)
        
        # Sanity checks for regression catching
        for v, a in zip(self.vehicles, safe_accels):
            assert -v.max_decel - 1e-6 <= a <= v.max_accel + 1e-6, f"Accel {a} out of bounds for vehicle {v.id}"
        
        # Update vehicle states
        for i, vehicle in enumerate(self.vehicles):
            accel = safe_accels[i]
            
            # Update velocity
            vehicle.velocity += accel * self.dt
            vehicle.velocity = np.clip(vehicle.velocity, 0, vehicle.target_velocity * 1.5)  # Increased from 1.2
            
            # Sanity check for negative speeds
            assert vehicle.velocity >= -1e-9, f"Negative velocity {vehicle.velocity} for vehicle {vehicle.id}"
            
            # Update position
            vehicle.update_position(self.dt)
            
            # Completion event (post-motion)
            inside_after = self._inside_box(vehicle.x, vehicle.y)
            if (not inside_after) and self._inside_box_prev[vehicle.id] and (not self._has_completed[vehicle.id]):
                # vehicle was inside this step and is now out of the box => completed a crossing
                self.metrics['completed'] += 1
                self._has_completed[vehicle.id] = True
            # update prev flag for the next frame
            self._inside_box_prev[vehicle.id] = inside_after
            
            # Store history
            self.history_x[vehicle.id].append(vehicle.x)
            self.history_y[vehicle.id].append(vehicle.y)
            self.history_v[vehicle.id].append(vehicle.velocity)
        
        # Update intersection reservations
        self.update_intersection_schedule()
        
        # Check collisions with continuous detection
        self.check_collisions_continuous()
        self.current_step += 1
    
    def create_car_shape(self, x: float, y: float, angle: int, length: float = 4.5, width: float = 2.0) -> np.ndarray:
        """Create detailed car shape"""
        half_length, half_width = length / 2, width / 2
        
        car_points = np.array([
            [-half_length, -half_width * 0.9],
            [-half_length * 0.8, -half_width],
            [half_length * 0.3, -half_width],
            [half_length * 0.9, -half_width * 0.7],
            [half_length, 0],
            [half_length * 0.9, half_width * 0.7],
            [half_length * 0.3, half_width],
            [-half_length * 0.8, half_width],
            [-half_length, half_width * 0.9],
        ])
        
        # Rotate and translate
        angle_rad = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        rotated_points = car_points @ rotation_matrix.T
        rotated_points[:, 0] += x
        rotated_points[:, 1] += y
        
        return rotated_points
    
    def setup_plot(self):
        """Initialize visualization"""
        # Main plot
        self.ax_main.set_xlim(-100, 100)
        self.ax_main.set_ylim(-100, 100)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_title('High-Performance Intersection Control', fontsize=16, fontweight='bold')
        self.ax_main.set_xlabel('X Position (m)')
        self.ax_main.set_ylabel('Y Position (m)')
        self.ax_main.set_facecolor('#1a1a2e')
        
        # Draw infrastructure
        intersection = Rectangle((-self.intersection_size/2, -self.intersection_size/2),
                               self.intersection_size, self.intersection_size,
                               fill=True, facecolor='#404050', edgecolor='white',
                               linewidth=2, alpha=0.8)
        self.ax_main.add_patch(intersection)
        
        # Draw roads
        road_length = 100
        for direction in ['north', 'south', 'east', 'west']:
            if direction in ['north', 'south']:
                if direction == 'north':
                    road = Rectangle((-self.lane_width/2, 0), self.lane_width/2, road_length,
                                   fill=True, facecolor='#505060', alpha=0.7)
                else:
                    road = Rectangle((0, -road_length), self.lane_width/2, road_length,
                                   fill=True, facecolor='#505060', alpha=0.7)
            else:
                if direction == 'east':
                    road = Rectangle((0, -self.lane_width/2), road_length, self.lane_width/2,
                                   fill=True, facecolor='#505060', alpha=0.7)
                else:
                    road = Rectangle((-road_length, 0), road_length, self.lane_width/2,
                                   fill=True, facecolor='#505060', alpha=0.7)
            self.ax_main.add_patch(road)
        
        # Lane lines
        line_color = '#ffff00'
        self.ax_main.plot([0, 0], [-road_length, -self.intersection_size/2], color=line_color, linestyle='--', linewidth=2, alpha=0.6)
        self.ax_main.plot([0, 0], [self.intersection_size/2, road_length], color=line_color, linestyle='--', linewidth=2, alpha=0.6)
        self.ax_main.plot([-road_length, -self.intersection_size/2], [0, 0], color=line_color, linestyle='--', linewidth=2, alpha=0.6)
        self.ax_main.plot([self.intersection_size/2, road_length], [0, 0], color=line_color, linestyle='--', linewidth=2, alpha=0.6)
        
        # Speed plot
        self.ax_speed.set_xlim(0, 60)
        self.ax_speed.set_ylim(0, 20)
        self.ax_speed.grid(True, alpha=0.3)
        self.ax_speed.set_title('Vehicle Speeds', fontsize=14, fontweight='bold')
        self.ax_speed.set_xlabel('Time (s)')
        self.ax_speed.set_ylabel('Speed (m/s)')
        
        # Initialize vehicle visualizations
        colors = ['#00aaff', '#ff4444', '#44ff44', '#ffaa00', '#aa44ff', '#ff8844', '#ff44aa', '#44ffff'] * 3
        
        for i, vehicle in enumerate(self.vehicles):
            color = colors[i % len(colors)]
            vehicle.color = color
            
            # Car shape
            car_points = self.create_car_shape(vehicle.x, vehicle.y, vehicle.approach_angle)
            self.vehicle_shapes[vehicle.id] = Polygon(car_points, facecolor=color, edgecolor='white',
                                                    linewidth=2, alpha=0.95)
            self.ax_main.add_patch(self.vehicle_shapes[vehicle.id])
            
            # Trail
            self.vehicle_trails[vehicle.id], = self.ax_main.plot([], [], color=color, alpha=0.6, linewidth=2)
            
            # Speed line
            self.speed_lines[vehicle.id], = self.ax_speed.plot([], [], color=color, linewidth=2, label=f'V{vehicle.id}')
            
            # Predicted path
            self.predicted_paths[vehicle.id], = self.ax_main.plot([], [], color=color, linestyle=':', alpha=0.5, linewidth=1)
        
        if self.vehicles:
            self.ax_speed.legend(loc='upper right')
        
        # Metrics text
        self.metrics_text = self.ax_main.text(0.02, 0.98, '', transform=self.ax_main.transAxes,
                                            verticalalignment='top', fontsize=11, fontweight='bold',
                                            color='white',
                                            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d44',
                                                    edgecolor='#00ff88', linewidth=2, alpha=0.95))
    
    def animate(self, frame):
        """Optimized animation update with frame skipping"""
        if not self.plot_initialized:
            return []
        
        # Frame skipping for performance
        self._skip_frames += 1
        if self._skip_frames < self._frame_skip_count:
            return []
        self._skip_frames = 0
        
        # Update warning cooldown
        self._warn_cooldown = max(0.0, self._warn_cooldown - self.dt)
        
        if frame < 1000:
            # Run simulation step
            self.simulate_step()
            
            # Update visualizations
            for vehicle in self.vehicles:
                if vehicle.id in self.vehicle_shapes:
                    # Update car shape
                    car_points = self.create_car_shape(vehicle.x, vehicle.y, vehicle.approach_angle)
                    self.vehicle_shapes[vehicle.id].set_xy(car_points)
                    
                    # Update trail
                    if len(self.history_x[vehicle.id]) > 1:
                        self.vehicle_trails[vehicle.id].set_data(
                            self.history_x[vehicle.id][-50:],
                            self.history_y[vehicle.id][-50:]
                        )
                    
                    # Update speed plot
                    times = [i * self.dt for i in range(len(self.history_v[vehicle.id]))]
                    self.speed_lines[vehicle.id].set_data(times, self.history_v[vehicle.id])
                    
                    # Auto-extend speed plot x-axis for long simulations
                    if times and times[-1] > self.ax_speed.get_xlim()[1]:
                        self.ax_speed.set_xlim(0, times[-1] + 5)
                    
                    # Update predicted path - optimized GPU trajectory handling
                    if hasattr(vehicle, '_gpu_trajectory') and vehicle._gpu_trajectory is not None:
                        # Only transfer small subset to CPU for visualization
                        with torch.no_grad():
                            pred_subset = vehicle._gpu_trajectory[:10].cpu().numpy()  # Reduced from 20 to 10
                            pred_x = pred_subset[:, 0]
                            pred_y = pred_subset[:, 1]
                            self.predicted_paths[vehicle.id].set_data(pred_x, pred_y)
                    elif hasattr(vehicle, 'planned_trajectory') and vehicle.planned_trajectory is not None:
                        pred_x = vehicle.planned_trajectory[:10, 0]  # Reduced points
                        pred_y = vehicle.planned_trajectory[:10, 1]
                        self.predicted_paths[vehicle.id].set_data(pred_x, pred_y)
            
            # Update metrics
            current_speeds = [v.velocity for v in self.vehicles]
            if current_speeds:
                min_speed = min(current_speeds)
                max_speed = max(current_speeds)
                avg_speed = np.mean(current_speeds)
                
                # Minimum distance
                if len(self.vehicles) > 1:
                    positions = np.array([[v.x, v.y] for v in self.vehicles])
                    pos_diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
                    distances = np.linalg.norm(pos_diff, axis=2)
                    distances[distances == 0] = np.inf
                    min_distance = np.min(distances)
                else:
                    min_distance = float('inf')
                
                # Vehicles in intersection
                in_intersection = sum(1 for v in self.vehicles
                                    if abs(v.x) < self.intersection_size/2 + 5 and
                                       abs(v.y) < self.intersection_size/2 + 5)
                
                time_elapsed = frame * self.dt
                efficiency = (avg_speed / np.mean([v.target_velocity for v in self.vehicles])) * 100
                
                metrics_str = f"TIME: {time_elapsed:.1f}s\n"
                metrics_str += f"VEHICLES: {len(self.vehicles)} active\n"
                metrics_str += f"IN INTERSECTION: {in_intersection}\n"
                metrics_str += f"MIN DISTANCE: {min_distance:.1f}m\n"
                metrics_str += f"SPEEDS: {min_speed:.1f} - {max_speed:.1f} m/s\n"
                metrics_str += f"AVG SPEED: {avg_speed:.1f} m/s\n"
                metrics_str += f"EFFICIENCY: {efficiency:.0f}%\n"
                metrics_str += f"UNRESERVED ATTEMPTS: {self.metrics['unreserved_attempts']}\n"
                metrics_str += f"COMPLETED: {self.metrics['completed']}\n"
                
                if self.collision_detected:
                    collision_pairs_str = ", ".join([f"V{p[0]}-V{p[1]}" for p in self.collision_pairs])
                    metrics_str += f"COLLISION: {collision_pairs_str}\n"
                    metrics_str += f"SAFETY: COLLISION DETECTED!"
                else:
                    metrics_str += f"SAFETY: {'SAFE' if min_distance > 8.0 else 'CLOSE' if min_distance > 5.0 else 'DANGER'}"
                
                self.metrics_text.set_text(metrics_str)
        
        return list(self.vehicle_shapes.values()) + list(self.vehicle_trails.values())
    
    def finalize_setup(self):
        if not self.plot_initialized:
            self.setup_plot()
            self.plot_initialized = True
    
    def run_animation(self):
        self.finalize_setup()
        # Optimized animation settings for high performance
        ani = animation.FuncAnimation(self.fig, self.animate, frames=1000,
                                    interval=50, blit=False, repeat=False,  # Adjusted for stability
                                    cache_frame_data=False)  # Reduce memory usage
        plt.tight_layout()
        plt.show(block=True)  # Block to keep window open
        return ani


def create_scenario():
    """Create optimized scenario with priority-based coordination"""
    sim = IntersectionSimulation(max_vehicles=8)
    
    # Assign priorities and safe initial positions to prevent immediate collisions
    vehicles = [
        VehicleState(1, np.array([-80.0, 0.0]), 12.0, 20.0, 0, 0, 0.8),   # North approach - increased from 15.0
        VehicleState(2, np.array([-120.0, 0.0]), 11.0, 20.0, 0, 0, 0.6),  # North approach, further back - increased from 15.0
        VehicleState(3, np.array([85.0, 0.0]), 13.0, 20.0, 180, 0, 0.7),  # South approach - increased from 15.0
        VehicleState(4, np.array([125.0, 0.0]), 12.0, 20.0, 180, 0, 0.5), # South approach, further back - increased from 15.0
        VehicleState(5, np.array([0.0, -75.0]), 10.0, 20.0, 90, 0, 0.9),  # East approach - increased from 15.0
        VehicleState(6, np.array([0.0, -115.0]), 11.0, 20.0, 90, 0, 0.4), # East approach, further back - increased from 15.0
        VehicleState(7, np.array([0.0, 80.0]), 9.0, 20.0, 270, 0, 0.3),   # West approach - increased from 15.0
        VehicleState(8, np.array([0.0, 120.0]), 10.0, 20.0, 270, 0, 0.2), # West approach, furthest back - increased from 15.0
    ]
    
    for vehicle in vehicles:
        sim.add_vehicle(vehicle)
    
    return sim


if __name__ == "__main__":
    print("High-Performance Intersection Control")
    print(f"Device: {device}")
    print("Features:")
    print("  - Predictive Collision Avoidance")
    print("  - Priority-Based Coordination")
    print("  - Batched GPU processing")
    print("  - Vectorized operations")
    print("  - Optimal data structures")
    print("  - Clean architecture")
    print("\nStarting simulation...")
    
    try:
        sim = create_scenario()
        animation_object = sim.run_animation()
        
        # Keep the program running to display animation
        input("\nPress Enter to exit...")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nSimulation error: {e}")
    finally:
        plt.close('all')
        print("Simulation complete!")