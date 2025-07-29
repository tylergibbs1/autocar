"""
Modern Autonomous Vehicle Traffic Simulation 
Testing hypothesis: Can autonomous vehicles eliminate traffic lights through coordination?
"""
import torch
import torch.nn as nn
import numpy as np

# Disable automatic compilation that requires C++ compiler
torch._dynamo.config.suppress_errors = True
torch.backends.cudnn.enabled = False
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque

# Reproducibility seeds
torch.manual_seed(0)
np.random.seed(0)
import random
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

def pick_device(n_vehicles: int, horizon: int) -> torch.device:
    """Pick device based on workload size - avoid GPU tax for small batches"""
    work = n_vehicles * horizon
    use_cuda = torch.cuda.is_available() and work >= 1500
    return torch.device('cuda' if use_cuda else 'cpu')

# GPU setup with workload-aware device selection - will be set in main
device = None

# Enable optimized matmul on Ampere+ GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

# Module-level constants to avoid rebuilding
DIR_MAP = {
    0: (1.0, 0.0),    # North
    90: (0.0, 1.0),   # East  
    180: (-1.0, 0.0), # South
    270: (0.0, -1.0), # West
}

@dataclass(slots=True)
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
    
    # Pre-rotated car shape template for performance
    _shape_template: Optional[np.ndarray] = None
    
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
    def __init__(self, horizon: int = 15, max_vehicles: int = 20):  # Reduced from 25 for better performance
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
        # limit to batch size for nearest neighbor context
        others = xyv[:batch_size, :2]  # limit to batch
        for i in range(batch_size):
            # own state
            state_cpu[i, :3] = xyv[i]
            # others (nearest-first for more informative context, same input size)
            fill, idx = 0, 3
            d = np.linalg.norm(others - xyv[i, :2], axis=1)
            # take K nearest including self, then skip self
            K = min(max_others + 1, batch_size)
            order = np.argpartition(d, kth=K-1)[:K]
            for j in order:
                if j == i or fill >= max_others: continue
                state_cpu[i, idx:idx+3] = xyv[j]
                idx += 3; fill += 1
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
        """Vectorized trajectory prediction with cumsums"""
        B = control_batch.shape[0]
        # initial state
        xyv0 = torch.tensor([[v.x, v.y, v.velocity] for v in vehicles[:B]],
                            device=device, dtype=torch.float32)                         # [B,3]
        dirs = torch.tensor([DIR_MAP[v.approach_angle] for v in vehicles[:B]],
                            device=device, dtype=torch.float32)                         # [B,2]
        target = torch.tensor([v.target_velocity*1.2 for v in vehicles[:B]],
                              device=device, dtype=torch.float32)                       # [B]

        # accelerations per step
        acc = control_batch.clamp(-1.0, 1.0) * 3.0                                     # [B,H]
        vel_pre = xyv0[:, 2:3] + self.dt * acc.cumsum(dim=1)                           # [B,H]
        vel = torch.minimum(vel_pre.clamp_min_(0.0), target.unsqueeze(1))              # [B,H]

        # positions
        step_disp = (vel * self.dt).unsqueeze(-1) * dirs.unsqueeze(1)                  # [B,H,2]
        pos = xyv0[:, :2].unsqueeze(1) + step_disp.cumsum(dim=1)                       # [B,H,2]

        # write into prealloc buffer
        out = self.trajectory_batch[:B]
        out[:, 0, :2] = xyv0[:, :2]
        out[:, 0, 2]  = xyv0[:, 2]
        out[:, 1:, :2] = pos
        out[:, 1:,  2] = vel
        return out
    
    def compute_costs_batch(self, control_batch: torch.Tensor, trajectory_batch: torch.Tensor, vehicles: List[VehicleState], reserved_ids: Optional[set] = None, center_half: float = 10.0, soon_steps: int = 10) -> torch.Tensor:
        """Vectorized cost computation"""
        batch_size = control_batch.shape[0]
        
        # soft reservation penalty
        if reserved_ids is None:
            reserved_mask = torch.zeros(batch_size, device=device)
        else:
            reserved_mask = torch.tensor([1.0 if v.id in reserved_ids else 0.0 for v in vehicles[:batch_size]],
                                         device=device)
        
        # Vectorized look-ahead proximity computation (avoid in-place ops for autograd)
        K = min(4, self.horizon + 1)  # Reduced from 6 for better performance
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
        
        # VECTORIZED velocity tracking with adaptive targets - NO PYTHON LOOP
        vel = trajectory_batch[:batch_size, 1:, 2]                   # [B,H] future velocities
        B, H = vel.shape
        
        # Build target velocities tensor
        target = torch.tensor([v.target_velocity for v in vehicles[:batch_size]], 
                             device=device).unsqueeze(1)             # [B,1]
        
        # Adaptive target based on proximity
        near_mask = (min_distance_to_others < 20.0)                  # [B]
        adaptive_target = torch.where(
            near_mask.unsqueeze(1),
            target * (0.3 + 0.7 * (min_distance_to_others / 20.0).unsqueeze(1)),
            target
        )                                                            # [B,1]
        
        # Velocity weights based on proximity
        velocity_weight = torch.where(
            near_mask, 
            torch.tensor(10.0, device=device), 
            torch.tensor(25.0, device=device)
        ).unsqueeze(1)                                               # [B,1]
        
        # Velocity tracking cost
        vel_errors = vel - adaptive_target                           # [B,H]
        velocity_costs = (velocity_weight * (vel_errors ** 2)).sum(dim=1)  # [B]
        
        # Jerk minimization (smooth velocity changes)
        if H > 1:
            vel_changes = vel[:, 1:] - vel[:, :-1]                   # [B,H-1]
            velocity_costs = velocity_costs + 5.0 * (vel_changes ** 2).sum(dim=1)
        
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
            control_costs += 10.0 * extreme_changes.float().sum(dim=1)
        
        # Encourage only the reserved vehicle to enter the center soon
        # Heuristic: if predicted position crosses the center bounds within ~1s, penalize unreserved
        K_check = min(soon_steps, self.horizon + 1)
        pos_check = trajectory_batch[:batch_size, :K_check, :2]          # [B,K,2]
        soon = ((pos_check[..., 0].abs() < center_half + 2.0) &
                (pos_check[..., 1].abs() < center_half + 2.0)).any(dim=1).float()  # [B]
        
        # Check if vehicles are already inside intersection (don't penalize them)
        already_inside = torch.tensor([
            1.0 if (abs(v.x) < center_half + 2.0 and abs(v.y) < center_half + 2.0) else 0.0 
            for v in vehicles[:batch_size]
        ], device=device)
        
        # penalty only if soon and not reserved AND not already inside
        not_reserved = (1.0 - reserved_mask)
        not_inside = (1.0 - already_inside)
        reservation_costs = 200.0 * soon * not_reserved * not_inside
        
        # Advanced predictive collision avoidance cost using CPA
        collision_costs = torch.zeros(batch_size, device=device)
        
        # Vectorized car-following costs
        following_costs = self._compute_following_costs_vectorized(trajectory_batch, vehicles, batch_size)

        if batch_size > 1:
            # states at first predicted step
            pos1 = trajectory_batch[:batch_size, 1, :2]    # [B,2]
            vel1 = trajectory_batch[:batch_size, 1, 2]     # [B]

            # Use cached tensors for better performance
            dirs = self._dirs[:batch_size] if hasattr(self, '_dirs') and self._dirs is not None else torch.tensor([DIR_MAP[v.approach_angle] for v in vehicles[:batch_size]], device=device, dtype=torch.float32)
            half_exts = self._half_exts[:batch_size] if hasattr(self, '_half_exts') and self._half_exts is not None else torch.tensor([[(veh.length*0.5 if veh.approach_angle in (0,180) else veh.width*0.5), (veh.width*0.5  if veh.approach_angle in (0,180) else veh.length*0.5)] for veh in vehicles[:batch_size]], device=device, dtype=torch.float32)

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

            margin, tau = 3.0, 1.0  # Increased margin significantly, reduced tau for immediate response
            gap = margin - d_min
            dist_cost = 50000.0 * torch.relu(gap) ** 2  # Much higher cost penalty
            time_weight = torch.exp(-t_star / tau)                                         # [B,B]
            base_pair_cost = dist_cost * time_weight                                       # [B,B]

            # Zero self-pairs and (optionally) keep only upper triangle to avoid double work
            eye = torch.eye(batch_size, device=device, dtype=torch.bool)
            base_pair_cost = base_pair_cost.masked_fill(eye, 0.0)

            # Priority-based weights (for "cost applied to i" and to j)
            pr = self._priorities[:batch_size] if hasattr(self, '_priorities') and self._priorities is not None else torch.tensor([veh.priority for veh in vehicles[:batch_size]], device=device, dtype=torch.float32)
            wi = torch.where(pr[:, None] > pr[None, :], 0.3,
                 torch.where(pr[:, None] < pr[None, :], 1.0, 0.7))                          # [B,B]

            # Row-sum gives per-vehicle contribution (same behavior as your loop)
            collision_costs = (base_pair_cost * wi).sum(dim=1)                              # [B]
        
        return velocity_costs + control_costs + reservation_costs + collision_costs + following_costs
    
    def _compute_following_costs_vectorized(self, trajectory_batch: torch.Tensor, vehicles: List[VehicleState], batch_size: int) -> torch.Tensor:
        """Vectorized car-following cost computation"""
        following_costs = torch.zeros(batch_size, device=device)
        
        if batch_size < 2:
            return following_costs
            
        # Get vehicle properties as tensors
        angles = torch.tensor([v.approach_angle for v in vehicles[:batch_size]], device=device)
        x = trajectory_batch[:batch_size, 1, 0]  # pos after first step
        y = trajectory_batch[:batch_size, 1, 1]
        v_now = trajectory_batch[:batch_size, 1, 2]
        
        # Initialize lead distance and speed tracking
        lead_dist = torch.full((batch_size,), float('inf'), device=device)
        lead_speed = torch.zeros(batch_size, device=device)
        
        # Process each direction separately
        for ang in (0, 90, 180, 270):
            mask = (angles == ang)
            idx = mask.nonzero(as_tuple=True)[0]
            if idx.numel() < 2: 
                continue
                
            if ang in (0, 180):  # North/South - use x coordinate
                order = torch.argsort(x[idx])
                if ang == 180: 
                    order = torch.flip(order, dims=[0])
                ordered = idx[order]
                # distances to next car
                if len(ordered) > 1:
                    d = x[ordered][1:] - x[ordered][:-1]
                    same_lane = (y[ordered][1:] - y[ordered][:-1]).abs() < 4.0
                    sel = ordered[:-1]
                    lead_dist[sel] = torch.where(same_lane, d.abs(), lead_dist[sel])
                    lead_speed[sel] = torch.where(same_lane, v_now[ordered][1:], lead_speed[sel])
            else:  # East/West - use y coordinate
                order = torch.argsort(y[idx])
                if ang == 270: 
                    order = torch.flip(order, dims=[0])
                ordered = idx[order]
                if len(ordered) > 1:
                    d = y[ordered][1:] - y[ordered][:-1]
                    same_lane = (x[ordered][1:] - x[ordered][:-1]).abs() < 4.0
                    sel = ordered[:-1]
                    lead_dist[sel] = torch.where(same_lane, d.abs(), lead_dist[sel])
                    lead_speed[sel] = torch.where(same_lane, v_now[ordered][1:], lead_speed[sel])
        
        # Compute following costs
        desired = torch.maximum(torch.full_like(lead_dist, 15.0), v_now * 2.0)
        too_close = (lead_dist < 40.0) & (lead_dist < desired)
        
        distance_error = (desired - lead_dist).clamp_min(0.0)
        speed_diff = (v_now - lead_speed).clamp_min(0.0)
        
        following_costs = torch.where(
            too_close, 
            100.0 * distance_error**2 + 60.0 * speed_diff**2,
            following_costs
        )
        
        return following_costs
    
    def solve_batch_mpc(self, vehicles: List[VehicleState], time_step: int, reserved_ids: Optional[set] = None, cached_dirs=None, cached_half_exts=None, cached_priorities=None) -> List[float]:
        """High-performance MPC solver with CUDA streams"""
        if not vehicles:
            return []
        
        batch_size = len(vehicles)
        
        # Store cached tensors for use in compute_costs_batch
        self._dirs = cached_dirs
        self._half_exts = cached_half_exts
        self._priorities = cached_priorities
        
        # Debug assertions to catch shape mismatches
        assert batch_size <= self.max_vehicles, f"Batch size {batch_size} exceeds max {self.max_vehicles}"
        
        # Use compute stream for main computation
        if self.compute_stream is not None:
            with torch.cuda.stream(self.compute_stream):
                state_batch = self.encode_states_batch(vehicles, time_step)
                
                # Neural network prediction - inference only, no gradients needed
                with torch.inference_mode():
                    control_pred_batch = self.neural_mpc(state_batch)
        else:
            state_batch = self.encode_states_batch(vehicles, time_step)
            # Neural network prediction - inference only, no gradients needed
            with torch.inference_mode():
                control_pred_batch = self.neural_mpc(state_batch)
        
        # Predictive gradient descent for collision avoidance
        control_batch = control_pred_batch.clone().detach()
        learning_rate = 0.15  # Balanced learning rate for stability
        
        # Clear any existing gradients for optimal performance  
        for param in self.neural_mpc.parameters():
            param.grad = None
        
        # More iterations for better collision avoidance planning
        max_iterations = 8  # Reduced from 15 for better performance
        best_controls = control_batch.clone()
        best_total = torch.full((), float('inf'), device=device)
        
        # EMA tracking for early stopping without CPU sync
        ema_grad = torch.tensor(0.0, device=device)
        
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
                # Branchless gradient clipping - no CPU sync
                grad_norm = grad.norm(p=2).clamp_min(1e-6)
                scale = (2.0 / grad_norm).clamp_max(1.0)
                grad.mul_(scale)
                
                # Update controls with momentum-like step
                control_batch = control_batch - learning_rate * grad
                
                # Clamp controls
                control_batch = torch.clamp(control_batch, -1.0, 1.0)
                
                # Track best solution without CPU sync - keep true best
                if (total_cost < best_total).item():
                    best_total = total_cost.detach()
                    best_controls = control_batch.clone()
                
                # Early stopping with EMA tracking (no CPU sync)
                if iteration > 3:
                    ema_grad = 0.9 * ema_grad + 0.1 * grad_norm
                    if (ema_grad < 0.01).all():  # tensor comparison for 0-d tensor
                        break
        
        # Use best solution found
        control_batch = best_controls
        
        # Update planned trajectories (final forward pass) - no gradients needed
        with torch.inference_mode():
            final_trajectories = self.predict_trajectories_batch(control_batch, vehicles)
            # Keep trajectories on GPU, only transfer when needed for visualization
            for i, vehicle in enumerate(vehicles):
                if hasattr(vehicle, '_gpu_trajectory'):
                    vehicle._gpu_trajectory = final_trajectories[i]
                else:
                    vehicle.planned_trajectory = final_trajectories[i].cpu().numpy()
        
        # NO MORE torch.cuda.synchronize() - kills stream overlap!
        
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
        
        # Try to compile the neural network for extra speed
        try:
            if hasattr(torch, 'compile'):
                neural_mpc = torch.compile(neural_mpc, fullgraph=False)
        except Exception as e:
            print(f"Note: torch.compile failed ({e}), continuing without compilation")
            pass  # compilation failed -> ignore
            
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
        
        # Car respawning system
        self.next_vehicle_id = max_vehicles + 1  # Start IDs after initial vehicles
        self.respawn_distance = 80.0  # Distance from center to respawn cars
        self.despawn_distance = 120.0  # Distance from center to remove cars
        
        # Visual collision tracking (matches what user sees)
        self.visual_collision_detected = False
        self.visual_collision_pairs: List[Tuple[int, int]] = []
        
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
        
        # History storage (ring buffers for efficiency - no realloc/GC churn)
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
        self._frame_skip_count = 1  # Reduced from 2 - update every frame but with optimized rendering
        
        # Cached tensors to avoid reallocation
        self._cached_angles = None
        self._dirs = None
        self._half_exts = None
        self._priorities = None
    
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
        
        # Pre-compute rotated car shape template for performance
        self._precompute_car_template(vehicle)
        
        self.vehicles.append(vehicle)
        # Use ring buffers with fixed size to avoid realloc/GC churn
        self.history_x[vehicle.id] = deque(maxlen=64)
        self.history_y[vehicle.id] = deque(maxlen=64)
        self.history_v[vehicle.id] = deque(maxlen=64)
        self._inside_box_prev[vehicle.id] = False
        self._has_completed[vehicle.id] = False
        # Invalidate cached tensors when adding vehicle
        self._cached_angles = None
        
        # Setup visualization for new vehicle if plot is initialized
        if self.plot_initialized:
            self._setup_vehicle_visualization(vehicle)
            # Update legend if needed
            self.ax_speed.legend(loc='upper right')
    
    def _update_cached_tensors(self):
        """Update cached tensors when vehicle roster changes"""
        angles = tuple(v.approach_angle for v in self.vehicles)
        if angles != self._cached_angles:
            self._cached_angles = angles
            batch_size = len(self.vehicles)
            
            self._dirs = torch.tensor([DIR_MAP[a] for a in angles], 
                                    device=device, dtype=torch.float32)
            
            self._half_exts = torch.tensor([
                [(v.length*0.5 if v.approach_angle in (0,180) else v.width*0.5),
                 (v.width*0.5  if v.approach_angle in (0,180) else v.length*0.5)]
                for v in self.vehicles
            ], device=device, dtype=torch.float32)
            
            self._priorities = torch.tensor([v.priority for v in self.vehicles], 
                                         device=device, dtype=torch.float32)
    
    def _precompute_car_template(self, vehicle: VehicleState):
        """Pre-compute rotated car shape template for performance"""
        half_length, half_width = vehicle.length / 2, vehicle.width / 2
        
        # Base car shape template
        base = np.array([
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
        
        # Pre-rotate based on approach angle
        angle_rad = np.radians(vehicle.approach_angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        vehicle._shape_template = base @ rotation_matrix.T
    
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

    def find_vehicle_in_front(self, ego_vehicle: VehicleState, max_distance: float = 50.0) -> Optional[VehicleState]:
        """Find the closest vehicle directly in front in the same lane"""
        closest_vehicle = None
        closest_distance = float('inf')
        
        for other in self.vehicles:
            if other.id == ego_vehicle.id:
                continue
            
            # Check if vehicles are in the same lane (same approach angle)
            if other.approach_angle != ego_vehicle.approach_angle:
                continue
            
            # Calculate if the other vehicle is in front
            if ego_vehicle.approach_angle == 0:  # North - moving in +x direction
                if other.x > ego_vehicle.x:  # Other is ahead
                    distance = other.x - ego_vehicle.x
                    # Check if in same lane (similar y coordinate)
                    if abs(other.y - ego_vehicle.y) < self.lane_width / 2 and distance < max_distance:
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_vehicle = other
                            
            elif ego_vehicle.approach_angle == 180:  # South - moving in -x direction
                if other.x < ego_vehicle.x:  # Other is ahead
                    distance = ego_vehicle.x - other.x
                    if abs(other.y - ego_vehicle.y) < self.lane_width / 2 and distance < max_distance:
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_vehicle = other
                            
            elif ego_vehicle.approach_angle == 90:  # East - moving in +y direction
                if other.y > ego_vehicle.y:  # Other is ahead
                    distance = other.y - ego_vehicle.y
                    if abs(other.x - ego_vehicle.x) < self.lane_width / 2 and distance < max_distance:
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_vehicle = other
                            
            elif ego_vehicle.approach_angle == 270:  # West - moving in -y direction
                if other.y < ego_vehicle.y:  # Other is ahead
                    distance = ego_vehicle.y - other.y
                    if abs(other.x - ego_vehicle.x) < self.lane_width / 2 and distance < max_distance:
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_vehicle = other
        
        return closest_vehicle

    def remove_distant_vehicles(self):
        """Remove vehicles that are too far from the intersection"""
        vehicles_to_remove = []
        for vehicle in self.vehicles:
            distance_from_center = max(abs(vehicle.x), abs(vehicle.y))
            if distance_from_center > self.despawn_distance:
                vehicles_to_remove.append(vehicle)
        
        for vehicle in vehicles_to_remove:
            self.vehicles.remove(vehicle)
            # Clean up visualization elements
            if vehicle.id in self.vehicle_shapes:
                self.vehicle_shapes[vehicle.id].remove()
                del self.vehicle_shapes[vehicle.id]
            if vehicle.id in self.vehicle_trails:
                del self.vehicle_trails[vehicle.id]
            if vehicle.id in self.speed_lines:
                self.speed_lines[vehicle.id].remove()
                del self.speed_lines[vehicle.id]
            if vehicle.id in self.predicted_paths:
                del self.predicted_paths[vehicle.id]
            # Clean up history
            if vehicle.id in self.history_x:
                del self.history_x[vehicle.id]
                del self.history_y[vehicle.id]
                del self.history_v[vehicle.id]
            # Clean up tracking data
            if vehicle.id in self._inside_box_prev:
                del self._inside_box_prev[vehicle.id]
            if vehicle.id in self._has_completed:
                del self._has_completed[vehicle.id]
            
            print(f"Removed vehicle {vehicle.id} (distance: {distance_from_center:.1f}m)")
        
        # Invalidate cached tensors when vehicle roster changes
        if vehicles_to_remove:
            self._cached_angles = None

    def spawn_new_vehicle(self, approach_angle: int):
        """Spawn a new vehicle from the given approach direction"""
        if len(self.vehicles) >= self.max_vehicles:
            return  # Don't exceed maximum vehicles
        
        # Create new vehicle with random properties
        import random
        
        # Position based on approach angle
        if approach_angle == 0:  # North
            position = np.array([-self.respawn_distance, 0.0])
        elif approach_angle == 180:  # South
            position = np.array([self.respawn_distance, 0.0])
        elif approach_angle == 90:  # East
            position = np.array([0.0, -self.respawn_distance])
        else:  # West (270)
            position = np.array([0.0, self.respawn_distance])
        
        # Random vehicle properties
        velocity = random.uniform(8.0, 15.0)
        target_velocity = random.uniform(18.0, 22.0)
        priority = random.uniform(0.1, 0.9)
        
        new_vehicle = VehicleState(
            id=self.next_vehicle_id,
            position=position,
            velocity=velocity,
            target_velocity=target_velocity,
            approach_angle=approach_angle,
            lane=0,
            priority=priority
        )
        
        self.add_vehicle(new_vehicle)
        self.next_vehicle_id += 1
        print(f"Spawned vehicle {new_vehicle.id} from {approach_angle}° (velocity: {velocity:.1f})")

    def manage_vehicle_population(self):
        """Maintain vehicle population by spawning new cars"""
        # Remove distant vehicles first
        self.remove_distant_vehicles()
        
        # Count vehicles by approach direction
        approach_counts = {0: 0, 90: 0, 180: 0, 270: 0}
        for vehicle in self.vehicles:
            approach_counts[vehicle.approach_angle] += 1
        
        # Spawn new vehicles if any direction has too few cars
        min_per_direction = 3  # Minimum vehicles per direction - increased from 2
        import random
        
        # Always maintain at least some vehicles in the simulation
        if len(self.vehicles) < 12:  # If we have fewer than 12 total vehicles - increased from 8
            # Force spawn from a random direction
            directions = [0, 90, 180, 270]
            for direction in directions:
                if approach_counts[direction] == 0 and len(self.vehicles) < self.max_vehicles:
                    self.spawn_new_vehicle(direction)
                    break
        else:
            # Normal probabilistic spawning
            for approach_angle, count in approach_counts.items():
                if count < min_per_direction and len(self.vehicles) < self.max_vehicles:
                    # Higher chance to spawn if we're below minimum
                    spawn_chance = 0.25 if count == 0 else 0.15  # 25% if none, 15% if below min
                    if random.random() < spawn_chance:
                        self.spawn_new_vehicle(approach_angle)
            
            # Additional spawning to maintain high traffic density
            if len(self.vehicles) < self.max_vehicles * 0.8:  # If below 80% capacity
                for approach_angle in [0, 90, 180, 270]:
                    if approach_counts[approach_angle] < 4 and len(self.vehicles) < self.max_vehicles:
                        if random.random() < 0.08:  # 8% chance per direction
                            self.spawn_new_vehicle(approach_angle)

    def check_visual_collisions_realtime(self):
        """Real-time collision detection matching visual representation"""
        if len(self.vehicles) < 2:
            return False, []
        
        visual_collisions = []
        
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                vi, vj = self.vehicles[i], self.vehicles[j]
                
                # Use same dimensions as visual shapes
                # Get current half-extents (not predictive)
                ext_i = self.half_extents_world_np(vi)
                ext_j = self.half_extents_world_np(vj)
                
                # Current positions (not predictive)
                pos_i = np.array([vi.x, vi.y])
                pos_j = np.array([vj.x, vj.y])
                
                # Calculate separation
                delta = pos_i - pos_j
                combined_extents = ext_i + ext_j
                
                # L∞ (rectangular) collision check for current time
                sep_x = abs(delta[0]) - combined_extents[0]
                sep_y = abs(delta[1]) - combined_extents[1]
                
                # Collision if both separations are negative (overlapping)
                if sep_x <= 0.0 and sep_y <= 0.0:
                    visual_collisions.append((vi.id, vj.id))
                # Reduced tolerance - only flag if really close visually
                elif sep_x <= 0.2 and sep_y <= 0.2:
                    # Very close - might appear as collision visually
                    visual_collisions.append((vi.id, vj.id))
        
        return len(visual_collisions) > 0, visual_collisions

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
                        self._warn_cooldown = 2.0  # Increased cooldown to reduce spam
                elif d_min < 1.5 and t_cpa < 2.0:  # Reduced sensitivity for close calls
                    if self._warn_cooldown <= 0.0:
                        print(f"CLOSE CALL! V{vi.id}–V{vj.id} will be {d_min:.1f}m apart in {t_cpa:.2f}s")
                        self._warn_cooldown = 1.5  # Longer cooldown for close calls
    
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

    def cbf_clip(self, u_nom, i, vehicles, dt, d_safe=12.0, alpha=0.5):  # Much larger safety distance and alpha
        """Proper discrete-time CBF safety filter with numerical safeguards and cross-traffic guard"""
        veh = vehicles[i]
        u_min, u_max = -veh.max_decel, veh.max_accel

        # Check if vehicle is in intersection - be less restrictive
        in_intersection = self._inside_box(veh.x, veh.y)
        if in_intersection:
            # Relax constraints for vehicles already in intersection
            d_safe = max(6.0, d_safe * 0.6)  # Still maintain reasonable safety in intersection
            alpha = max(0.2, alpha * 0.5)   # Less conservative in intersection
            u_min = max(u_min, -2.0)        # Limit hard braking in intersection

        # Cross-traffic guard: check if any vehicle will reach intersection center within 1.0s
        intersection_eta_i = self._time_to_intersection_center(veh)
        
        # Car-following check: find vehicle directly in front
        front_vehicle = self.find_vehicle_in_front(veh, max_distance=40.0)
        if front_vehicle is not None:
            # Calculate distance to front vehicle
            if veh.approach_angle == 0:  # North
                front_distance = front_vehicle.x - veh.x
            elif veh.approach_angle == 180:  # South
                front_distance = veh.x - front_vehicle.x
            elif veh.approach_angle == 90:  # East
                front_distance = front_vehicle.y - veh.y
            else:  # West
                front_distance = veh.y - front_vehicle.y
            
            # Apply stronger car-following constraints (but not in intersection)
            if front_distance < 25.0 and not in_intersection:  # Increased from 20.0
                # More conservative parameters for car-following
                d_safe = max(d_safe, front_distance * 0.9)  # Increased from 0.8
                alpha = min(0.9, alpha * 2.5)  # More conservative
                
                # Limit acceleration if getting too close
                if front_distance < 20.0:  # Increased from 15.0
                    u_max = min(u_max, -1.0)  # Stronger deceleration
                elif front_distance < 25.0 and veh.velocity > front_vehicle.velocity:
                    u_max = min(u_max, -0.5)  # Force deceleration if faster
        
        for j, oth in enumerate(vehicles):
            if j == i:
                continue
            
            # Check for perpendicular cross-traffic conflict
            intersection_eta_j = self._time_to_intersection_center(oth)
            if (abs(intersection_eta_i - intersection_eta_j) < 1.0 and 
                intersection_eta_i < 5.0 and intersection_eta_j < 5.0):
                # Perpendicular conflict detected - but less restrictive if in intersection
                if not in_intersection:
                    alpha = min(0.6, alpha * 2.0)  # More conservative CBF
                    u_max = min(u_max, 0.3)        # Cap aggressive acceleration even more
            
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
        
        # Update cached tensors if needed
        self._update_cached_tensors()
        
        # Solve MPC for all vehicles - performance optimization
        zone = self.conflict_zones['center']
        now = self.current_step * self.dt
        H_soon = 1.0  # match the ~1s "soon" horizon used in compute_costs_batch
        reserved_ids = {
            vid for (vid, start, end) in zone['reservations']
            if not (end <= now or start >= now + H_soon)   # overlaps [now, now+H_soon]
        }
        optimal_accels_list = self.controller.solve_batch_mpc(self.vehicles, self.current_step, reserved_ids=reserved_ids, cached_dirs=self._dirs, cached_half_exts=self._half_exts, cached_priorities=self._priorities)
        
        # Convert to tensor and keep on GPU for CBF filtering
        optimal_accels = torch.tensor(optimal_accels_list, device=device, dtype=torch.float32)
        
        # Track metrics (event-based, pre-motion)
        if self.vehicles:
            for veh in self.vehicles:
                inside_now = self._inside_box(veh.x, veh.y)
                if (not self._inside_box_prev[veh.id]) and inside_now:
                    # just entered; count if not reserved for the upcoming 1s window
                    if veh.id not in reserved_ids:
                        self.metrics['unreserved_attempts'] += 1
                self._inside_box_prev[veh.id] = inside_now
        
        # Apply CBF safety filter - stays on GPU, no sync!
        safe_accels = self.cbf_clip_batch(optimal_accels)
        
        # Single CPU transfer only when we need to update Python objects
        safe_accels_cpu = safe_accels.cpu().tolist()
        
        # Sanity checks for regression catching with reasonable tolerance
        for v, a in zip(self.vehicles, safe_accels_cpu):
            if not (-v.max_decel - 0.1 <= a <= v.max_accel + 0.1):
                print(f"Warning: Vehicle {v.id} accel {a:.3f} near limits (bounds: -{v.max_decel:.1f} to {v.max_accel:.1f})")
                # Clamp to safe bounds as fallback
                safe_accels_cpu[self.vehicles.index(v)] = max(-v.max_decel, min(a, v.max_accel))
        
        # Update vehicle states
        for i, vehicle in enumerate(self.vehicles):
            # Handle case where new vehicles were spawned during this step
            if i >= len(safe_accels_cpu):
                accel = 0.0  # New vehicles start with zero acceleration
            else:
                accel = safe_accels_cpu[i]
            
            # Update velocity
            vehicle.velocity += accel * self.dt
            
            # Check if vehicle is in intersection after velocity update
            inside_intersection = self._inside_box(vehicle.x, vehicle.y)
            
            # Apply speed limits based on location
            if inside_intersection:
                # In intersection: prevent complete stops but allow natural deceleration
                vehicle.velocity = max(vehicle.velocity, 1.0)  # Reduced minimum from 2.0 to 1.0
                vehicle.velocity = min(vehicle.velocity, vehicle.target_velocity * 1.2)
            else:
                # Outside intersection: normal speed limits
                vehicle.velocity = np.clip(vehicle.velocity, 0, vehicle.target_velocity * 1.5)
            
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
                
                # Immediately spawn a new vehicle in the same direction
                if len(self.vehicles) < self.max_vehicles:
                    self.spawn_new_vehicle(vehicle.approach_angle)
                    print(f"V{vehicle.id} completed intersection - spawning new vehicle from {vehicle.approach_angle}°")
            # update prev flag for the next frame
            self._inside_box_prev[vehicle.id] = inside_after
            
            # Store history
            self.history_x[vehicle.id].append(vehicle.x)
            self.history_y[vehicle.id].append(vehicle.y)
            self.history_v[vehicle.id].append(vehicle.velocity)
        
        # Update intersection reservations
        self.update_intersection_schedule()
        
        # Check collisions less frequently for performance
        if self.current_step % 3 == 0:  # Every 3rd step
            self.check_collisions_continuous()
        
        # Always check visual collisions for accurate display
        visual_collision_detected, visual_collision_pairs = self.check_visual_collisions_realtime()
        
        # Store both types for metrics
        self.visual_collision_detected = visual_collision_detected
        self.visual_collision_pairs = visual_collision_pairs
        
        # Manage vehicle population (respawn/despawn)
        if self.current_step % 5 == 0:  # Every 5 steps (~0.5 seconds) - increased frequency
            self.manage_vehicle_population()
        
        self.current_step += 1
    
    def create_car_shape(self, vehicle: VehicleState) -> np.ndarray:
        """Create car shape using pre-rotated template - only translate"""
        if vehicle._shape_template is None:
            self._precompute_car_template(vehicle)
        
        # Copy template and translate to current position
        points = vehicle._shape_template.copy()
        points[:, 0] += vehicle.x
        points[:, 1] += vehicle.y
        
        return points
    
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
        
        # Initialize vehicle visualizations with more diverse colors
        self.colors = [
            '#00aaff',  # Blue
            '#ff4444',  # Red  
            '#44ff44',  # Green
            '#ffaa00',  # Orange
            '#aa44ff',  # Purple
            '#ff8844',  # Orange-red
            '#ff44aa',  # Pink
            '#44ffff',  # Cyan
            '#88ff44',  # Lime
            '#ff4488',  # Rose
            '#44aaff',  # Light blue
            '#ffaa44',  # Gold
            '#aa88ff',  # Lavender
            '#44ff88',  # Mint
            '#ff8888',  # Light red
            '#88aaff'   # Periwinkle
        ] * 2  # Duplicate for more vehicles
        
        for i, vehicle in enumerate(self.vehicles):
            self._setup_vehicle_visualization(vehicle, i)
        
        if self.vehicles:
            self.ax_speed.legend(loc='upper right')
        
        # Metrics text
        self.metrics_text = self.ax_main.text(0.02, 0.98, '', transform=self.ax_main.transAxes,
                                            verticalalignment='top', fontsize=11, fontweight='bold',
                                            color='white', animated=True,
                                            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d44',
                                                    edgecolor='#00ff88', linewidth=2, alpha=0.95))
    
    def _setup_vehicle_visualization(self, vehicle: VehicleState, color_index: int = None):
        """Setup visualization elements for a single vehicle"""
        if color_index is None:
            # Better color assignment: base on vehicle ID and approach angle for variety
            base_color_index = (vehicle.id - 1) % len(self.colors)
            # Add some variation based on approach angle to spread colors better
            angle_offset = {0: 0, 90: 2, 180: 4, 270: 6}.get(vehicle.approach_angle, 0)
            color_index = (base_color_index + angle_offset) % len(self.colors)
        
        color = self.colors[color_index % len(self.colors)]
        vehicle.color = color
        
        # Car shape
        car_points = self.create_car_shape(vehicle)
        self.vehicle_shapes[vehicle.id] = Polygon(car_points, facecolor=color, edgecolor='white',
                                                linewidth=2, alpha=0.95, animated=True)
        self.ax_main.add_patch(self.vehicle_shapes[vehicle.id])
        
        # Trail
        self.vehicle_trails[vehicle.id], = self.ax_main.plot([], [], color=color, alpha=0.6, linewidth=2, animated=True)
        
        # Speed line
        self.speed_lines[vehicle.id], = self.ax_speed.plot([], [], color=color, linewidth=2, label=f'V{vehicle.id}', animated=True)
        
        # Predicted path
        self.predicted_paths[vehicle.id], = self.ax_main.plot([], [], color=color, linestyle=':', alpha=0.5, linewidth=1, animated=True)
    
    def animate(self, frame):
        """Highly optimized animation update with aggressive performance tuning"""
        if not self.plot_initialized:
            return []
        
        # More aggressive frame skipping for smoother UI
        self._skip_frames += 1
        if self._skip_frames < self._frame_skip_count:
            return []
        self._skip_frames = 0
        
        # Update warning cooldown
        self._warn_cooldown = max(0.0, self._warn_cooldown - self.dt)
        
        if frame < 1000:
            # Run simulation step
            self.simulate_step()
            
            # Batch update all visualizations efficiently
            updated_artists = []
            
            # Pre-calculate common values once
            current_time = frame * self.dt
            
            for vehicle in self.vehicles:
                if vehicle.id not in self.vehicle_shapes:
                    continue
                    
                # Update car shape (most expensive operation)
                car_points = self.create_car_shape(vehicle)
                self.vehicle_shapes[vehicle.id].set_xy(car_points)
                updated_artists.append(self.vehicle_shapes[vehicle.id])
                
                # Update trail with reduced history for performance
                hist_len = len(self.history_x[vehicle.id])
                if hist_len > 1:
                    # Use fewer trail points for better performance
                    trail_points = min(10, hist_len)  # Reduced from 25 to 10
                    trail_data_x = list(self.history_x[vehicle.id])[-trail_points:]
                    trail_data_y = list(self.history_y[vehicle.id])[-trail_points:]
                    self.vehicle_trails[vehicle.id].set_data(trail_data_x, trail_data_y)
                    updated_artists.append(self.vehicle_trails[vehicle.id])
                
                # Update speed plot less frequently for better performance
                if frame % 5 == 0:  # Update every 5th frame instead of 2nd
                    times = [i * self.dt for i in range(hist_len)]
                    if times and self.history_v[vehicle.id]:  # Ensure data exists
                        speed_data = list(self.history_v[vehicle.id])
                        self.speed_lines[vehicle.id].set_data(times, speed_data)
                        # Ensure speed plot stays visible
                        if vehicle.id not in getattr(self, '_speed_plot_updated', set()):
                            self.speed_lines[vehicle.id].set_visible(True)
                            if not hasattr(self, '_speed_plot_updated'):
                                self._speed_plot_updated = set()
                            self._speed_plot_updated.add(vehicle.id)
                
                # Update predicted path with minimal GPU transfers
                if frame % 10 == 0:  # Update every 10th frame for prediction (less frequent)
                    if hasattr(vehicle, '_gpu_trajectory') and vehicle._gpu_trajectory is not None:
                        with torch.no_grad():
                            # Even fewer prediction points for performance
                            pred_subset = vehicle._gpu_trajectory[:4].cpu().numpy()  # Reduced from 6 to 4
                            self.predicted_paths[vehicle.id].set_data(pred_subset[:, 0], pred_subset[:, 1])
                    elif hasattr(vehicle, 'planned_trajectory') and vehicle.planned_trajectory is not None:
                        pred_x = vehicle.planned_trajectory[:4, 0]  # Reduced points
                        pred_y = vehicle.planned_trajectory[:4, 1]
                        self.predicted_paths[vehicle.id].set_data(pred_x, pred_y)
            
            # Update metrics less frequently
            if frame % 5 == 0:  # Update every 5th frame
                self._update_metrics_display(frame)
                updated_artists.append(self.metrics_text)
            
            # Auto-extend speed plot x-axis more frequently
            if frame % 5 == 0 and self.vehicles:  # Every 5th frame
                max_time = max(len(self.history_v[v.id]) * self.dt for v in self.vehicles if self.history_v[v.id])
                if max_time > self.ax_speed.get_xlim()[1]:
                    self.ax_speed.set_xlim(0, max_time + 5)
                
                # Also update y-axis to fit all speeds
                all_speeds = []
                for v in self.vehicles:
                    if self.history_v[v.id]:
                        all_speeds.extend(self.history_v[v.id])
                
                if all_speeds:
                    max_speed = max(all_speeds)
                    if max_speed > self.ax_speed.get_ylim()[1]:
                        self.ax_speed.set_ylim(0, max_speed + 2)
                
                # Force redraw of speed plot
                self.ax_speed.relim()
                self.ax_speed.autoscale_view(scalex=True, scaley=True)
        
        # Return all artists that need updating
        return (list(self.vehicle_shapes.values()) + 
                list(self.vehicle_trails.values()) + 
                list(self.speed_lines.values()) +
                list(self.predicted_paths.values()))
    
    def _update_metrics_display(self, frame):
        """Separated metrics update for performance"""
        current_speeds = [v.velocity for v in self.vehicles]
        if not current_speeds:
            return
            
        min_speed = min(current_speeds)
        max_speed = max(current_speeds)
        avg_speed = np.mean(current_speeds)
        
        # Minimum distance calculation (expensive, do less frequently)
        if len(self.vehicles) > 1:
            positions = np.array([[v.x, v.y] for v in self.vehicles])
            pos_diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distances = np.linalg.norm(pos_diff, axis=2)
            distances[distances == 0] = np.inf
            min_distance = np.min(distances)
        else:
            min_distance = float('inf')
        
        # Count vehicles in car-following mode
        following_count = 0
        for vehicle in self.vehicles:
            front_vehicle = self.find_vehicle_in_front(vehicle, max_distance=30.0)
            if front_vehicle is not None:
                following_count += 1
        
        # Vehicles in intersection
        in_intersection = sum(1 for v in self.vehicles
                            if abs(v.x) < self.intersection_size/2 + 5 and
                               abs(v.y) < self.intersection_size/2 + 5)
        
        time_elapsed = frame * self.dt
        efficiency = (avg_speed / np.mean([v.target_velocity for v in self.vehicles])) * 100
        
        metrics_str = f"TIME: {time_elapsed:.1f}s | VEHICLES: {len(self.vehicles)}\n"
        metrics_str += f"IN INTERSECTION: {in_intersection} | MIN DIST: {min_distance:.1f}m\n"
        metrics_str += f"FOLLOWING: {following_count} | SPEEDS: {avg_speed:.1f} m/s ({efficiency:.0f}%)\n"
        metrics_str += f"UNRESERVED: {self.metrics['unreserved_attempts']} | COMPLETED: {self.metrics['completed']}\n"
        
        # Show both collision types
        if self.visual_collision_detected:
            visual_pairs_str = ", ".join([f"V{p[0]}-V{p[1]}" for p in self.visual_collision_pairs])
            metrics_str += f"🔴 VISUAL COLLISION: {visual_pairs_str}\n"
        
        if self.collision_detected:
            pred_pairs_str = ", ".join([f"V{p[0]}-V{p[1]}" for p in self.collision_pairs])
            metrics_str += f"⚠️ PREDICTED COLLISION: {pred_pairs_str}\n"
        
        # Overall safety status
        if self.visual_collision_detected:
            metrics_str += f"SAFETY: ACTUAL COLLISION!"
        elif self.collision_detected:
            metrics_str += f"SAFETY: COLLISION PREDICTED"
        else:
            safety_status = 'SAFE' if min_distance > 8.0 else 'CLOSE' if min_distance > 5.0 else 'DANGER'
            metrics_str += f"SAFETY: {safety_status}"
        
        self.metrics_text.set_text(metrics_str)
    
    def finalize_setup(self):
        if not self.plot_initialized:
            self.setup_plot()
            self.plot_initialized = True
    
    def run_animation(self):
        self.finalize_setup()
        # Enable blitting for better performance now that artists are marked as animated
        ani = animation.FuncAnimation(self.fig, self.animate, frames=1000,
                                    interval=33, blit=True, repeat=False,  # Enabled blitting for better performance
                                    cache_frame_data=False)  # Reduce memory usage
        plt.tight_layout()
        plt.show(block=True)  # Block to keep window open
        return ani

    def cbf_clip_batch(self, u_nom: torch.Tensor) -> torch.Tensor:
        """Batched CBF safety filter - keeps everything on GPU"""
        B = len(self.vehicles)
        if B == 0:
            return u_nom
            
        veh = self.vehicles
        # tensors of per-veh params on device
        max_acc = torch.tensor([v.max_accel for v in veh], device=device)
        max_dec = torch.tensor([v.max_decel for v in veh], device=device)
        
        # Build velocity and position tensors
        vx, vy = [], []
        px, py = [], []
        ang = []
        for v in veh:
            px.append(v.x); py.append(v.y); ang.append(v.approach_angle)
            if v.approach_angle == 0:  vx.append(v.velocity);  vy.append(0.0)
            elif v.approach_angle == 180: vx.append(-v.velocity); vy.append(0.0)
            elif v.approach_angle == 90:  vx.append(0.0); vy.append(v.velocity)
            else: vx.append(0.0); vy.append(-v.velocity)
            
        px = torch.tensor(px, device=device); py = torch.tensor(py, device=device)
        vx = torch.tensor(vx, device=device); vy = torch.tensor(vy, device=device)
        ang = torch.tensor(ang, device=device)

        # For ego axis (x for N/S, y for E/W)
        use_x = (ang == 0) | (ang == 180)
        p = torch.where(use_x, px, py)        # [B]
        v = torch.where(use_x, vx, vy)        # [B]

        # pairwise separation along ego axis
        s = p[:, None] - p[None, :]           # [B,B]
        vrel = v[:, None] - v[None, :]        # [B,B]

        d_safe = 8.0  # Reduced from 12.0 to allow closer following
        alpha = 0.3   # Reduced from 0.5 for less aggressive safety
        dt = self.dt

        # one-step linearized bound (as in your scalar CBF)
        s_bar = s + vrel * dt
        A = s_bar * (dt * dt)
        h = s * s - (d_safe * d_safe)
        c = (s_bar * s_bar) - (1.0 - alpha) * (h) - alpha * (d_safe * d_safe)

        # bounds per pair; ignore self-pairs
        eye = torch.eye(B, device=device, dtype=torch.bool)
        A = A.masked_fill(eye, 0.0)
        c = c.masked_fill(eye, 0.0)

        # candidate bounds for u: when A>0, u ≥ -c/A; when A<0, u ≤ -c/A
        with torch.no_grad():
            u_min = -max_dec
            u_max =  max_acc
            valid_pos = A > 1e-6
            valid_neg = A < -1e-6
            bound_pos = (-c / A).masked_fill(~valid_pos, float('-inf'))  # lower bounds
            bound_neg = (-c / A).masked_fill(~valid_neg, float('inf'))   # upper bounds

            # per-vehicle bound is the max of lower-bounds and min of upper-bounds across j
            u_min = torch.max(u_min, torch.max(bound_pos, dim=1).values)
            u_max = torch.min(u_max, torch.min(bound_neg, dim=1).values)

            # project u_nom into [u_min, u_max] and ensure physical limits
            result = torch.minimum(torch.maximum(u_nom, u_min), u_max)
            # Clamp to absolute physical limits as final safety check
            result = torch.clamp(result, -6.0, 3.0)  # max_decel=6.0, max_accel=3.0
            return result


def create_scenario(device_override=None):
    """Create optimized scenario with priority-based coordination"""
    global device
    if device_override:
        device = device_override
    sim = IntersectionSimulation(max_vehicles=16)
    
    # Assign priorities and safe initial positions to prevent immediate collisions
    vehicles = [
        VehicleState(1, np.array([-80.0, 0.0]), 12.0, 20.0, 0, 0, 0.8),   # North approach - increased from 15.0
        VehicleState(2, np.array([-120.0, 0.0]), 11.0, 20.0, 0, 0, 0.6),  # North approach, further back - increased from 15.0
        VehicleState(3, np.array([-160.0, 0.0]), 10.0, 20.0, 0, 0, 0.4),  # North approach, even further back
        VehicleState(4, np.array([-200.0, 0.0]), 9.0, 20.0, 0, 0, 0.7),   # North approach, furthest back
        VehicleState(5, np.array([85.0, 0.0]), 13.0, 20.0, 180, 0, 0.7),  # South approach - increased from 15.0
        VehicleState(6, np.array([125.0, 0.0]), 12.0, 20.0, 180, 0, 0.5), # South approach, further back - increased from 15.0
        VehicleState(7, np.array([165.0, 0.0]), 11.0, 20.0, 180, 0, 0.3), # South approach, even further back
        VehicleState(8, np.array([205.0, 0.0]), 10.0, 20.0, 180, 0, 0.6), # South approach, furthest back
        VehicleState(9, np.array([0.0, -75.0]), 10.0, 20.0, 90, 0, 0.9),  # East approach - increased from 15.0
        VehicleState(10, np.array([0.0, -115.0]), 11.0, 20.0, 90, 0, 0.4), # East approach, further back - increased from 15.0
        VehicleState(11, np.array([0.0, -155.0]), 12.0, 20.0, 90, 0, 0.8), # East approach, even further back
        VehicleState(12, np.array([0.0, -195.0]), 9.0, 20.0, 90, 0, 0.5),  # East approach, furthest back
        VehicleState(13, np.array([0.0, 80.0]), 9.0, 20.0, 270, 0, 0.3),   # West approach - increased from 15.0
        VehicleState(14, np.array([0.0, 120.0]), 10.0, 20.0, 270, 0, 0.2), # West approach, furthest back - increased from 15.0
        VehicleState(15, np.array([0.0, 160.0]), 11.0, 20.0, 270, 0, 0.6), # West approach, even further back
        VehicleState(16, np.array([0.0, 200.0]), 8.0, 20.0, 270, 0, 0.9),  # West approach, furthest back
    ]
    
    for vehicle in vehicles:
        sim.add_vehicle(vehicle)
    
    return sim


if __name__ == "__main__":
    print("High-Performance Intersection Control")
    
    # Update device selection based on actual workload
    max_vehicles = 16  # From create_scenario - doubled
    horizon = 15
    device = pick_device(max_vehicles, horizon)
    print(f"Device selected: {device} (workload: {max_vehicles}×{horizon}={max_vehicles*horizon})")
    
    print("Features:")
    print("  - Predictive Collision Avoidance")
    print("  - Priority-Based Coordination")
    print("  - Batched GPU processing")
    print("  - Vectorized operations")
    print("  - Optimal data structures")
    print("  - Clean architecture")
    print("  - GPU-resident CBF safety filter")
    print("  - Branchless gradient clipping")
    print("  - Vectorized cost computation")
    print("\nStarting simulation...")
    
    try:
        sim = create_scenario(device_override=device)
        animation_object = sim.run_animation()
        
        # Keep the program running to display animation
        input("\nPress Enter to exit...")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        import traceback
        print(f"\nSimulation error: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
    finally:
        plt.close('all')
        print("Simulation complete!")