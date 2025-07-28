"""
Simple Intersection Test: Traffic Lights vs Reservation-Based Coordination
==========================================================================

Testing the hypothesis: Can autonomous vehicles eliminate traffic lights?

This implements a minimal but scientifically valid comparison using:
1. Microscopic traffic simulation (Pygame-based)
2. IDM car-following + simple lane-keeping
3. Two policies: Fixed-time signals vs Reservation management

Based on Dresner & Stone's Autonomous Intersection Management research.
"""

import numpy as np
import math
import random
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import statistics
import matplotlib.pyplot as plt

# Optional pygame import for visual mode
try:
    import pygame
except ImportError:
    pygame = None

# Set random seeds for reproducible results
random.seed(42)
np.random.seed(42)

# Configuration constants
SIMULATION_SPEED = 1.0  # Real-time multiplier
SPAWN_RATES = [300, 600, 900]  # vehicles/hour for testing
TRIAL_DURATION = 300  # seconds per trial
NUM_TRIALS = 10  # trials per condition
INTERSECTION_SIZE = 30  # meters

@dataclass
class SimulationMetrics:
    """Metrics collected during simulation"""
    total_delay: float = 0.0
    total_stops: int = 0
    max_queue_length: int = 0
    vehicles_processed: int = 0
    speed_time_integral: float = 0.0  # Time-weighted speed accumulator
    simulation_time: float = 0.0
    fuel_consumption: float = 0.0
    
    def average_delay(self) -> float:
        return self.total_delay / max(1, self.vehicles_processed)
    
    def stops_per_vehicle(self) -> float:
        return self.total_stops / max(1, self.vehicles_processed)
    
    def average_speed(self) -> float:
        return self.speed_time_integral / max(1e-9, self.simulation_time)

class IDMController:
    """Intelligent Driver Model for realistic car-following"""
    def __init__(self):
        # IDM parameters (typical values)
        self.v0 = 15.0  # desired speed (m/s) ~54 km/h
        self.T = 1.5    # safe time headway (s)
        self.a = 2.0    # max acceleration (m/s²)
        self.b = 3.0    # comfortable deceleration (m/s²)
        self.s0 = 2.0   # minimum spacing (m)
        self.delta = 4  # acceleration exponent
    
    def calculate_acceleration(self, speed: float, leader_speed: float, 
                             gap: float, desired_speed: float = None) -> float:
        """Calculate acceleration using IDM"""
        v = speed
        v0 = desired_speed or self.v0
        dv = v - leader_speed if leader_speed is not None else 0
        
        # Desired spacing
        s_star = self.s0 + max(0, v * self.T + (v * dv) / (2 * math.sqrt(self.a * self.b)))
        
        # IDM acceleration
        acceleration = self.a * (1 - (v / v0)**self.delta - (s_star / max(gap, 0.1))**2)
        
        return max(-self.b, min(self.a, acceleration))

class Vehicle:
    """Simple vehicle with IDM following and lane-keeping"""
    def __init__(self, x: float, y: float, heading: float, vehicle_id: int):
        self.id = vehicle_id
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = 0.0
        self.acceleration = 0.0
        self.length = 4.5  # meters
        
        # Navigation
        self.route = self._determine_route()
        self.target_lane = self._get_target_lane()
        
        # Metrics
        self.total_delay = 0.0
        self.stop_count = 0
        self.was_stopped = False
        self.spawn_time = 0.0
        self.fuel_used = 0.0
        
        # IDM controller
        self.idm = IDMController()
        
        # Reservation system
        self.reservation_request = None
        self.has_reservation = False
        self.reservation_time = None
    
    def _determine_route(self) -> str:
        """Determine turning movement based on spawn location"""
        # Force straight-through for now (avoiding turn conflicts in 3x3 reservation grid)
        return 'straight'
    
    def _get_target_lane(self) -> Tuple[float, float]:
        """Get target lane center coordinates"""
        # Simplified: assume 4-lane intersection (2 per direction)
        if abs(self.heading) < 0.1:  # Eastbound
            return self.x, 400 - 3.5  # Right lane
        elif abs(self.heading - math.pi) < 0.1:  # Westbound  
            return self.x, 400 + 3.5  # Right lane
        elif abs(self.heading - math.pi/2) < 0.1:  # Southbound
            return 400 + 3.5, self.y  # Right lane
        else:  # Northbound
            return 400 - 3.5, self.y  # Right lane
    
    def get_leader(self, vehicles: List['Vehicle']) -> Optional['Vehicle']:
        """Find the vehicle directly ahead"""
        min_distance = float('inf')
        leader = None
        
        for other in vehicles:
            if other.id == self.id:
                continue
                
            # Check if vehicle is ahead in same direction
            dx = other.x - self.x
            dy = other.y - self.y
            
            # Project onto heading direction
            along_track = dx * math.cos(self.heading) + dy * math.sin(self.heading)
            cross_track = abs(-dx * math.sin(self.heading) + dy * math.cos(self.heading))
            
            if along_track > 0 and cross_track < 7.0 and along_track < min_distance:
                min_distance = along_track
                leader = other
        
        return leader
    
    def distance_to_intersection(self) -> float:
        """Calculate distance to intersection center"""
        return math.sqrt((self.x - 400)**2 + (self.y - 400)**2)
    
    def update(self, dt: float, vehicles: List['Vehicle'], 
               intersection_policy: 'IntersectionPolicy'):
        """Update vehicle state"""
        # Find leader vehicle
        leader = self.get_leader(vehicles)
        
        # Calculate gap and relative speed
        if leader:
            gap = math.sqrt((leader.x - self.x)**2 + (leader.y - self.y)**2) - self.length
            leader_speed = leader.speed
        else:
            gap = 100.0  # No leader, assume clear road
            leader_speed = self.idm.v0
        
        # Check intersection permission
        can_proceed = intersection_policy.can_proceed(self)
        
        if not can_proceed and self.distance_to_intersection() < 50:
            # Stop for intersection
            target_speed = 0.0
            gap = min(gap, self.distance_to_intersection() - INTERSECTION_SIZE/2)
        else:
            target_speed = self.idm.v0
        
        # IDM acceleration
        self.acceleration = self.idm.calculate_acceleration(
            self.speed, leader_speed, gap, target_speed
        )
        
        # Update kinematics
        self.speed = max(0, self.speed + self.acceleration * dt)
        
        # Update position
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
        
        # Simple lane keeping (snap to center)
        target_x, target_y = self._get_target_lane()
        lateral_error_x = target_x - self.x
        lateral_error_y = target_y - self.y
        
        # Gentle correction toward lane center
        correction_strength = 0.5
        if abs(self.heading) < 0.2 or abs(self.heading - math.pi) < 0.2:  # Horizontal
            self.y += lateral_error_y * correction_strength * dt
        else:  # Vertical
            self.x += lateral_error_x * correction_strength * dt
        
        # Update metrics
        if self.speed < 0.5:
            if not self.was_stopped:
                self.stop_count += 1
                self.was_stopped = True
            self.total_delay += dt
        else:
            self.was_stopped = False
        
        # Fuel consumption (simplified model)
        if self.speed > 0:
            # Base consumption + acceleration penalty
            base_consumption = 0.1 * dt
            accel_penalty = max(0, self.acceleration) * 0.05 * dt
            self.fuel_used += base_consumption + accel_penalty
    
    def is_in_intersection(self) -> bool:
        """Check if vehicle is currently in intersection"""
        return (370 < self.x < 430 and 370 < self.y < 430)
    
    def has_exited(self) -> bool:
        """Check if vehicle has exited the simulation area"""
        return (self.x < -50 or self.x > 850 or self.y < -50 or self.y > 850)

class IntersectionPolicy:
    """Base class for intersection control policies"""
    def __init__(self):
        self.current_time = 0.0
    
    def update(self, dt: float):
        self.current_time += dt
    
    def can_proceed(self, vehicle: Vehicle) -> bool:
        raise NotImplementedError

class FixedTimeSignal(IntersectionPolicy):
    """Traditional fixed-time traffic signal"""
    def __init__(self, cycle_time: float = 60.0, green_time: float = 25.0):
        super().__init__()
        self.cycle_time = cycle_time
        self.green_time = green_time
        
    def get_current_phase(self) -> str:
        """Get current signal phase"""
        cycle_position = self.current_time % self.cycle_time
        if cycle_position < self.green_time:
            return 'NS_GREEN'  # North-South green
        elif cycle_position < self.green_time + 5:
            return 'NS_YELLOW'
        elif cycle_position < self.green_time + 5 + self.green_time:
            return 'EW_GREEN'  # East-West green
        else:
            return 'EW_YELLOW'
    
    def can_proceed(self, vehicle: Vehicle) -> bool:
        """Check if vehicle can proceed through intersection"""
        if vehicle.distance_to_intersection() > 50:
            return True  # Not near intersection
        
        phase = self.get_current_phase()
        
        # Determine vehicle direction
        if abs(vehicle.heading) < 0.2 or abs(vehicle.heading - math.pi) < 0.2:
            # East-West direction
            return phase in ['EW_GREEN']
        else:
            # North-South direction
            return phase in ['NS_GREEN']

class ReservationManager(IntersectionPolicy):
    """Reservation-based intersection management (Dresner & Stone approach)"""
    def __init__(self):
        super().__init__()
        self.reservations = []  # [(vehicle_id, start_time, end_time, path)]
        self.grid_size = 2.0  # meters per grid cell
        self.intersection_bounds = (370, 430, 370, 430)  # x_min, x_max, y_min, y_max
        
    def get_space_time_path(self, vehicle: Vehicle) -> List[Tuple[int, int, float]]:
        """Calculate space-time path through intersection"""
        path = []
        
        # Simplified: assume vehicle follows straight line through intersection
        dist_to_intersection = vehicle.distance_to_intersection()
        time_to_intersection = max(0.1, dist_to_intersection / max(0.1, vehicle.speed))
        
        # Calculate time in intersection
        intersection_crossing_time = INTERSECTION_SIZE / max(5.0, vehicle.speed)
        
        # Discretize path into grid cells
        entry_time = self.current_time + time_to_intersection
        exit_time = entry_time + intersection_crossing_time
        
        # Simple grid occupation (center path)
        grid_x = int((400 - self.intersection_bounds[0]) / self.grid_size)
        grid_y = int((400 - self.intersection_bounds[2]) / self.grid_size)
        
        # Reserve multiple cells for safety
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                path.append((grid_x + dx, grid_y + dy, entry_time, exit_time))
        
        return path
    
    def check_conflict(self, new_path: List[Tuple[int, int, float, float]]) -> bool:
        """Check if new path conflicts with existing reservations"""
        for vehicle_id, start_time, end_time, existing_path in self.reservations:
            for new_cell in new_path:
                nx, ny, n_start, n_end = new_cell
                for ex, ey, e_start, e_end in existing_path:
                    # Space overlap
                    if nx == ex and ny == ey:
                        # Time overlap
                        if not (n_end <= e_start or n_start >= e_end):
                            return True
        return False
    
    def request_reservation(self, vehicle: Vehicle) -> bool:
        """Request intersection reservation"""
        if vehicle.has_reservation:
            return True
        
        path = self.get_space_time_path(vehicle)
        
        if not self.check_conflict(path):
            # Grant reservation
            entry_time = min(p[2] for p in path)
            exit_time = max(p[3] for p in path)
            
            self.reservations.append((vehicle.id, entry_time, exit_time, path))
            vehicle.has_reservation = True
            vehicle.reservation_time = entry_time
            return True
        
        return False
    
    def cleanup_expired_reservations(self):
        """Remove expired reservations"""
        self.reservations = [
            (vid, start, end, path) for vid, start, end, path in self.reservations
            if end > self.current_time
        ]
    
    def can_proceed(self, vehicle: Vehicle) -> bool:
        """Check if vehicle can proceed through intersection"""
        if vehicle.distance_to_intersection() > 50:
            return True
        
        # Clean up old reservations
        self.cleanup_expired_reservations()
        
        # Try to get reservation if needed
        if not vehicle.has_reservation:
            if vehicle.distance_to_intersection() < 40:
                self.request_reservation(vehicle)
        
        # Check if vehicle has valid reservation for current time
        if vehicle.has_reservation:
            for vid, start_time, end_time, _ in self.reservations:
                if vid == vehicle.id:
                    return start_time <= self.current_time <= end_time
        
        return False

class IntersectionSimulation:
    """Main simulation controller"""
    def __init__(self, policy: IntersectionPolicy, spawn_rate: float, visual: bool = False, arrival_times: Optional[List[float]] = None):
        self.policy = policy
        self.spawn_rate = spawn_rate  # vehicles per hour
        self.vehicles = []
        self.next_vehicle_id = 1
        self.simulation_time = 0.0
        self.metrics = SimulationMetrics()
        self.visual = visual
        
        # Pre-generated arrival times for reproducible experiments
        self.arrival_times = arrival_times or []
        self.next_arrival_index = 0
        
        # Spawn points (entry roads) - moved further out to avoid instant exit
        self.spawn_points = [
            (50, 397, 0),           # West entry, heading East
            (750, 403, math.pi),    # East entry, heading West  
            (397, 50, math.pi/2),   # North entry, heading South
            (403, 750, -math.pi/2), # South entry, heading North
        ]
        
        # Initialize pygame for visualization
        if self.visual and pygame:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 800))
            pygame.display.set_caption(f"Intersection Test - {type(policy).__name__}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        elif self.visual and not pygame:
            print("⚠️  pygame not available - visual mode disabled")
            self.visual = False
    
    def spawn_vehicle(self):
        """Spawn a new vehicle at random entry point"""
        x, y, heading = random.choice(self.spawn_points)
        
        # Add some variation
        if abs(heading) < 0.1 or abs(heading - math.pi) < 0.1:  # Horizontal
            y += random.uniform(-2, 2)
        else:  # Vertical
            x += random.uniform(-2, 2)
        
        vehicle = Vehicle(x, y, heading, self.next_vehicle_id)
        vehicle.spawn_time = self.simulation_time
        self.vehicles.append(vehicle)
        self.next_vehicle_id += 1
    
    def should_spawn_vehicle(self) -> bool:
        """Determine if a new vehicle should be spawned"""
        # Use pre-generated arrival times if available
        if self.arrival_times:
            if (self.next_arrival_index < len(self.arrival_times) and 
                self.simulation_time >= self.arrival_times[self.next_arrival_index]):
                self.next_arrival_index += 1
                return True
            return False
        else:
            # Fallback to Poisson process for visual demos
            spawn_rate_per_second = self.spawn_rate / 3600.0
            return random.random() < spawn_rate_per_second
    
    def update(self, dt: float):
        """Update simulation state"""
        self.simulation_time += dt
        
        # Update intersection policy
        self.policy.update(dt)
        
        # Spawn new vehicles
        if self.should_spawn_vehicle():
            self.spawn_vehicle()
        
        # Update existing vehicles
        for vehicle in self.vehicles[:]:  # Copy list to allow removal
            vehicle.update(dt, self.vehicles, self.policy)
            
            # Remove vehicles that have exited
            if vehicle.has_exited():
                # Record metrics
                travel_time = self.simulation_time - vehicle.spawn_time
                self.metrics.total_delay += vehicle.total_delay
                self.metrics.total_stops += vehicle.stop_count
                self.metrics.vehicles_processed += 1
                self.metrics.fuel_consumption += vehicle.fuel_used
                
                self.vehicles.remove(vehicle)
        
        # Update queue length metric
        queue_count = sum(1 for v in self.vehicles 
                         if v.distance_to_intersection() < 100 and v.speed < 1.0)
        self.metrics.max_queue_length = max(self.metrics.max_queue_length, queue_count)
        
        # Update time-weighted average speed
        if self.vehicles:
            current_avg_speed = statistics.mean(v.speed for v in self.vehicles)
            self.metrics.speed_time_integral += current_avg_speed * dt
        
        self.metrics.simulation_time += dt
        
        # Handle pygame events and draw if visual mode
        if self.visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            self.draw()
    
    def draw(self):
        """Draw the simulation"""
        if not self.visual:
            return
            
        # Clear screen
        self.screen.fill((20, 20, 20))
        
        # Draw roads
        road_color = (60, 60, 60)
        # Horizontal roads
        pygame.draw.rect(self.screen, road_color, (0, 370, 800, 60))  # Main horizontal road
        # Vertical roads  
        pygame.draw.rect(self.screen, road_color, (370, 0, 60, 800))  # Main vertical road
        
        # Draw intersection
        intersection_color = (80, 80, 80)
        pygame.draw.rect(self.screen, intersection_color, (370, 370, 60, 60))
        
        # Draw lane lines (dashed)
        lane_color = (120, 120, 120)
        # Horizontal center line
        for x in range(0, 800, 20):
            if x < 370 or x > 430:  # Skip intersection area
                pygame.draw.rect(self.screen, lane_color, (x, 398, 10, 2))
        # Vertical center line
        for y in range(0, 800, 20):
            if y < 370 or y > 430:  # Skip intersection area
                pygame.draw.rect(self.screen, lane_color, (398, y, 2, 10))
        
        # Draw traffic light state (if applicable)
        if isinstance(self.policy, FixedTimeSignal):
            phase = self.policy.get_current_phase()
            light_color = (0, 255, 0) if 'GREEN' in phase else (255, 255, 0) if 'YELLOW' in phase else (255, 0, 0)
            
            # Draw simple traffic light
            pygame.draw.circle(self.screen, light_color, (350, 350), 8)
            
            # Show which direction has green
            phase_text = self.font.render(f"Phase: {phase}", True, (255, 255, 255))
            self.screen.blit(phase_text, (10, 10))
        else:
            # Show reservation count
            res_count = len(self.policy.reservations) if hasattr(self.policy, 'reservations') else 0
            res_text = self.font.render(f"Active Reservations: {res_count}", True, (255, 255, 255))
            self.screen.blit(res_text, (10, 10))
        
        # Draw vehicles
        for vehicle in self.vehicles:
            self.draw_vehicle(vehicle)
        
        # Draw metrics
        metrics_y = 40
        metrics = [
            f"Time: {self.simulation_time:.1f}s",
            f"Vehicles: {len(self.vehicles)}",
            f"Processed: {self.metrics.vehicles_processed}",
                         f"Avg Speed: {self.metrics.average_speed():.1f} m/s",
            f"Spawn Rate: {self.spawn_rate} veh/h"
        ]
        
        for i, metric in enumerate(metrics):
            text = self.font.render(metric, True, (255, 255, 255))
            self.screen.blit(text, (10, metrics_y + i * 25))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
    
    def draw_vehicle(self, vehicle: Vehicle):
        """Draw a single vehicle"""
        if not self.visual:
            return
            
        # Vehicle color based on state
        if vehicle.speed < 0.5:
            color = (255, 100, 100)  # Red if stopped
        elif vehicle.has_reservation:
            color = (100, 255, 100)  # Green if has reservation
        else:
            color = (100, 150, 255)  # Blue otherwise
        
        # Vehicle dimensions
        length, width = 8, 4
        
        # Calculate vehicle corners
        cos_h = math.cos(vehicle.heading)
        sin_h = math.sin(vehicle.heading)
        
        corners = [
            (-length/2, -width/2), (length/2, -width/2),
            (length/2, width/2), (-length/2, width/2)
        ]
        
        rotated_corners = []
        for corner_x, corner_y in corners:
            rotated_x = corner_x * cos_h - corner_y * sin_h + vehicle.x
            rotated_y = corner_x * sin_h + corner_y * cos_h + vehicle.y
            rotated_corners.append((rotated_x, rotated_y))
        
        # Draw vehicle
        pygame.draw.polygon(self.screen, color, rotated_corners)
        
        # Draw direction indicator (small white dot at front)
        front_x = vehicle.x + cos_h * length/2
        front_y = vehicle.y + sin_h * length/2
        pygame.draw.circle(self.screen, (255, 255, 255), (int(front_x), int(front_y)), 2)
    
    def get_current_metrics(self) -> SimulationMetrics:
        """Get current simulation metrics"""
        return self.metrics

def run_visual_demo(policy_class, spawn_rate: int = 600, duration: int = 60):
    """Run a visual demonstration of the intersection policy"""
    if not pygame:
        print("⚠️  pygame not available - cannot run visual demo")
        return
        
    print(f"\n🎬 Visual Demo: {policy_class.__name__}")
    print(f"Spawn rate: {spawn_rate} vehicles/hour")
    print("Press ESC or close window to stop")
    print("=" * 50)
    
    # Create policy and simulation
    if policy_class == FixedTimeSignal:
        policy = policy_class()
    else:
        policy = policy_class()
    
    sim = IntersectionSimulation(policy, spawn_rate, visual=True)
    
    # Run simulation
    dt = 1/30  # 30 FPS for smooth visuals
    max_steps = int(duration / dt)
    
    for step in range(max_steps):
        sim.update(dt)
        
        # Check if pygame window was closed
        if sim.visual and not pygame.get_init():
            break
    
    if pygame.get_init():
        pygame.quit()
    
    # Count remaining vehicles' metrics (vehicles that didn't exit)
    for vehicle in sim.vehicles:
        sim.metrics.total_delay += vehicle.total_delay
        sim.metrics.total_stops += vehicle.stop_count
        sim.metrics.vehicles_processed += 1
        sim.metrics.fuel_consumption += vehicle.fuel_used
    
    print(f"\n📊 Demo Results:")
    metrics = sim.get_current_metrics()
    print(f"  Vehicles processed: {metrics.vehicles_processed}")
    print(f"  Average delay: {metrics.average_delay():.1f}s")
    print(f"  Stops per vehicle: {metrics.stops_per_vehicle():.2f}")
    print(f"  Max queue length: {metrics.max_queue_length}")
    print(f"  Note: Includes vehicles still in network at end")

def generate_arrival_times(spawn_rate: float, duration: float, seed: int = None) -> List[float]:
    """Generate deterministic arrival times for reproducible experiments"""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate inter-arrival times using exponential distribution
    avg_interarrival = 3600.0 / spawn_rate  # seconds between arrivals
    
    arrivals = []
    current_time = 0.0
    
    while current_time < duration:
        # Exponential inter-arrival time
        inter_arrival = np.random.exponential(avg_interarrival)
        current_time += inter_arrival
        if current_time < duration:
            arrivals.append(current_time)
    
    return arrivals

def run_experiment_batch(policy_name: str, policy_class, spawn_rates: List[int], 
                        num_trials: int = NUM_TRIALS) -> Dict:
    """Run a batch of experiments for one policy"""
    print(f"\n🧪 Testing {policy_name}")
    print("=" * 50)
    print(f"Trials per condition: {num_trials}")
    print(f"Duration per trial: {TRIAL_DURATION}s")
    
    all_results = {}
    
    for spawn_rate in spawn_rates:
        print(f"\n📊 Spawn rate: {spawn_rate} vehicles/hour")
        trial_results = []
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}...", end="", flush=True)
            
            # Generate deterministic arrival times for this trial
            arrival_times = generate_arrival_times(spawn_rate, TRIAL_DURATION, seed=1000 + trial)
            
            # Create policy and simulation
            if policy_class == FixedTimeSignal:
                policy = policy_class()
            else:
                policy = policy_class()
            
            sim = IntersectionSimulation(policy, spawn_rate, arrival_times=arrival_times)
            
            # Run simulation
            dt = 0.1  # 100ms time steps
            steps = int(TRIAL_DURATION / dt)
            
            for step in range(steps):
                sim.update(dt)
            
            # Count remaining vehicles' metrics (vehicles that didn't exit)
            for vehicle in sim.vehicles:
                sim.metrics.total_delay += vehicle.total_delay
                sim.metrics.total_stops += vehicle.stop_count
                sim.metrics.vehicles_processed += 1
                sim.metrics.fuel_consumption += vehicle.fuel_used
            
            # Collect results
            metrics = sim.get_current_metrics()
            trial_results.append({
                'avg_delay': metrics.average_delay(),
                'stops_per_vehicle': metrics.stops_per_vehicle(),
                'max_queue': metrics.max_queue_length,
                'throughput': metrics.vehicles_processed / (TRIAL_DURATION / 3600),  # vehicles/hour
                'avg_speed': metrics.average_speed(),
                'fuel_per_vehicle': metrics.fuel_consumption / max(1, metrics.vehicles_processed)
            })
            
            print(" ✓")
        
        all_results[spawn_rate] = trial_results
        
        # Print summary for this spawn rate
        avg_delay = statistics.mean(r['avg_delay'] for r in trial_results)
        avg_throughput = statistics.mean(r['throughput'] for r in trial_results)
        avg_stops = statistics.mean(r['stops_per_vehicle'] for r in trial_results)
        
        print(f"    Avg delay: {avg_delay:.1f}s")
        print(f"    Avg throughput: {avg_throughput:.0f} veh/hr")
        print(f"    Avg stops per vehicle: {avg_stops:.2f}")
    
    return all_results

def statistical_comparison(results_baseline: Dict, results_test: Dict, stats_module=None):
    """Perform statistical comparison between policies"""
    print("\n📈 STATISTICAL ANALYSIS")
    print("=" * 50)
    
    if stats_module is None:
        print("⚠️  Statistical tests skipped (scipy not available)")
        print("    Results shown without p-values")
        use_stats = False
    else:
        use_stats = True
    
    for spawn_rate in SPAWN_RATES:
        print(f"\n🚗 Spawn Rate: {spawn_rate} vehicles/hour")
        print("-" * 30)
        
        baseline_delays = [r['avg_delay'] for r in results_baseline[spawn_rate]]
        test_delays = [r['avg_delay'] for r in results_test[spawn_rate]]
        
        baseline_throughput = [r['throughput'] for r in results_baseline[spawn_rate]]
        test_throughput = [r['throughput'] for r in results_test[spawn_rate]]
        
        # Statistical tests (if available)
        if use_stats:
            delay_stat, delay_p = stats_module.ttest_ind(baseline_delays, test_delays)
            throughput_stat, throughput_p = stats_module.ttest_ind(baseline_throughput, test_throughput)
            
            # Effect sizes (Cohen's d)
            def cohens_d(x1, x2):
                n1, n2 = len(x1), len(x2)
                s1, s2 = statistics.stdev(x1), statistics.stdev(x2)
                pooled_std = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
                if pooled_std < 1e-9:  # Avoid division by zero
                    return 0.0
                return (statistics.mean(x1) - statistics.mean(x2)) / pooled_std
            
            delay_effect = cohens_d(baseline_delays, test_delays)
            throughput_effect = cohens_d(test_throughput, baseline_throughput)
        else:
            delay_p = throughput_p = None
            delay_effect = throughput_effect = None
        
        # Results
        delay_improvement = (statistics.mean(baseline_delays) - statistics.mean(test_delays)) / statistics.mean(baseline_delays) * 100
        throughput_improvement = (statistics.mean(test_throughput) - statistics.mean(baseline_throughput)) / statistics.mean(baseline_throughput) * 100
        
        print(f"Average Delay:")
        print(f"  Baseline: {statistics.mean(baseline_delays):.1f}s ± {statistics.stdev(baseline_delays):.1f}")
        print(f"  Reservation: {statistics.mean(test_delays):.1f}s ± {statistics.stdev(test_delays):.1f}")
        if use_stats:
            print(f"  Improvement: {delay_improvement:+.1f}% (p={delay_p:.3f}, d={delay_effect:.2f})")
        else:
            print(f"  Improvement: {delay_improvement:+.1f}%")
        
        print(f"Throughput:")
        print(f"  Baseline: {statistics.mean(baseline_throughput):.0f} veh/hr ± {statistics.stdev(baseline_throughput):.0f}")
        print(f"  Reservation: {statistics.mean(test_throughput):.0f} veh/hr ± {statistics.stdev(test_throughput):.0f}")
        if use_stats:
            print(f"  Improvement: {throughput_improvement:+.1f}% (p={throughput_p:.3f}, d={throughput_effect:.2f})")
        else:
            print(f"  Improvement: {throughput_improvement:+.1f}%")
        
        # Significance assessment
        if use_stats:
            if delay_p < 0.05 and delay_improvement > 0:
                print(f"  ✅ SIGNIFICANT delay reduction")
            elif delay_p >= 0.05:
                print(f"  ⚠️  No significant difference in delay")
            else:
                print(f"  ❌ Delay increased significantly")
            
            if throughput_p < 0.05 and throughput_improvement > 0:
                print(f"  ✅ SIGNIFICANT throughput increase")
            elif throughput_p >= 0.05:
                print(f"  ⚠️  No significant difference in throughput")
            else:
                print(f"  ❌ Throughput decreased significantly")
        else:
            # Simple thresholds without p-values
            if delay_improvement > 15:
                print(f"  ✅ LARGE delay reduction")
            elif delay_improvement > 5:
                print(f"  ✅ MODERATE delay reduction")
            elif delay_improvement > 0:
                print(f"  ⚠️  Small delay reduction")
            else:
                print(f"  ❌ Delay increased")
            
            if throughput_improvement > 10:
                print(f"  ✅ LARGE throughput increase")
            elif throughput_improvement > 3:
                print(f"  ✅ MODERATE throughput increase")
            elif throughput_improvement > 0:
                print(f"  ⚠️  Small throughput increase")
            else:
                print(f"  ❌ Throughput decreased")

def plot_results(results_baseline: Dict, results_test: Dict):
    """Plot comparison results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    spawn_rates = list(results_baseline.keys())
    
    # Extract metrics
    baseline_delays = [[r['avg_delay'] for r in results_baseline[sr]] for sr in spawn_rates]
    test_delays = [[r['avg_delay'] for r in results_test[sr]] for sr in spawn_rates]
    
    baseline_throughput = [[r['throughput'] for r in results_baseline[sr]] for sr in spawn_rates]
    test_throughput = [[r['throughput'] for r in results_test[sr]] for sr in spawn_rates]
    
    baseline_stops = [[r['stops_per_vehicle'] for r in results_baseline[sr]] for sr in spawn_rates]
    test_stops = [[r['stops_per_vehicle'] for r in results_test[sr]] for sr in spawn_rates]
    
    baseline_fuel = [[r['fuel_per_vehicle'] for r in results_baseline[sr]] for sr in spawn_rates]
    test_fuel = [[r['fuel_per_vehicle'] for r in results_test[sr]] for sr in spawn_rates]
    
    # Box plots
    positions_baseline = [i - 0.2 for i in range(len(spawn_rates))]
    positions_test = [i + 0.2 for i in range(len(spawn_rates))]
    
    # Average delay
    ax1.boxplot(baseline_delays, positions=positions_baseline, widths=0.3, 
                patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    ax1.boxplot(test_delays, positions=positions_test, widths=0.3,
                patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax1.set_title('Average Delay per Vehicle')
    ax1.set_ylabel('Delay (seconds)')
    ax1.set_xticks(range(len(spawn_rates)))
    ax1.set_xticklabels([f'{sr} veh/hr' for sr in spawn_rates])
    # Remove individual legends and add a shared one
    ax1.text(0.02, 0.98, '■ Traffic Lights', transform=ax1.transAxes, 
             verticalalignment='top', color='lightcoral', fontsize=10)
    ax1.text(0.02, 0.90, '■ Reservation', transform=ax1.transAxes, 
             verticalalignment='top', color='lightblue', fontsize=10)
    
    # Throughput
    ax2.boxplot(baseline_throughput, positions=positions_baseline, widths=0.3,
                patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    ax2.boxplot(test_throughput, positions=positions_test, widths=0.3,
                patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax2.set_title('Intersection Throughput')
    ax2.set_ylabel('Vehicles/hour')
    ax2.set_xticks(range(len(spawn_rates)))
    ax2.set_xticklabels([f'{sr} veh/hr' for sr in spawn_rates])
    
    # Stops per vehicle
    ax3.boxplot(baseline_stops, positions=positions_baseline, widths=0.3,
                patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    ax3.boxplot(test_stops, positions=positions_test, widths=0.3,
                patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax3.set_title('Stops per Vehicle')
    ax3.set_ylabel('Number of stops')
    ax3.set_xticks(range(len(spawn_rates)))
    ax3.set_xticklabels([f'{sr} veh/hr' for sr in spawn_rates])
    
    # Fuel consumption per vehicle
    ax4.boxplot(baseline_fuel, positions=positions_baseline, widths=0.3,
                patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    ax4.boxplot(test_fuel, positions=positions_test, widths=0.3,
                patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax4.set_title('Fuel Consumption per Vehicle')
    ax4.set_ylabel('Fuel units')
    ax4.set_xticks(range(len(spawn_rates)))
    ax4.set_xticklabels([f'{sr} veh/hr' for sr in spawn_rates])
    
    plt.suptitle('Traffic Lights vs Reservation-Based Intersection Management', fontsize=16)
    plt.tight_layout()
    plt.savefig('intersection_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # Prevent memory leak

def main():
    """Run the complete experiment"""
    print("🚦 Autonomous Intersection Management Experiment")
    print("=" * 60)
    print("Testing hypothesis: Can AVs eliminate traffic lights?")
    print("Method: Rule-based comparison (no neural networks needed)")
    print("=" * 60)
    
    # Ask user what they want to see
    choice = input("""
Choose what to run:
1. Visual demo: Traffic Lights (60 seconds)
2. Visual demo: Reservation System (60 seconds)  
3. Visual comparison: Both systems side-by-side
4. Full statistical experiment (no visuals, ~7 minutes)
5. Quick experiment (3 trials each, ~2 minutes)

Enter choice (1-5): """).strip()
    
    if choice in ['1', '2', '3']:
        # Visual demos
        if choice == '1':
            run_visual_demo(FixedTimeSignal, 600, 60)
        elif choice == '2':
            run_visual_demo(ReservationManager, 600, 60)
        elif choice == '3':
            print("\n🎬 Running side-by-side comparison...")
            print("First: Traffic Light System")
            run_visual_demo(FixedTimeSignal, 600, 45)
            
            print("\nSecond: Reservation System")
            run_visual_demo(ReservationManager, 600, 45)
        
        return
    
    # Statistical experiments
    num_trials = 3 if choice == '5' else NUM_TRIALS
    
    # Optional scipy for statistical analysis
    try:
        from scipy import stats
    except ImportError:
        print("⚠️  scipy not available - statistical tests will be skipped")
        stats = None
    
    start_time = time.time()
    
    # Run experiments
    print(f"\n🔬 Experimental design:")
    print(f"  • Spawn rates: {SPAWN_RATES} vehicles/hour")
    print(f"  • Trials per condition: {num_trials}")
    print(f"  • Trial duration: {TRIAL_DURATION} seconds")
    print(f"  • Total simulations: {len(SPAWN_RATES) * num_trials * 2}")
    print(f"  • Estimated time: ~{len(SPAWN_RATES) * num_trials * 2 * 0.5:.0f} seconds")
    print(f"  • Using paired arrival streams for fair comparison")
    
    # Baseline: Fixed-time traffic signals
    baseline_results = run_experiment_batch(
        "Fixed-Time Traffic Signals (Baseline)", 
        FixedTimeSignal, 
        SPAWN_RATES, 
        num_trials
    )
    
    # Test: Reservation-based management
    test_results = run_experiment_batch(
        "Reservation-Based Management (AIM)", 
        ReservationManager, 
        SPAWN_RATES, 
        num_trials
    )
    
    # Statistical analysis
    statistical_comparison(baseline_results, test_results, stats)
    
    # Generate plots
    print(f"\n📊 Generating visualization...")
    plot_results(baseline_results, test_results)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n🏁 EXPERIMENT COMPLETED")
    print("=" * 60)
    print(f"Total runtime: {total_time:.1f} seconds")
    print(f"Results saved: intersection_comparison.png")
    
    # Conclusion
    print(f"\n🎯 HYPOTHESIS ASSESSMENT:")
    
    # Calculate overall improvements
    all_delay_improvements = []
    all_throughput_improvements = []
    
    for spawn_rate in SPAWN_RATES:
        baseline_delay = statistics.mean(r['avg_delay'] for r in baseline_results[spawn_rate])
        test_delay = statistics.mean(r['avg_delay'] for r in test_results[spawn_rate])
        delay_imp = (baseline_delay - test_delay) / baseline_delay * 100
        all_delay_improvements.append(delay_imp)
        
        baseline_throughput = statistics.mean(r['throughput'] for r in baseline_results[spawn_rate])
        test_throughput = statistics.mean(r['throughput'] for r in test_results[spawn_rate])
        throughput_imp = (test_throughput - baseline_throughput) / baseline_throughput * 100
        all_throughput_improvements.append(throughput_imp)
    
    avg_delay_improvement = statistics.mean(all_delay_improvements)
    avg_throughput_improvement = statistics.mean(all_throughput_improvements)
    
    if avg_delay_improvement > 20 and avg_throughput_improvement > 10:
        print("  ✅ HYPOTHESIS SUPPORTED: Reservation management significantly")
        print("     outperforms traffic lights in delay and throughput")
        print("  🚀 AVs can likely eliminate traditional traffic signals!")
    elif avg_delay_improvement > 10 or avg_throughput_improvement > 5:
        print("  ⚠️  HYPOTHESIS PARTIALLY SUPPORTED: Some improvements observed")
        print("     but benefits may depend on traffic conditions")
    else:
        print("  ❌ HYPOTHESIS NOT SUPPORTED: No clear advantage for reservation")
        print("     management under these conditions")
    
    print(f"\n📋 Key findings:")
    print(f"  • Average delay improvement: {avg_delay_improvement:+.1f}%")
    print(f"  • Average throughput improvement: {avg_throughput_improvement:+.1f}%")
    print(f"  • This simple rule-based test provides the foundation")
    print(f"    for more complex AV coordination research")
    
    print(f"\n🔬 Next steps for research:")
    print(f"  • Test with mixed AV/human traffic")
    print(f"  • Scale to multiple intersections")
    print(f"  • Add pedestrians and cyclists")
    print(f"  • Consider ML-based coordination only after")
    print(f"    exhausting rule-based approaches")

if __name__ == "__main__":
    main() 