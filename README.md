# 🚗 Autonomous Vehicle Traffic Simulation

**Testing the hypothesis: "In a world where all cars are autonomous, we won't need stop signs or red lights because each car will know where every other car is."**

## ⚡ Ultra-Clean Implementation

Everything you need in **one file**: `autocar.py`

### 🧠 Modern AI Stack
- **Transformer neural networks** (multi-head attention like GPT)
- **Kalman filters** for state estimation (like Tesla)
- **Reinforcement learning** for coordination
- **No outdated LSTMs** - pure modern architecture

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch pygame numpy matplotlib

# Run everything
python autocar.py
```

### Simple Workflow

**Option 1: Train then simulate**
```bash
python autocar.py
# Choose: 3 (Both)
# Enter episodes: 100
# Watch: Training → Simulation
```

**Option 2: Just train**
```bash
python autocar.py
# Choose: 1 (Train)
# Model saved automatically
```

**Option 3: Just simulate** (requires trained model)
```bash
python autocar.py
# Choose: 2 (Simulate)
# Watch autonomous coordination
```

## 🔬 What You'll See

### Training Phase
- Vehicles start chaotic, gradually learn coordination
- Real-time metrics: rewards, throughput, loss
- Model automatically saved when complete

### Simulation Phase  
- Smooth intersection coordination without traffic lights
- Color-coded vehicle behaviors (normal, cooperative, etc.)
- Real-time performance metrics
- Interactive controls

## 🎮 Controls

- **SPACE**: Add more vehicles
- **ESC**: Exit

## 📁 Ultra-Clean Structure

```
autocar/
├── autocar.py          # Everything in one file
├── requirements.txt    # Minimal dependencies
└── README.md          # This file
```

That's it! No bloat, no confusion.

## 🧬 Technical Details

### Transformer Architecture
- **Input**: 64-dim sensor observations
- **Layers**: 4 transformer blocks with 8 attention heads
- **Output**: 4 actions (accelerate, maintain, brake, cooperate)
- **Training**: Q-learning with experience replay

### Kalman Filter
- **State**: [x, y, velocity_x, velocity_y]  
- **Sensors**: Noisy GPS + odometry
- **Purpose**: Robust state estimation like real AVs

### Vehicle Physics
- Realistic acceleration/deceleration limits
- Speed constraints and safety margins
- Collision detection and avoidance

## 📊 Expected Results

With modern transformer + Kalman architecture:
- **Learning**: Vehicles learn optimal intersection timing
- **Coordination**: Attention mechanism enables V2V understanding  
- **Efficiency**: Should outperform fixed traffic signals
- **Safety**: Collision avoidance through predictive modeling

## 🎯 Success Criteria

**Hypothesis supported if:**
- Throughput > 30 vehicles/minute through intersections
- Smooth coordination without traffic control infrastructure
- Low collision risk maintained throughout simulation
- Cooperative behaviors emerge from training

---

**Built with state-of-the-art transformer neural networks and realistic autonomous vehicle control systems.**

*One file. Modern AI. Clean results.*