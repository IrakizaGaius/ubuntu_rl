# Ubuntu RL Environment

An advanced reinforcement learning environment simulating Ubuntu philosophy - "I am because we are". Agents learn to balance personal goals with community welfare through dynamic interactions with NPCs, resource management, and ethical decision-making.

## 🌟 Features

- **12 Actions**: 8-directional movement + 4 interaction types (Help, Share, Ignore, Take)
- **16-Dimensional Observations**: Position, velocity, scores, spatial awareness
- **Physics-Based Movement**: Realistic velocity, friction, and collision detection
- **Dynamic Entities**: People needing help, emergencies, resource requesters, obstacles
- **Health & Urgency Systems**: Time-sensitive decision making
- **Resource Management**: Collect and share resources with the community
- **Particle Effects**: Visual feedback for actions
- **Interactive 3D Visualization**: Real-time animated playback with Plotly

## 📁 Project Structure

```
ubuntu_rl/
├── environment/
│   ├── __init__.py
│   ├── custom_env.py       # Custom Gymnasium environment implementation
│   └── rendering.py         # Visualization GUI components 
├── training/
│   ├── __init__.py
│   ├── dqn_training.py            # DQN (Value-Based)
│   ├── pg_training.py             # PPO (Policy Gradient)
│   ├── reinforce_training.py      # REINFORCE (Policy Gradient)
│   └── actor_critic_training.py   # A2C (Actor-Critic)
├── models/
│   ├── dqn/                       # Saved DQN models
│   ├── pg/                        # Saved PPO and REINFORCE models
│   └── actor_critic/              # Saved A2C models
├── main.py                 # Entry point for running best performing model
├── train_all.py            # Train all four algorithms and compare
├── compare_models.py       # Detailed comparison of trained models
├── config.py               # Environment configuration parameters
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## 🚀 Quick Start

### Installation

1. Clone the repository or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

**Train all four algorithms at once (recommended):**

```bash
python train_all.py
```

**Or train algorithms individually:**

```bash
# DQN (Value-Based)
python training/dqn_training.py

# PPO (Policy Gradient)
python training/pg_training.py

# REINFORCE (Policy Gradient)
python training/reinforce_training.py

# A2C (Actor-Critic)
python training/actor_critic_training.py
```

**Training parameters:**
- **DQN**: 200,000 timesteps, learning rate 0.0001
- **PPO**: 200,000 timesteps, learning rate 0.0003
- **REINFORCE**: 200,000 timesteps, learning rate 0.0007
- **A2C**: 200,000 timesteps, learning rate 0.0007

### Model Comparison

Compare all trained models:

```bash
python compare_models.py --episodes 10
```

This will evaluate each algorithm and provide detailed performance metrics.

### Visualization

Run the trained agent with interactive 3D visualization:
```bash
python main.py
```

This will:
1. Load the best performing model (PPO or DQN)
2. Run a full episode (up to 200 steps)
3. Display statistics (rewards, scores, action distribution)
4. Open an interactive 3D animation in your browser

## 🎮 Environment Details

### Action Space

The environment provides 12 discrete actions:

| Action ID | Type | Description |
|-----------|------|-------------|
| 0-7 | Movement | 8-directional movement (N, NE, E, SE, S, SW, W, NW) |
| 8 | Help | Help nearby person in need (Ubuntu action) |
| 9 | Share | Share resources with community |
| 10 | Ignore | Ignore someone in need (Selfish action) |
| 11 | Take | Take resources for yourself |

### Observation Space

16-dimensional continuous observation vector:
- Agent position (x, y)
- Agent velocity (vx, vy)
- Community score, Personal score
- Nearest person info (rel_x, rel_y, urgency, distance)
- Nearest obstacle info (rel_x, rel_y, distance)
- Resources carried
- Nearby entity counts

### Rewards

- **Ubuntu actions** (Help): +10 + urgency bonus
- **Selfish actions** (Ignore, Take): +3
- **Negative outcomes**: -5
- **Step survival**: +0.1
- **Entity death penalty**: -2
- **Collision penalty**: -2

### Entities

1. **People**: Need help, have urgency and health that decay over time
2. **Emergencies**: Critical situations requiring immediate help
3. **Resource Requesters**: NPCs needing resources
4. **Resource Deposits**: Collectible items
5. **Obstacles**: Static or moving barriers

## 🔧 Configuration

Edit `config.py` to adjust environment parameters:

```python
ENV_SIZE = 5                 # Environment dimensions (-5 to +5)
MAX_EPISODE_STEPS = 500      # Maximum steps per episode
REWARD_UBUNTU = 10           # Reward for helping others
REWARD_SELFISH = 3           # Reward for selfish actions
REWARD_NEGATIVE = -5         # Penalty for negative outcomes
COMMUNITY_DECAY = 0.002      # Community score decay rate
```

## 📊 Visualization Features

The interactive 3D visualization includes:

- **3D Environment View**: 
  - Agent (blue diamond) with resource counter
  - NPCs with status indicators (🙋❗🆘✓)
  - Obstacles (gray squares)
  - Resources (gold diamonds)
  - Particle effects
  - Agent trail
  - Velocity indicators

- **Score Graphs**:
  - Community score over time
  - Personal score over time
  - Resources carried

- **Reward Timeline**:
  - Cumulative reward
  - Instant rewards with color coding

- **Interactive Controls**:
  - Play/Pause/Restart buttons
  - Timeline slider
  - 3D rotation and zoom
  - Hover tooltips

## � Algorithms Implemented

### 1. DQN - Deep Q-Network (Value-Based)
- **Type**: Value-Based
- **Approach**: Learns Q-values for state-action pairs
- **Strengths**: Sample efficient with experience replay, good for discrete actions
- **Use Case**: When you need stable learning with replay buffer

### 2. PPO - Proximal Policy Optimization (Policy Gradient)
- **Type**: Policy Gradient
- **Approach**: Directly learns policy with clipped objectives
- **Strengths**: More stable than vanilla policy gradient, good performance
- **Use Case**: Recommended for most RL tasks, good balance

### 3. REINFORCE (Policy Gradient)
- **Type**: Pure Policy Gradient
- **Approach**: Monte Carlo policy gradient with full episode returns
- **Strengths**: Simple and straightforward, no value function needed
- **Use Case**: Educational purposes, baseline comparisons

### 4. A2C - Advantage Actor-Critic (Actor-Critic)
- **Type**: Actor-Critic
- **Approach**: Combines policy (actor) and value function (critic)
- **Strengths**: Reduces variance, fast training
- **Use Case**: When you need both policy and value estimates

## 🧪 Training Tips

1. **Start with `train_all.py`**: Train all algorithms at once for fair comparison
2. **Monitor training**: Check episode rewards increasing over time
3. **Use `compare_models.py`**: Evaluate all models objectively
4. **Hyperparameter Tuning**:
   - Adjust learning rates in training scripts
   - Modify `config.py` for environment difficulty
   - Increase `total_timesteps` for better convergence

## 📈 Performance Metrics

Monitor during training:
- Episode reward (should increase over time)
- Episode length (should reach MAX_EPISODE_STEPS)
- Community score (should remain high)
- Action distribution (should balance movement and interaction)

Typical successful training:
- Initial reward: -346
- Final reward: +230+
- Community score: 0.85+
- Episode completion: 200 steps

### Algorithm Comparison Metrics
Use `compare_models.py` to evaluate:
- **Mean Reward**: Average reward over evaluation episodes
- **Standard Deviation**: Consistency of performance
- **Community Score**: How well agent maintains community welfare
- **Episode Length**: Number of steps before termination

## 🤝 Contributing

Feel free to:
- Add new entity types
- Implement new RL algorithms
- Enhance visualization features
- Optimize hyperparameters
- Improve documentation

## 📝 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/)
- RL algorithms from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- Visualization with [Plotly](https://plotly.com/)
- Inspired by Ubuntu philosophy: "I am because we are"

## 🐛 Troubleshooting

**No trained models found:**
```bash
python training/pg_training.py  # Train a model first
```

**Import errors:**
```bash
pip install -r requirements.txt  # Reinstall dependencies
```

**Visualization not opening:**
- Check that plotly is installed
- Try running in a different browser
- Ensure no firewall is blocking local connections

**Training too slow:**
- Reduce `total_timesteps` for faster testing
- Adjust `MAX_EPISODE_STEPS` in config.py
- Use a more powerful machine or GPU

## 📚 Further Reading

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Ubuntu Philosophy](https://en.wikipedia.org/wiki/Ubuntu_philosophy)
