# **Ubuntu RL Environment**

A reinforcement learning environment where an AI agent learns Ubuntu philosophy - the African concept of interconnected humanity meaning "I am because we are". The agent navigates a 9×9 grid world, making moral and ethical decisions while balancing navigation efficiency, philosophy diversity, and community welfare to achieve Ubuntu Mastery.

## **🌟 Features**

- **Multi-Objective Learning**: Balance goal-reaching, moral decision-making, and philosophy diversity
- **Rich Semantic Observations**: 5×5 vision grid with one-hot encoded philosophy types, temporal memory (5-step history), and comprehensive evaluation metrics
- **10 Philosophy Types**: 5 positive (help_needy, share_knowledge, show_compassion, build_community, forgive) and 5 negative (selfish_act, harm_other, hoard_resources, show_greed, violence)
- **Dynamic Environment**: Adversarial philosophy placement, dynamic spawning (5% per step), and robust diversity requirements
- **Comprehensive Evaluation**: Tracks diversity score, consistency score, harm minimization, and composite Ubuntu score
- **Beautiful Visualization**: Interactive pygame rendering with Ubuntu Temple, philosophy symbols, and AI robot agent

## **📁 Project Structure**

```bash
ubuntu_rl/

├── Configurations/

│ └── config.py # Environment configuration (ENV_SIZE=9.0, MAX_EPISODE_STEPS=500)

├── environment/

│ ├── \__init_\_.py

│ └── custom_env.py # Custom Gymnasium environment with Ubuntu philosophy

├── models/

│ ├── dqn/ # DQN models and results

│ │ ├── plots/ # Training analysis plots

│ │ └── dqn_results_\*.csv # Hyperparameter search results

│ ├── ppo/ # PPO models and results

│ ├── a2c/ # A2C models and results

│ └── reinforce/ # REINFORCE models and results

├── training/

│ ├── dqn_training.py # DQN hyperparameter search (10 configs)

│ ├── pg_training.py # PPO hyperparameter search (10 configs)

│ ├── actor_critic_training.py # A2C hyperparameter search (10 configs)

│ └── reinforce_training.py # REINFORCE hyperparameter search (10 configs)

├── requirements.txt # Project dependencies

└── README.md # Project documentation
```

## **🚀 Quick Start**

### **Installation**

1. Clone the repository:

```bash
git clone &lt;repository-url&gt;

cd ubuntu_rl
```

1. Create and activate a virtual environment:

```bash
python -m venv .venv

source .venv/bin/activate # On Windows: .venv\\Scripts\\activate
```

**4 discrete actions for grid navigation:**

| Action | Direction | Effect                      |
| ------ | --------- | --------------------------- |
| 0      | LEFT      | Move one cell left (x - 1)  |
| 1      | RIGHT     | Move one cell right (x + 1) |
| 2      | UP        | Move one cell up (y + 1)    |
| 3      | DOWN      | Move one cell down (y - 1)  |

1. Install dependencies:

```bash
pip install -r requirements.txt
```

### **Training**

Each training script runs **10 different hyperparameter configurations** and automatically:

- Trains all configurations with 150,000 timesteps each
- Saves all models to models/`<algorithm>`/
- Identifies and saves the best model as ubuntu_agent_`<algorithm>`_best.zip
- Generates comprehensive analysis plots in models/`<algorithm>`/plots/
- Saves results table to CSV with timestamp

**Train algorithms:**

```bash
# DQN (Deep Q-Network) - Value-Based
python training/dqn_training.py

# PPO (Proximal Policy Optimization) - Policy Gradient
python training/pg_training.py

# A2C (Advantage Actor-Critic) - Actor-Critic
python training/actor_critic_training.py

# REINFORCE (Pure Policy Gradient) - Baseline
python training/reinforce_training.py
```

**Total Reward:** $R_{total} = R_{cell} + R_{progress} + R_{movement} + R_{terminal}$

| Component                                               | Condition                                       | Reward Range |
| ------------------------------------------------------- | ----------------------------------------------- | ------------ |
| Good Philosophy                                         | help_needy, share_knowledge, etc.               | +12 to +30   |
| Bad Philosophy                                          | violence, harm_other, etc.                      | -50 to -8    |
| Progress                                                | Moving toward goal                              | +0.5 to +2.0 |
| Proximity                                               | Within 3 cells of goal                          | +2.0 to +8.0 |
| Wall/Stuck                                              | Invalid moves, oscillation                      | -2.0 to -4.0 |
| Goal Reached                                            | Ubuntu Mastery (score ≥ 60 + diversity ≥ 70%) | +400         |
| Goal Reached                                            | Partial success                                 | +200 to +250 |
| Violence Limit                                          | 5+ bad actions                                  | -200         |
| \# PPO (Proximal Policy Optimization) - Policy Gradient |                                                 |              |

python training/pg_training.py
\# REINFORCE (Pure Policy Gradient) - Baseline

python training/reinforce_training.py

**Good Philosophies:**

| Philosophy      | Reward                   |
| --------------- | ------------------------ |
| help_needy      | +30 (15 × 2 multiplier) |
| share_knowledge | +24                      |
| show_compassion | +20                      |
| build_community | +16                      |
| forgive         | +12                      |

**Bad Philosophies:**

| Philosophy      | Reward |
| --------------- | ------ |
| violence        | -50    |
| harm_other      | -15    |
| selfish_act     | -12    |
| hoard_resources | -10    |
| show_greed      | -8     |

- Results CSV: models/&lt;algorithm&gt;/&lt;algorithm&gt;\_results_&lt;timestamp&gt;.csv
- Plots:
  -cumulative_rewards_all_configs.png - All 10 configurations

**Pygame rendering features:**

- Sky gradient with sun and clouds
- Ubuntu Temple (golden shrine at goal)
- Philosophy symbols: Rounded squares (good) vs triangles (bad)
- AI Robot agent with state-based colors and animations
- Nature elements: Trees, bushes, houses, flowers
- Path trails: Visual history of agent movement
- Particle effects: Philosophy interaction feedback
- Info panel: Real-time Ubuntu score, state, and actions

python -c "

**Contributions welcome!**

Areas for improvement:

- Additional RL algorithms (SAC, TD3, Rainbow DQN)
- Curriculum learning for moral progression
- Multi-agent Ubuntu scenarios
- Transfer learning across grid sizes
- Interpretability analysis of learned policies

```python
import gymnasium as gym
model = DQN.load('models/dqn/ubuntu_agent_dqn_best')

for episode in range(3):

obs, _ = env.reset()

done = False

total_reward = 0

while not done:

action, _ = model.predict(obs, deterministic=True)

obs, reward, terminated, truncated, info = env.step(action)

total_reward += reward

done = terminated or truncated

env.render()

print(f'Episode {episode+1}: Reward={total_reward:.1f}, Ubuntu Score={env.ubuntu_score:.1f}, State={env.ubuntu_state}')

env.close()
```

## **🎮 Environment Details**

### **Action Space**

4 discrete actions for grid navigation:

| **Action** | **Direction** | **Effect**            |
| ---------------- | ------------------- | --------------------------- |
| 0                | LEFT                | Move one cell left (x - 1)  |
| 1                | RIGHT               | Move one cell right (x + 1) |
| 2                | UP                  | Move one cell up (y + 1)    |
| 3                | DOWN                | Move one cell down (y - 1)  |

### **Observation Space**

**Type:** Box(low=-1.0, high=1.0, shape=(1584,), dtype=float32)

**Components:**

1. **Vision Grid (250 dims)**: 5×5 grid around agent, each cell encoded as 10-dim one-hot vector for philosophy type
2. **Position & Progress (6 dims)**: Normalized x/y position, Ubuntu score, positive/negative actions, distance to goal
3. **Ubuntu State (3 dims)**: One-hot encoding of LEARNING/UBUNTU_LEARNER/UBUNTU_MASTER states
4. **Evaluation Metrics (3 dims)**: Diversity score, consistency score, harm minimization score
5. **History (1320 dims)**: Last 5 observations flattened for temporal reasoning

### **Reward Structure**

**Total Reward:** R_total = R_cell + R_progress + R_movement + R_terminal

| **Component**       | **Condition**                         | **Reward Range** |
| ------------------------- | ------------------------------------------- | ---------------------- |
| **Good Philosophy** | help_needy, share_knowledge, etc.           | +12 to +30             |
| **Bad Philosophy**  | violence, harm_other, etc.                  | \-50 to -8             |
| **Progress**        | Moving toward goal                          | +0.5 to +2.0           |
| **Proximity**       | Within 3 cells of goal                      | +2.0 to +8.0           |
| **Wall/Stuck**      | Invalid moves, oscillation                  | \-2.0 to -4.0          |
| **Goal Reached**    | Ubuntu Mastery (score≥60 + diversity≥70%) | +400                   |
| **Goal Reached**    | Partial success                             | +200 to +250           |
| **Violence Limit**  | 5+ bad actions                              | \-200                  |

### **Ubuntu States**

The agent progresses through 4 moral development states:

1. **LEARNING** (White robot): Initial exploration phase
2. **UBUNTU_LEARNER** (Green robot): Score ≥30, showing positive growth
3. **UBUNTU_MASTER** (Gold robot with crown): Score ≥60 + reached goal + 70% diversity
4. **ANTI_UBUNTU** (Red robot): 5+ bad actions, fallen to negative path

### **Philosophy Types & Rewards**

**Good Philosophies:**

- help_needy: +30 (15 × 2 multiplier)
- share_knowledge: +24
- show_compassion: +20
- build_community: +16
- forgive: +12

**Bad Philosophies:**

- violence: -50
- harm_other: -15
- selfish_act: -12
- hoard_resources: -10
- show_greed: -8

## **📊 Performance Benchmarks**

### **Top Performing Configurations**

**DQN:**

- Best: Config 10 (LR=0.01, γ=0.99, ε_final=0.05) → **778.7** mean reward
- Convergence: ~600-800 episodes (50-60% of training)
- Key insight: Aggressive learning + high gamma + low exploration = optimal

**A2C:**

- Best: Config 1 (LR=0.0007, γ=0.99) → **697.7** mean reward
- Convergence: Stable throughout training
- Key insight: Moderate LR + high gamma + balanced actor-critic

**PPO:**

- Best: TBD (run python training/pg_training.py)
- Expected: High sample efficiency with long rollouts

### **Critical Findings**

1. **Gamma is crucial**: All algorithms fail catastrophically with γ<0.95 (rewards drop to 250-450)
2. **Diversity requirement**: Forces exploration beyond greedy exploitation
3. **Sparse rewards**: Terminal +400 requires long-horizon credit assignment
4. **Multi-objective**: Must balance navigation, morality, and diversity simultaneously

## **🔧 Configuration**

Edit Configurations/config.py:

ENV_SIZE = 9.0 # 9×9 grid world

MAX_EPISODE_STEPS = 500 # Maximum steps per episode

Key environment parameters (in custom_env.py):

- vision_radius = 2: 5×5 vision grid
- history_length = 5: 5-step temporal memory
- ubuntu_threshold = 60: Score needed for mastery
- violence_limit = 5: Max bad actions before termination
- required_diversity = 7: Must encounter 7/10 philosophy types

## **📈 Hyperparameter Search Results**

### **DQN Tested Configurations**

Varies: Learning rate (0.0001-0.01), gamma (0.5-0.99), buffer size (50K-200K), batch size (32-128), final epsilon (0.05-0.2)

### **A2C Tested Configurations**

Varies: Learning rate (0.0001-0.1), gamma (0.1-0.99), n_steps (5), entropy coef (0.001-1.0), vf_coef (0.01-1.0)

### **PPO Tested Configurations**

Varies: Learning rate (0.0001-0.0005), gamma (0.95-0.99), n_steps (1024-4096), batch size (32-128), n_epochs (10-20), clip_range (0.2-0.3), entropy coef (0.001-0.05)

## **🎨 Visualization Features**

The pygame rendering includes:

- **Sky gradient** with sun and clouds
- **Ubuntu Temple** (golden shrine at goal)
- **Philosophy symbols**: Rounded squares (good) vs triangles (bad)
- **AI Robot agent** with state-based colors and animations
- **Nature elements**: Trees, bushes, houses, flowers
- **Path trails**: Visual history of agent movement
- **Particle effects**: Philosophy interaction feedback
- **Info panel**: Real-time Ubuntu score, state, and actions

## **🤝 Contributing**

Contributions welcome! Areas for improvement:

- Additional RL algorithms (SAC, TD3, Rainbow DQN)
- Curriculum learning for moral progression
- Multi-agent Ubuntu scenarios
- Transfer learning across grid sizes
- Interpretability analysis of learned policies

## **📝 License**

This project is open source and available for educational and research purposes.

## **🙏 Acknowledgments**

- Built with [Gymnasium](https://gymnasium.farama.org/)
- RL algorithms from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- Inspired by Ubuntu philosophy: "Umuntu ngumuntu ngabantu" (A person is a person through other people)
- Environment design emphasizes value alignment and multi-objective decision-making in AI systems.
