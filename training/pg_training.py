"""
PPO Hyperparameter Search Script with Visualization for Ubuntu RL Environment
Runs 10 different configurations, saves results, and generates analysis plots
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from environment.custom_env import UbuntuEnv
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Utility to ensure hashable types for results
def safe_value(val):
    """Convert numpy types to native Python types"""
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return val.item()
        else:
            return val.tolist()
    elif isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    elif isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    elif isinstance(val, tuple):
        return tuple(safe_value(x) for x in val)
    elif isinstance(val, list):
        return [safe_value(x) for x in val]
    elif isinstance(val, dict):
        return {safe_value(k): safe_value(v) for k, v in val.items()}
    return val


class DetailedProgressCallback(BaseCallback):
    """Callback for tracking detailed training metrics"""
    
    def __init__(self, check_freq=1000):
        super().__init__()
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.clip_fractions = []
        self._seen_episodes = set()  # Track which episodes we've already logged
        
    def _on_step(self):
        # Track episode rewards - FIX: Properly handle ep_info_buffer
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                # Create a unique identifier for this episode
                # Use timestep + reward as identifier (not perfect but works)
                if 'r' in info and 'l' in info:
                    episode_id = (self.num_timesteps, float(info['r']))
                    
                    if episode_id not in self._seen_episodes:
                        self._seen_episodes.add(episode_id)
                        # Convert to native Python types immediately
                        reward = float(info['r'])
                        length = int(info['l'])
                        self.episode_rewards.append(reward)
                        self.episode_lengths.append(length)
                        self.timesteps.append(int(self.num_timesteps))
        
        # Track training metrics periodically
        if self.n_calls % 100 == 0:
            try:
                # Get latest losses if available
                if hasattr(self.model, 'logger') and self.model.logger is not None:
                    policy_loss = self.model.logger.name_to_value.get('train/policy_loss', None)
                    value_loss = self.model.logger.name_to_value.get('train/value_loss', None)
                    entropy_loss = self.model.logger.name_to_value.get('train/entropy_loss', None)
                    clip_fraction = self.model.logger.name_to_value.get('train/clip_fraction', None)
                    
                    if policy_loss is not None:
                        self.policy_losses.append(float(policy_loss))
                    if value_loss is not None:
                        self.value_losses.append(float(value_loss))
                    if entropy_loss is not None:
                        self.entropy_losses.append(float(entropy_loss))
                    if clip_fraction is not None:
                        self.clip_fractions.append(float(clip_fraction))
            except:
                pass
        
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            avg_reward = float(np.mean(self.episode_rewards[-10:]))
            print(f"  {self.n_calls:7d} steps | {len(self.episode_rewards):3d} eps | Reward: {avg_reward:+7.2f}")
        
        return True


def train_ppo_config(config_id, lr, gamma, n_steps, batch_size, n_epochs, clip_range, ent_coef,
                     total_timesteps=150000, save_path="models/ppo/"):
    """Train PPO with specific hyperparameters"""
    
    print(f"\n{'='*70}")
    print(f"CONFIG {config_id}: LR={lr}, γ={gamma}, steps={n_steps}, batch={batch_size}, epochs={n_epochs}, clip={clip_range}")
    print('='*70)
    
    env = UbuntuEnv(render_mode=None, vision_radius=2, enable_history=True)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0
    )
    
    callback = DetailedProgressCallback(check_freq=5000)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False
        )
        
        # Calculate mean reward from last 20 episodes
        if len(callback.episode_rewards) >= 20:
            mean_reward = float(np.mean(callback.episode_rewards[-20:]))
        elif len(callback.episode_rewards) > 0:
            mean_reward = float(np.mean(callback.episode_rewards))
        else:
            mean_reward = -999.0
        
        # Test the model
        obs, _ = env.reset()
        test_reward = 0.0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            test_reward += float(reward)
            if terminated or truncated:
                break
        
        print(f"✅ Config {config_id} Complete | Train: {mean_reward:+.2f} | Test: {test_reward:+.2f}")
        
        # Save model
        model_path = f"{save_path}ubuntu_ppo_config{config_id}"
        model.save(model_path)
        
        # Return results with safe values
        return {
            'config_id': int(config_id),
            'learning_rate': float(lr),
            'gamma': float(gamma),
            'n_steps': int(n_steps),
            'batch_size': int(batch_size),
            'n_epochs': int(n_epochs),
            'clip_range': float(clip_range),
            'ent_coef': float(ent_coef),
            'mean_reward': float(mean_reward),
            'test_reward': float(test_reward),
            'episodes': int(len(callback.episode_rewards)),
            'model_path': str(model_path),
            'episode_rewards': [float(r) for r in callback.episode_rewards],
            'timesteps': [int(t) for t in callback.timesteps],
            'policy_losses': [float(l) for l in callback.policy_losses],
            'value_losses': [float(l) for l in callback.value_losses],
            'entropy_losses': [float(l) for l in callback.entropy_losses],
            'clip_fractions': [float(c) for c in callback.clip_fractions]
        }
        
    except Exception as e:
        print(f"❌ Config {config_id} Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config_id': int(config_id),
            'learning_rate': float(lr),
            'gamma': float(gamma),
            'n_steps': int(n_steps),
            'batch_size': int(batch_size),
            'n_epochs': int(n_epochs),
            'clip_range': float(clip_range),
            'ent_coef': float(ent_coef),
            'mean_reward': -999.0,
            'test_reward': -999.0,
            'episodes': 0,
            'model_path': None,
            'episode_rewards': [],
            'timesteps': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'clip_fractions': []
        }


def plot_cumulative_rewards(results, save_path="models/ppo/plots/"):
    """Plot cumulative rewards over episodes for all configurations"""
    
    os.makedirs(save_path, exist_ok=True)
    
    # Filter out failed configs
    valid_results = [r for r in results if len(r['episode_rewards']) > 0]
    
    if len(valid_results) == 0:
        print("No valid results to plot.")
        return
    
    # Create figure with subplots (2x5 grid for 10 configs)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('PPO Cumulative Rewards Over Episodes - All Configurations', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, result in enumerate(valid_results):
        ax = axes[idx]
        
        episodes = list(range(1, len(result['episode_rewards']) + 1))
        cumulative_rewards = np.cumsum(result['episode_rewards'])
        
        ax.plot(episodes, cumulative_rewards, linewidth=2, color='#16A085')
        ax.fill_between(episodes, 0, cumulative_rewards, alpha=0.3, color='#16A085')
        
        ax.set_title(f"Config {result['config_id']}\nLR={result['learning_rate']}, γ={result['gamma']}", 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel('Cumulative Reward', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add mean reward annotation
        ax.text(0.95, 0.05, f'Mean: {result["mean_reward"]:.1f}', 
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(valid_results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}cumulative_rewards_all_configs.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}cumulative_rewards_all_configs.png")
    plt.close()


def main():
    """Run hyperparameter search with 10 different configurations"""

    # Create save directories
    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("models/ppo/plots", exist_ok=True)

    print("\n" + "="*70)
    print("PPO HYPERPARAMETER SEARCH - UBUNTU RL ENVIRONMENT")
    print("="*70)

    # Define 10 different hyperparameter configurations
    configs = [
        # Config 1: Baseline
        {'lr': 0.0003, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.01},

        # Config 2: Higher learning rate
        {'lr': 0.0005, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.01},

        # Config 3: Lower learning rate
        {'lr': 0.0001, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.01},

        # Config 4: Lower gamma
        {'lr': 0.0003, 'gamma': 0.95, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.01},

        # Config 5: More steps (keep batch_size valid)
        {'lr': 0.0003, 'gamma': 0.99, 'n_steps': 4096, 'batch_size': 128, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.01},

        # Config 6: Larger batch
        {'lr': 0.0003, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 128, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.01},

        # Config 7: More epochs
        {'lr': 0.0003, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 20, 'clip_range': 0.2, 'ent_coef': 0.01},

        # Config 8: Larger clip range
        {'lr': 0.0003, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.3, 'ent_coef': 0.01},

        # Config 9: Higher entropy
        {'lr': 0.0003, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.05},

        # Config 10: Smaller steps, smaller batch
        {'lr': 0.0003, 'gamma': 0.99, 'n_steps': 1024, 'batch_size': 32, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.001},
    ]

    results = []

    # Train each configuration
    for i, config in enumerate(configs, 1):
        result = train_ppo_config(
            config_id=i,
            lr=config['lr'],
            gamma=config['gamma'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            total_timesteps=150000
        )
        results.append(result)

    # Create results DataFrame
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['episode_rewards', 'timesteps', 'policy_losses', 'value_losses', 'entropy_losses', 'clip_fractions']}
                       for r in results])
    df = df.sort_values('mean_reward', ascending=False)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"models/ppo/ppo_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    # Print results table
    print("\n" + "="*70)
    print("RESULTS SUMMARY (Sorted by Mean Reward)")
    print("="*70)
    print(df[['config_id', 'learning_rate', 'gamma', 'n_steps', 'batch_size',
              'n_epochs', 'clip_range', 'ent_coef', 'mean_reward']].to_string(index=False))

    # Identify best configuration
    best_config_id = df.iloc[0]['config_id']
    best_result = [r for r in results if r['config_id'] == best_config_id][0]

    print("\n" + "="*70)
    print("BEST CONFIGURATION:")
    print("="*70)
    print(f"Config ID: {best_result['config_id']}")
    print(f"Learning Rate: {best_result['learning_rate']}")
    print(f"Gamma: {best_result['gamma']}")
    print(f"N Steps: {best_result['n_steps']}")
    print(f"Batch Size: {best_result['batch_size']}")
    print(f"N Epochs: {best_result['n_epochs']}")
    print(f"Clip Range: {best_result['clip_range']}")
    print(f"Entropy Coefficient: {best_result['ent_coef']}")
    print(f"Mean Reward: {best_result['mean_reward']:.2f}")
    print(f"Test Reward: {best_result['test_reward']:.2f}")
    print(f"Model Path: {best_result['model_path']}.zip")

    # Copy best model to standard location
    if best_result['model_path']:
        import shutil
        best_model_path = "models/ppo/ubuntu_agent_ppo_best"
        shutil.copy(f"{best_result['model_path']}.zip", f"{best_model_path}.zip")
        print(f"\n✅ Best model saved to: {best_model_path}.zip")

    print(f"\n📊 Full results saved to: {csv_path}")

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING ANALYSIS PLOTS...")
    print("="*70)

    plot_cumulative_rewards(results, save_path="models/ppo/plots/")

    print("\n✅ All plots generated successfully!")
    print("="*70)

    return df, results