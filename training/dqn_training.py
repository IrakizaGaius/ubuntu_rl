"""
DQN Hyperparameter Search Script with Visualization for Ubuntu RL Environment
Runs 10 different configurations, saves results, and generates analysis plots
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
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
        self.losses = []
        self.q_values = []
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
                # Get latest loss if available
                if hasattr(self.model, 'logger') and self.model.logger is not None:
                    loss = self.model.logger.name_to_value.get('train/loss', None)
                    if loss is not None:
                        self.losses.append(float(loss))
            except:
                pass
        
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            avg_reward = float(np.mean(self.episode_rewards[-10:]))
            print(f"  {self.n_calls:7d} steps | {len(self.episode_rewards):3d} eps | Reward: {avg_reward:+7.2f}")
        
        return True


def train_dqn_config(config_id, lr, gamma, buffer_size, batch_size, exploration_final, 
                     total_timesteps=150000, save_path="models/dqn/"):
    """Train DQN with specific hyperparameters"""
    
    print(f"\n{'='*70}")
    print(f"CONFIG {config_id}: LR={lr}, γ={gamma}, Buffer={buffer_size}, Batch={batch_size}, ε_final={exploration_final}")
    print('='*70)
    
    env = UbuntuEnv(render_mode=None, vision_radius=2, enable_history=True)
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=lr,
        buffer_size=buffer_size,
        learning_starts=2000,
        batch_size=batch_size,
        tau=1.0,
        gamma=gamma,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final,
        max_grad_norm=10,
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
        model_path = f"{save_path}ubuntu_dqn_config{config_id}"
        model.save(model_path)
        
        # Return results with safe values
        return {
            'config_id': int(config_id),
            'learning_rate': float(lr),
            'gamma': float(gamma),
            'buffer_size': int(buffer_size),
            'batch_size': int(batch_size),
            'exploration_final': float(exploration_final),
            'mean_reward': float(mean_reward),
            'test_reward': float(test_reward),
            'episodes': int(len(callback.episode_rewards)),
            'model_path': str(model_path),
            'episode_rewards': [float(r) for r in callback.episode_rewards],
            'timesteps': [int(t) for t in callback.timesteps],
            'losses': [float(l) for l in callback.losses]
        }
        
    except Exception as e:
        print(f"❌ Config {config_id} Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config_id': int(config_id),
            'learning_rate': float(lr),
            'gamma': float(gamma),
            'buffer_size': int(buffer_size),
            'batch_size': int(batch_size),
            'exploration_final': float(exploration_final),
            'mean_reward': -999.0,
            'test_reward': -999.0,
            'episodes': 0,
            'model_path': None,
            'episode_rewards': [],
            'timesteps': [],
            'losses': []
        }


def plot_cumulative_rewards(results, save_path="models/dqn/plots/"):
    """Plot cumulative rewards over episodes for all configurations"""
    
    os.makedirs(save_path, exist_ok=True)
    
    # Filter out failed configs
    valid_results = [r for r in results if len(r['episode_rewards']) > 0]
    
    if len(valid_results) == 0:
        print("No valid results to plot.")
        return
    
    # Create figure with subplots (2x5 grid for 10 configs)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('DQN Cumulative Rewards Over Episodes - All Configurations', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, result in enumerate(valid_results):
        ax = axes[idx]
        
        episodes = list(range(1, len(result['episode_rewards']) + 1))
        cumulative_rewards = np.cumsum(result['episode_rewards'])
        
        ax.plot(episodes, cumulative_rewards, linewidth=2, color='#2E86AB')
        ax.fill_between(episodes, 0, cumulative_rewards, alpha=0.3, color='#2E86AB')
        
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
    """Plot hyperparameter impact analysis"""
    
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('DQN Hyperparameter Impact Analysis', fontsize=16, fontweight='bold')
    
    # 1. Learning Rate vs Mean Reward
    ax = axes[0, 0]
    lr_grouped = df.groupby('learning_rate')['mean_reward'].agg(['mean', 'std'])
    ax.bar(range(len(lr_grouped)), lr_grouped['mean'], yerr=lr_grouped['std'], 
          capsize=5, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(lr_grouped)))
    ax.set_xticklabels([f"{lr:.4f}" for lr in lr_grouped.index])
    ax.set_xlabel('Learning Rate', fontweight='bold')
    ax.set_ylabel('Mean Reward', fontweight='bold')
    ax.set_title('Learning Rate Impact')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Gamma vs Mean Reward
    ax = axes[0, 1]
    gamma_grouped = df.groupby('gamma')['mean_reward'].agg(['mean', 'std'])
    ax.bar(range(len(gamma_grouped)), gamma_grouped['mean'], yerr=gamma_grouped['std'],
          capsize=5, color='#A23B72', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(gamma_grouped)))
    ax.set_xticklabels([f"{g:.2f}" for g in gamma_grouped.index])
    ax.set_xlabel('Gamma (γ)', fontweight='bold')
    ax.set_ylabel('Mean Reward', fontweight='bold')
    ax.set_title('Discount Factor Impact')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Buffer Size vs Mean Reward
    ax = axes[0, 2]
    buffer_grouped = df.groupby('buffer_size')['mean_reward'].agg(['mean', 'std'])
    ax.bar(range(len(buffer_grouped)), buffer_grouped['mean'], yerr=buffer_grouped['std'],
          capsize=5, color='#F18F01', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(buffer_grouped)))
    ax.set_xticklabels([f"{int(b/1000)}K" for b in buffer_grouped.index])
    ax.set_xlabel('Replay Buffer Size', fontweight='bold')
    ax.set_ylabel('Mean Reward', fontweight='bold')
    ax.set_title('Buffer Size Impact')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Batch Size vs Mean Reward
    ax = axes[1, 0]
    batch_grouped = df.groupby('batch_size')['mean_reward'].agg(['mean', 'std'])
    ax.bar(range(len(batch_grouped)), batch_grouped['mean'], yerr=batch_grouped['std'],
          capsize=5, color='#C73E1D', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(batch_grouped)))
    ax.set_xticklabels([f"{int(b)}" for b in batch_grouped.index])
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Mean Reward', fontweight='bold')
    ax.set_title('Batch Size Impact')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Final Exploration vs Mean Reward
    ax = axes[1, 1]
    eps_grouped = df.groupby('exploration_final')['mean_reward'].agg(['mean', 'std'])
    ax.bar(range(len(eps_grouped)), eps_grouped['mean'], yerr=eps_grouped['std'],
          capsize=5, color='#6A994E', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(eps_grouped)))
    ax.set_xticklabels([f"{e:.2f}" for e in eps_grouped.index])
    ax.set_xlabel('Final Epsilon (ε)', fontweight='bold')
    ax.set_ylabel('Mean Reward', fontweight='bold')
    ax.set_title('Exploration Strategy Impact')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Overall Performance Comparison
    ax = axes[1, 2]
    top_5_df = df.nlargest(5, 'mean_reward')
    colors_bar = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    ax.barh(range(len(top_5_df)), top_5_df['mean_reward'], color=colors_bar, 
           alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_5_df)))
    ax.set_yticklabels([f"Config {int(cid)}" for cid in top_5_df['config_id']])
    ax.set_xlabel('Mean Reward', fontweight='bold')
    ax.set_ylabel('Configuration', fontweight='bold')
    ax.set_title('Top 5 Configurations')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}hyperparameter_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}hyperparameter_analysis.png")
    plt.close()


def main():
    """Run hyperparameter search with 10 different configurations"""
    
    # Create save directories
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("models/dqn/plots", exist_ok=True)
    
    print("\n" + "="*70)
    print("DQN HYPERPARAMETER SEARCH - UBUNTU RL ENVIRONMENT")
    print("="*70)
    
    # Define 10 different hyperparameter configurations
    configs = [
        # Config 1: Baseline (original settings)
        {'lr': 0.0003, 'gamma': 0.99, 'buffer': 200000, 'batch': 64, 'eps_final': 0.1},
        
        # Config 2: Higher learning rate
        {'lr': 0.001, 'gamma': 0.50, 'buffer': 100000, 'batch': 64, 'eps_final': 0.2},
        
        # Config 3: Lower learning rate for stability
        {'lr': 0.0001, 'gamma': 0.10, 'buffer': 100000, 'batch': 32, 'eps_final': 0.05},
        
        # Config 4: Lower gamma (shorter-term focus)
        {'lr': 0.0005, 'gamma': 0.95, 'buffer': 200000, 'batch': 64, 'eps_final': 0.1},
        
        # Config 5: Larger buffer for more diversity
        {'lr': 0.0008, 'gamma': 0.99, 'buffer': 200000, 'batch': 128, 'eps_final': 0.3},
        
        # Config 6: Smaller buffer for faster adaptation
        {'lr': 0.0003, 'gamma': 0.9, 'buffer': 300000, 'batch': 64, 'eps_final': 0.1},
        
        # Config 7: Larger batch size
        {'lr': 0.001, 'gamma': 0.99, 'buffer': 100000, 'batch': 128, 'eps_final': 0.1},
        
        # Config 8: Smaller batch for more frequent updates
        {'lr': 0.001, 'gamma': 0.8, 'buffer': 100000, 'batch': 128, 'eps_final': 0.4},
        
        # Config 9: Higher final exploration
        {'lr': 0.0007, 'gamma': 0.79, 'buffer': 200000, 'batch': 64, 'eps_final': 0.2},
        
        # Config 10: Lower final exploration (more exploitation)
        {'lr': 0.01, 'gamma': 0.99, 'buffer': 100000, 'batch': 64, 'eps_final': 0.05},
    ]
    
    results = []
    
    # Train each configuration
    for i, config in enumerate(configs, 1):
        result = train_dqn_config(
            config_id=i,
            lr=config['lr'],
            gamma=config['gamma'],
            buffer_size=config['buffer'],
            batch_size=config['batch'],
            exploration_final=config['eps_final'],
            total_timesteps=150000
        )
        results.append(result)
    
    # Create results DataFrame
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['episode_rewards', 'timesteps', 'losses']} 
                       for r in results])
    df = df.sort_values('mean_reward', ascending=False)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"models/dqn/dqn_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Print results table
    print("\n" + "="*70)
    print("RESULTS SUMMARY (Sorted by Mean Reward)")
    print("="*70)
    print(df[['config_id', 'learning_rate', 'gamma', 'buffer_size', 'batch_size', 
              'exploration_final', 'mean_reward']].to_string(index=False))
    
    # Identify best configuration
    best_config_id = df.iloc[0]['config_id']
    best_result = [r for r in results if r['config_id'] == best_config_id][0]
    
    print("\n" + "="*70)
    print("BEST CONFIGURATION:")
    print("="*70)
    print(f"Config ID: {best_result['config_id']}")
    print(f"Learning Rate: {best_result['learning_rate']}")
    print(f"Gamma: {best_result['gamma']}")
    print(f"Buffer Size: {best_result['buffer_size']}")
    print(f"Batch Size: {best_result['batch_size']}")
    print(f"Final Exploration: {best_result['exploration_final']}")
    print(f"Mean Reward: {best_result['mean_reward']:.2f}")
    print(f"Test Reward: {best_result['test_reward']:.2f}")
    print(f"Model Path: {best_result['model_path']}.zip")
    
    # Copy best model to standard location
    if best_result['model_path']:
        import shutil
        best_model_path = "models/dqn/ubuntu_agent_dqn_best"
        shutil.copy(f"{best_result['model_path']}.zip", f"{best_model_path}.zip")
        print(f"\n✅ Best model saved to: {best_model_path}.zip")
    
    print(f"\n📊 Full results saved to: {csv_path}")
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING ANALYSIS PLOTS...")
    print("="*70)
    
    plot_cumulative_rewards(results, save_path="models/dqn/plots/")
    
    print("\n✅ All plots generated successfully!")
    print("="*70)
    
    return df, results


if __name__ == "__main__":
    results_df, all_results = main()