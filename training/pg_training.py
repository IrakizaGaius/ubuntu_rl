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
import warnings
warnings.filterwarnings('ignore')


class DetailedProgressCallback(BaseCallback):
    """Callback for tracking detailed training metrics"""
    
    def __init__(self, check_freq=1000):
        super().__init__()
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        
    def _on_step(self):
        # Track episode rewards
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info and 'l' in info:
                    # Simple duplicate check
                    if len(self.episode_rewards) == 0 or info['r'] != self.episode_rewards[-1]:
                        self.episode_rewards.append(float(info['r']))
                        self.episode_lengths.append(int(info['l']))
                        self.timesteps.append(int(self.num_timesteps))
        
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            avg_reward = float(np.mean(self.episode_rewards[-10:]))
            print(f"  {self.n_calls:7d} steps | {len(self.episode_rewards):3d} eps | Reward: {avg_reward:+7.2f}")
        
        return True


def train_ppo_config(config_id, lr, gamma, n_steps, batch_size, n_epochs, clip_range, ent_coef,
                     total_timesteps=150000, save_path="models/ppo/"):
    """Train PPO with specific hyperparameters"""
    
    print(f"\n{'='*70}")
    print(f"CONFIG {config_id}: LR={lr}, γ={gamma}, steps={n_steps}, batch={batch_size}, epochs={n_epochs}")
    print(f"  clip={clip_range}, entropy={ent_coef}")
    print('='*70)
    
    try:
        # Validate hyperparameters
        if batch_size > n_steps:
            print(f"⚠️  WARNING: batch_size ({batch_size}) > n_steps ({n_steps}), adjusting batch_size to {n_steps}")
            batch_size = n_steps
        
        print("Creating environment...")
        env = UbuntuEnv(render_mode=None, vision_radius=2, enable_history=True)
        print("✓ Environment created")
        
        print("Initializing PPO model...")
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
        print("✓ Model initialized")
        
        callback = DetailedProgressCallback(check_freq=5000)
        
        print(f"Starting training for {total_timesteps:,} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
        )
        print("✓ Training completed")
        
        # Calculate mean reward
        if len(callback.episode_rewards) >= 20:
            mean_reward = float(np.mean(callback.episode_rewards[-20:]))
        elif len(callback.episode_rewards) > 0:
            mean_reward = float(np.mean(callback.episode_rewards))
        else:
            mean_reward = -999.0
        
        print("Testing model...")
        # Test the model
        obs, _ = env.reset()
        test_reward = 0.0
        test_steps = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            test_reward += float(reward)
            test_steps += 1
            if terminated or truncated:
                break
        
        print(f"✅ Config {config_id} Complete | Train: {mean_reward:+.2f} | Test: {test_reward:+.2f} ({test_steps} steps)")
        
        # Save model
        model_path = f"{save_path}ubuntu_ppo_config{config_id}"
        model.save(model_path)
        print(f"✓ Model saved to {model_path}.zip")
        
        env.close()
        
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
            'timesteps': [int(t) for t in callback.timesteps]
        }
        
    except Exception as e:
        print(f"❌ Config {config_id} Failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")
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
            'timesteps': []
        }


def plot_cumulative_rewards(results, save_path="models/ppo/plots/"):
    """Plot cumulative rewards over episodes for all configurations"""
    
    os.makedirs(save_path, exist_ok=True)
    
    valid_results = [r for r in results if len(r['episode_rewards']) > 0]
    
    if len(valid_results) == 0:
        print("⚠️  No valid results to plot.")
        return
    
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
        
        ax.text(0.95, 0.05, f'Mean: {result["mean_reward"]:.1f}', 
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=8)
    
    for idx in range(len(valid_results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}cumulative_rewards_all_configs.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}cumulative_rewards_all_configs.png")
    plt.close()


def main():
    """Run hyperparameter search with 10 different configurations"""

    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("models/ppo/plots", exist_ok=True)

    print("\n" + "="*70)
    print("PPO HYPERPARAMETER SEARCH - UBUNTU RL ENVIRONMENT")
    print("="*70)

    # Define 10 VALID hyperparameter configurations
    # Key constraint: batch_size must be <= n_steps
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

    for i, config in enumerate(configs, 1):
        print(f"\n\n{'#'*70}")
        print(f"# STARTING CONFIG {i}/10")
        print(f"{'#'*70}")
        
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
        
        print(f"✓ Config {i}/10 completed")

    print("\n" + "="*70)
    print("ALL CONFIGURATIONS COMPLETED")
    print("="*70)

    # Create results DataFrame
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['episode_rewards', 'timesteps']}
                       for r in results])
    df = df.sort_values('mean_reward', ascending=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"models/ppo/ppo_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "="*70)
    print("RESULTS SUMMARY (Sorted by Mean Reward)")
    print("="*70)
    print(df[['config_id', 'learning_rate', 'gamma', 'n_steps', 'batch_size',
              'n_epochs', 'clip_range', 'ent_coef', 'mean_reward']].to_string(index=False))

    best_config_id = int(df.iloc[0]['config_id'])
    best_result = [r for r in results if r['config_id'] == best_config_id][0]

    print("\n" + "="*70)
    print("BEST CONFIGURATION:")
    print("="*70)
    for key in ['config_id', 'learning_rate', 'gamma', 'n_steps', 'batch_size', 'n_epochs', 'clip_range', 'ent_coef', 'mean_reward', 'test_reward']:
        print(f"{key}: {best_result[key]}")

    if best_result['model_path']:
        import shutil
        best_model_path = "models/ppo/ubuntu_agent_ppo_best"
        shutil.copy(f"{best_result['model_path']}.zip", f"{best_model_path}.zip")
        print(f"\n✅ Best model saved to: {best_model_path}.zip")

    print(f"\n📊 Full results saved to: {csv_path}")

    print("\n" + "="*70)
    print("GENERATING ANALYSIS PLOTS...")
    print("="*70)

    plot_cumulative_rewards(results, save_path="models/ppo/plots/")

    print("\n✅ All done!")
    print("="*70)

    return df, results


if __name__ == "__main__":
    try:
        results_df, all_results = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()