"""
DQN Training Script for Ubuntu RL Environment
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from environment.custom_env import UbuntuEnv
import numpy as np


class ProgressCallback(BaseCallback):
    """Callback for tracking training progress with early stopping"""
    
    def __init__(self, check_freq=1000, patience=20, min_improvement=10):
        super().__init__()
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.patience = patience  # Stop after this many checks without improvement
        self.min_improvement = min_improvement  # Minimum reward improvement to reset patience
        self.best_reward = float('-inf')
        self.patience_counter = 0
        
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer) > len(self.episode_rewards):
            for info in self.model.ep_info_buffer:
                if 'r' in info and 'l' in info:
                    if len(self.episode_rewards) == 0 or info['r'] != self.episode_rewards[-1]:
                        self.episode_rewards.append(info['r'])
                        self.episode_lengths.append(info['l'])
        
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards[-10:])
            print(f"{self.n_calls:7d} steps | {len(self.episode_rewards)} eps | Reward: {avg_reward:+.2f}")
            
            # Early stopping check
            if avg_reward > self.best_reward + self.min_improvement:
                self.best_reward = avg_reward
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"Early stopping: No improvement for {self.patience} checks (best: {self.best_reward:.2f})")
                return False
        
        return True


def train_dqn(total_timesteps=300000, model_save_path="models/dqn/ubuntu_agent_dqn"):
    """
    Train DQN agent on UPGRADED A+ Ubuntu RL environment
    
    Args:
        total_timesteps: Total training timesteps
        model_save_path: Path to save trained model
    """
    print("🌍 TRAINING DQN AGENT (A+ Upgraded Environment)")
    env = UbuntuEnv(render_mode=None, vision_radius=2, enable_history=True)
    print(f"Actions: {env.action_space} | Obs: {env.observation_space.shape}")
    print(f"   👀 Vision: {2*env.vision_radius+1}x{2*env.vision_radius+1} grid")
    print(f"   🧠 History: {env.history_length} steps")
    print(f"   🎯 Diversity requirement: {env.required_diversity}/10 philosophies")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0003,  # Higher learning rate
        buffer_size=100000,  # Larger buffer
        learning_starts=2000,  # More initial exploration
        batch_size=64,  # Larger batch
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,  # Longer exploration
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,  # More final exploration
        max_grad_norm=10,
        verbose=0
    )
    
    print(f"Training {total_timesteps:,} steps...")
    callback = ProgressCallback(check_freq=1000, patience=30, min_improvement=5)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False
    )
    
    model.save(model_save_path)
    print(f"✅ Episodes: {len(callback.episode_rewards)} | Avg reward: {np.mean(callback.episode_rewards[-20:]):.2f}")
    print("Testing...")
    
    try:
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"Test: {steps} steps | Reward: {total_reward:.1f} | Ubuntu Score: {env.ubuntu_score:.1f}")
    except Exception as e:
        print(f"Test failed (non-critical): {e}")
    
    print(f"Saved: {model_save_path}.zip")
    
    return model, callback


if __name__ == "__main__":
    train_dqn()
