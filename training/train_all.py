"""
Train All Four RL Algorithms and Compare Results
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dqn_training import train_dqn
from training.pg_training import train_ppo
from training.reinforce_training import train_reinforce
from training.actor_critic_training import train_a2c
import numpy as np
import time


def train_all_algorithms(timesteps=200000):
    """
    Train all four RL algorithms on the same environment
    
    Algorithms:
    1. DQN (Value-Based)
    2. PPO (Policy Gradient - Proximal Policy Optimization)
    3. REINFORCE (Policy Gradient - Monte Carlo)
    4. A2C (Actor-Critic)
    """
    print("\n" + "=" * 70)
    print("🚀 TRAINING ALL FOUR RL ALGORITHMS")
    print("=" * 70)
    print("\nThis will train the following algorithms:")
    print("  1. DQN           - Deep Q-Network (Value-Based)")
    print("  2. PPO           - Proximal Policy Optimization (Policy Gradient)")
    print("  3. REINFORCE     - Monte Carlo Policy Gradient")
    print("  4. A2C           - Advantage Actor-Critic")
    print("\nAll models will train on the same Ubuntu RL environment.")
    print(f"Training timesteps per model: {timesteps:,}")
    print("=" * 70)
    
    results = {}
    
    # Train DQN
    print("\n\n" + "🔷" * 35)
    print("ALGORITHM 1/4: DQN (VALUE-BASED)")
    print("🔷" * 35)
    start_time = time.time()
    try:
        model_dqn, callback_dqn = train_dqn(total_timesteps=timesteps)
        results['DQN'] = {
            'final_reward': np.mean(callback_dqn.episode_rewards[-20:]) if len(callback_dqn.episode_rewards) >= 20 else np.mean(callback_dqn.episode_rewards),
            'total_episodes': len(callback_dqn.episode_rewards),
            'training_time': time.time() - start_time,
            'all_rewards': callback_dqn.episode_rewards
        }
    except Exception as e:
        print(f"❌ DQN training failed: {e}")
        results['DQN'] = {'error': str(e)}
    
    # Train PPO
    print("\n\n" + "🟢" * 35)
    print("ALGORITHM 2/4: PPO (POLICY GRADIENT)")
    print("🟢" * 35)
    start_time = time.time()
    try:
        model_ppo, callback_ppo = train_ppo(total_timesteps=timesteps)
        results['PPO'] = {
            'final_reward': np.mean(callback_ppo.episode_rewards[-20:]) if len(callback_ppo.episode_rewards) >= 20 else np.mean(callback_ppo.episode_rewards),
            'total_episodes': len(callback_ppo.episode_rewards),
            'training_time': time.time() - start_time,
            'all_rewards': callback_ppo.episode_rewards
        }
    except Exception as e:
        print(f"❌ PPO training failed: {e}")
        results['PPO'] = {'error': str(e)}
    
    # Train REINFORCE
    print("\n\n" + "🟡" * 35)
    print("ALGORITHM 3/4: REINFORCE (POLICY GRADIENT)")
    print("🟡" * 35)
    start_time = time.time()
    try:
        model_reinforce, callback_reinforce = train_reinforce(total_timesteps=timesteps)
        results['REINFORCE'] = {
            'final_reward': np.mean(callback_reinforce.episode_rewards[-20:]) if len(callback_reinforce.episode_rewards) >= 20 else np.mean(callback_reinforce.episode_rewards),
            'total_episodes': len(callback_reinforce.episode_rewards),
            'training_time': time.time() - start_time,
            'all_rewards': callback_reinforce.episode_rewards
        }
    except Exception as e:
        print(f"❌ REINFORCE training failed: {e}")
        results['REINFORCE'] = {'error': str(e)}
    
    # Train A2C
    print("\n\n" + "🔵" * 35)
    print("ALGORITHM 4/4: A2C (ACTOR-CRITIC)")
    print("🔵" * 35)
    start_time = time.time()
    try:
        model_a2c, callback_a2c = train_a2c(total_timesteps=timesteps)
        results['A2C'] = {
            'final_reward': np.mean(callback_a2c.episode_rewards[-20:]) if len(callback_a2c.episode_rewards) >= 20 else np.mean(callback_a2c.episode_rewards),
            'total_episodes': len(callback_a2c.episode_rewards),
            'training_time': time.time() - start_time,
            'all_rewards': callback_a2c.episode_rewards
        }
    except Exception as e:
        print(f"❌ A2C training failed: {e}")
        results['A2C'] = {'error': str(e)}
    
    # Print comparison
    print("\n\n" + "=" * 70)
    print("📊 TRAINING RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Algorithm':<15} {'Avg Reward':<12} {'Episodes':<12} {'Time (s)':<12} {'Status'}")
    print("-" * 70)
    
    for algo_name in ['DQN', 'PPO', 'REINFORCE', 'A2C']:
        result = results[algo_name]
        if 'error' in result:
            print(f"{algo_name:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} ❌ Failed")
        else:
            print(f"{algo_name:<15} {result['final_reward']:>11.2f} {result['total_episodes']:>11} {result['training_time']:>11.1f} ✅ Success")
    
    # Find best algorithm
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_algo = max(valid_results.items(), key=lambda x: x[1]['final_reward'])
        print("\n" + "=" * 70)
        print(f"🏆 BEST PERFORMING ALGORITHM: {best_algo[0]}")
        print(f"   Final Reward: {best_algo[1]['final_reward']:.2f}")
        print(f"   Total Episodes: {best_algo[1]['total_episodes']}")
        print(f"   Training Time: {best_algo[1]['training_time']:.1f}s")
        print("=" * 70)
    
    print("\n✨ All training complete!")
    print("\n📁 Trained models saved to:")
    print("   - models/dqn/ubuntu_agent_dqn.zip")
    print("   - models/pg/ubuntu_agent_ppo.zip")
    print("   - models/pg/ubuntu_agent_reinforce.zip")
    print("   - models/actor_critic/ubuntu_agent_a2c.zip")
    print("\n💡 Run 'python main.py' to visualize the best model!")
    print("💡 Run 'python compare_models.py' for detailed comparison!")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train all RL algorithms")
    parser.add_argument('--timesteps', type=int, default=200000, 
                       help='Training timesteps per algorithm (default: 200000)')
    args = parser.parse_args()
    
    train_all_algorithms(timesteps=args.timesteps)
