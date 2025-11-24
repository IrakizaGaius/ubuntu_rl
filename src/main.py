#!/usr/bin/env python3
"""Play UbuntuEnv with trained model and Pygame visualization."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import UbuntuEnv


def try_load_model(path: str):
    """Try loading model with SB3 algorithms."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    
    # Try algorithms based on file path hint
    if 'dqn' in path.lower():
        try:
            return DQN.load(path), "DQN"
        except Exception as e:
            print(f"Failed to load as DQN: {e}")
    
    if 'ppo' in path.lower() or 'pg' in path.lower():
        try:
            return PPO.load(path), "PPO"
        except Exception as e:
            print(f"Failed to load as PPO: {e}")
    
    if 'a2c' in path.lower() or 'actor' in path.lower() or 'reinforce' in path.lower():
        try:
            return A2C.load(path), "A2C"
        except Exception as e:
            print(f"Failed to load as A2C: {e}")
    
    # Try all if path doesn't give hints
    for name, cls in [("PPO", PPO), ("DQN", DQN), ("A2C", A2C)]:
        try:
            return cls.load(path), name
        except Exception:
            continue
    
    raise RuntimeError(f"Failed to load model from {path} with any algorithm")


def play_episode_simple(model, env, episode_num: int, verbose: bool = True, render: bool = False):
    """Play a single episode with optional pygame rendering."""
    CELL_EMOJIS = {'goal': '🏛️', 'good': '💚', 'bad': '⚠️', 'empty': '🌳', 'start': '🏁'}
    REASON_EMOJIS = {
        'ubuntu_mastered': '👑 UBUNTU MASTERED!', 'goal_reached': '🏆 Goal Reached!',
        'too_many_bad_choices': '💔 Lost the Way...', 'stuck_oscillating': '🔄 Got Stuck',
        'max_steps': '⏱️  Time Limit'
    }
    STATE_EMOJIS = {'UBUNTU_MASTER': '👑', 'UBUNTU_LEARNER': '📈', 'ANTI_UBUNTU': '⚠️', 'LEARNING': '🎓'}
    ACTION_NAMES = ['LEFT ⬅️', 'RIGHT ➡️', 'UP ⬆️', 'DOWN ⬇️']
    
    if verbose:
        print(f"\n{'='*60}\n🎮 Episode {episode_num} - Ubuntu Society Journey\n{'='*60}")
    
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    
    if render:
        env.render()
        time.sleep(1.0)
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        
        if verbose:
            pos = env.grid_pos
            cell = env.grid[tuple(pos)]
            cell_emoji = CELL_EMOJIS.get(cell['type'], '❓')
            print(f"\n   Step {episode_length + 1}:")
            print(f"     📍 Position: ({pos[0]}, {pos[1]}) {cell_emoji}")
            print(f"     🎯 Current location: {cell['philosophy']} ({cell['type']})")
            print(f"     🚶 Moving: {ACTION_NAMES[action]}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if render:
            env.render()
            time.sleep(0.3)
        
        episode_reward += reward
        episode_length += 1
        
        if verbose:
            new_cell = env.grid[tuple(env.grid_pos)]
            if new_cell['type'] == 'good' and info.get('cell_type') == 'good':
                print(f"     ✨ Positive interaction! {info.get('philosophy', '')} +{info.get('cell_reward', 0)}")
            elif new_cell['type'] == 'bad' and info.get('cell_type') == 'bad':
                print(f"     ⚠️  Negative encounter! {info.get('philosophy', '')} {info.get('cell_reward', 0)}")
            elif new_cell['type'] == 'goal':
                print(f"     🏛️ Reached Ubuntu Temple!")
            print(f"     📊 Reward: {reward:+.2f} | Ubuntu Score: {env.ubuntu_score:.1f}")
        
        if terminated or truncated or episode_length >= 500:
            if verbose and (terminated or truncated):
                reason = REASON_EMOJIS.get(info.get('reason', 'unknown'), info.get('reason', 'unknown'))
                print(f"\n   🏁 Episode ended: {reason}")
            break
    
    if verbose:
        ubuntu_state = info.get('ubuntu_state', 'UNKNOWN')
        state_emoji = STATE_EMOJIS.get(ubuntu_state, '❓')
        progress = min(env.ubuntu_score / env.ubuntu_threshold * 100, 100)
        
        print(f"\n📊 Episode Summary:")
        print(f"   🎯 Total Reward: {episode_reward:.1f}")
        print(f"   👣 Journey Length: {episode_length} steps")
        print(f"   🌟 Ubuntu Score: {info.get('ubuntu_score', 0):.1f} / 100")
        print(f"   {state_emoji} Ubuntu State: {ubuntu_state}")
        print(f"   💚 Good actions: {env.positive_actions} | 💔 Bad actions: {env.negative_actions}")
        print(f"   📈 Progress: {progress:.1f}% toward Ubuntu Mastery")
        print(f"\n   🎓 COMPREHENSIVE EVALUATION (New A+ Metrics):")
        print(f"      🌈 Diversity Score: {info.get('diversity_score', 0):.3f} ({info.get('philosophies_encountered', 0)}/10 philosophies)")
        print(f"      ✅ Diversity Met: {'YES ✓' if info.get('diversity_met', False) else 'NO ✗'} (need 7/10)")
        print(f"      🎭 Consistency: {info.get('consistency_score', 0):.3f} (moral coherence)")
        print(f"      🛡️  Harm Minimization: {info.get('harm_minimization_score', 0):.3f} (ethics)")
        print(f"      🏆 Composite Ubuntu Score: {info.get('composite_ubuntu_score', 0):.3f}/1.0")
    
    return {
        'reward': episode_reward,
        'length': episode_length,
        'ubuntu_score': info.get('ubuntu_score', 0),
        'ubuntu_state': info.get('ubuntu_state', 'UNKNOWN')
    }

def main():
    parser = argparse.ArgumentParser(description="Play UbuntuEnv with trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model (.zip)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--render", action="store_true", help="Enable Pygame GUI visualization")
    args = parser.parse_args()
    
    print(f"\n{'='*80}\n🏛️  UBUNTU SOCIETY - Journey to Ubuntu Mastery\n{'='*80}")
    print(f"📦 Loading AI agent from: {args.model_path}")
    model, model_name = try_load_model(args.model_path)
    print(f"Loaded {model_name} model successfully")
    
    render_mode = "human" if args.render else None
    
    env = UbuntuEnv(render_mode=render_mode, vision_radius=2, enable_history=True)
    
    if args.render:
        print(f"\n   Visualization enabled!")
    else:
        print(f"\n   📝 Console mode (add --render for visual experience)")
    
    try:
        all_stats = []
        for ep in range(1, args.episodes + 1):
            stats = play_episode_simple(model, env, ep, verbose=True, render=args.render)
            all_stats.append(stats)
        
        if len(all_stats) > 1:
            rewards = [s['reward'] for s in all_stats]
            lengths = [s['length'] for s in all_stats]
            ubuntu_scores = [s['ubuntu_score'] for s in all_stats]
            ubuntu_masters = sum(1 for s in all_stats if s['ubuntu_state'] == 'UBUNTU_MASTER')
            ubuntu_learners = sum(1 for s in all_stats if s['ubuntu_state'] == 'UBUNTU_LEARNER')
            
            print(f"\n{'='*80}\n🏆 Overall Journey Results\n{'='*80}")
            print(f"   📊 Episodes Completed: {len(all_stats)}")
            print(f"\n   🎯 Performance Metrics:")
            print(f"      Reward    - mean: {np.mean(rewards):>7.1f}  std: {np.std(rewards):>6.1f}  min: {np.min(rewards):>7.1f}  max: {np.max(rewards):>7.1f}")
            print(f"      Length    - mean: {np.mean(lengths):>7.1f}  min: {np.min(lengths):>7.0f}  max: {np.max(lengths):>7.0f}")
            print(f"      Ubuntu    - mean: {np.mean(ubuntu_scores):>7.1f}  std: {np.std(ubuntu_scores):>6.1f}")
            print(f"\n   🌟 Ubuntu Mastery Achieved:")
            print(f"      👑 Ubuntu Masters: {ubuntu_masters}/{len(all_stats)} ({ubuntu_masters/len(all_stats)*100:.1f}%)")
            print(f"      📈 Ubuntu Learners: {ubuntu_learners}/{len(all_stats)} ({ubuntu_learners/len(all_stats)*100:.1f}%)")
            print(f"{'='*80}")
    finally:
        env.close()
    
    print("\nDone!")


if __name__ == "__main__":
    main()