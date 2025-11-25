import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from Configurations.config import *
except Exception:
    ENV_SIZE = 9.0
    MAX_EPISODE_STEPS = 500


class Entity:
    def __init__(self, position, entity_type):
        self.position = np.array(position, dtype=np.float32)
        self.entity_type = entity_type


class UbuntuEnv(gym.Env):
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, vision_radius=2, enable_history=True):
        super().__init__()
        self.render_mode = render_mode
        self.vision_radius = vision_radius  # Strategic planning: see ahead
        self.enable_history = enable_history  # Memory for LSTM
        self.history_length = 5  # Last 5 steps
        
        self.action_space = spaces.Discrete(4)
        
        vision_dim = (2 * vision_radius + 1) ** 2 * 10  # 5×5 grid, 10 philosophy types one-hot
        base_dim = 6 + 5 + 3  # 14 total
        self.base_obs_dim = vision_dim + base_dim
        
        if enable_history:
            # Include history as flattened sequence
            obs_dim = self.base_obs_dim + (self.history_length * self.base_obs_dim)
        else:
            obs_dim = self.base_obs_dim
            
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        self.window = None
        self.clock = None
        self.window_size = 800
        
        self.max_speed = 0.5
        self.friction = 0.85
        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.last_direction = None
        self.direction_persistence = 0
        self.furthest_x = -8.0
        
        self.ubuntu_score = 0.0
        self.negative_actions = 0
        self.positive_actions = 0
        self.ubuntu_threshold = 60
        self.violence_limit = 5
        
        # Philosophy types for semantic encoding
        self.good_morals = ['help_needy', 'share_knowledge', 'show_compassion', 'build_community', 'forgive']
        self.bad_morals = ['selfish_act', 'harm_other', 'hoard_resources', 'show_greed', 'violence']
        self.all_philosophies = self.good_morals + self.bad_morals
        self.philosophy_to_idx = {p: i for i, p in enumerate(self.all_philosophies)}
        
        # Robustness: diversity tracking
        self.encountered_philosophies = set()
        self.required_diversity = 7  # Must encounter 7/10 philosophy types
        
        # History for memory/planning
        self.observation_history = []
        
        # Evaluation metrics
        self.diversity_score = 0.0
        self.consistency_score = 0.0
        self.harm_minimization_score = 0.0
        self.composite_ubuntu_score = 0.0
        
        self.particles = []
        self.agent_color = [1.0, 1.0, 1.0, 1.0]
        
        # Dynamic philosophy spawning counter
        self.dynamic_spawn_chance = 0.05  # 5% per step
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid_size = 9
        self.cell_size = 1.0
        self.grid = {}
        good_morals = self.good_morals
        bad_morals = self.bad_morals
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                near_goal = x >= (self.grid_size - 2)
                rand = random.random()
                if near_goal:
                    if rand < 0.55:
                        self.grid[(x, y)] = {'type': 'empty', 'philosophy': 'neutral', 'reward': 0, 'visited': False}
                    else:
                        philosophy = random.choice(good_morals)
                        self.grid[(x, y)] = {
                            'type': 'good',
                            'philosophy': philosophy,
                            'reward': float(self._get_philosophy_reward_value(philosophy)),
                            'visited': False
                        }
                else:
                    if rand < 0.55:
                        self.grid[(x, y)] = {'type': 'empty', 'philosophy': 'neutral', 'reward': 0, 'visited': False}
                    elif rand < 0.90:
                        philosophy = random.choice(good_morals)
                        self.grid[(x, y)] = {
                            'type': 'good',
                            'philosophy': philosophy,
                            'reward': float(self._get_philosophy_reward_value(philosophy)),
                            'visited': False
                        }
                    else:
                        philosophy = random.choice(bad_morals)
                        self.grid[(x, y)] = {
                            'type': 'bad',
                            'philosophy': philosophy,
                            'reward': float(self._get_philosophy_reward_value(philosophy)),
                            'visited': False
                        }

        self.start_pos = (0, self.grid_size // 2)
        self.grid[self.start_pos] = {'type': 'empty', 'philosophy': 'start', 'reward': 0.0, 'visited': True}

        self.goal_pos = (self.grid_size - 1, self.grid_size // 2)
        self.grid[self.goal_pos] = {'type': 'goal', 'philosophy': 'ubuntu_mastery', 'reward': 100.0, 'visited': False}

        # Reset agent state
        self.grid_pos = [int(self.start_pos[0]), int(self.start_pos[1])]
        self.position = np.array([self.grid_pos[0] * self.cell_size, self.grid_pos[1] * self.cell_size], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.ubuntu_score = 0.0
        self.negative_actions = 0
        self.positive_actions = 0
        self.steps = 0
        self.agent_color = [1.0, 1.0, 1.0, 1.0]
        self.particles = []
        self.ubuntu_state = 'LEARNING'
        self.decision_history = []
        self.last_action = None
        self.last_position = list(self.start_pos)
        self.repeated_moves = 0
        self.furthest_progress = 0
        self.position_history = []
        self.stuck_counter = 0
        self.last_dist_to_goal = float(self.grid_size * 2)
        
        # Reset robustness & evaluation tracking
        self.encountered_philosophies = set()
        self.observation_history = []
        self.diversity_score = 0.0
        self.consistency_score = 0.0
        self.harm_minimization_score = 0.0
        self.composite_ubuntu_score = 0.0

        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self):
        """Generate sanitized observation with vision, semantics, and history."""

        vision_obs = []
        for dy in range(-self.vision_radius, self.vision_radius + 1):
            for dx in range(-self.vision_radius, self.vision_radius + 1):
                x = int(self.grid_pos[0] + dx)
                y = int(self.grid_pos[1] + dy)

                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    cell = self.grid[(x, y)]
                    philosophy = cell['philosophy']

                    one_hot = np.zeros(10, dtype=np.float32)
                    if philosophy in self.philosophy_to_idx:
                        one_hot[int(self.philosophy_to_idx[philosophy])] = 1.0
                    elif philosophy in ['start', 'neutral']:
                        pass
                    elif philosophy == 'ubuntu_mastery':
                        one_hot[:5] = 0.5
                else:
                    one_hot = np.zeros(10, dtype=np.float32)

                vision_obs.extend(one_hot)

        vision_obs = np.array(vision_obs, dtype=np.float32)

        pos_x_norm = float(self.grid_pos[0]) / float(self.grid_size - 1)
        pos_y_norm = float(self.grid_pos[1]) / float(self.grid_size - 1)
        ubuntu_norm = float(np.clip(self.ubuntu_score / 300.0, -1.0, 1.0))
        pos_norm = float(np.clip(self.positive_actions / 10.0, 0.0, 1.0))
        neg_norm = float(np.clip(self.negative_actions / 10.0, 0.0, 1.0))
        dist_to_goal = abs(self.grid_pos[0] - self.goal_pos[0]) + abs(self.grid_pos[1] - self.goal_pos[1])
        dist_norm = float(dist_to_goal / float(self.grid_size * 2))

        state_encoding = np.zeros(3, dtype=np.float32)
        if self.ubuntu_state == 'LEARNING':
            state_encoding[0] = 1.0
        elif self.ubuntu_state == 'UBUNTU_LEARNER':
            state_encoding[1] = 1.0
        elif self.ubuntu_state == 'UBUNTU_MASTER':
            state_encoding[2] = 1.0
        elif self.ubuntu_state == 'ANTI_UBUNTU':
            state_encoding[2] = -1.0

        self.diversity_score = float(len(self.encountered_philosophies) / 10.0)
        self.consistency_score = float(np.clip(1.0 - (self.negative_actions / max(self.positive_actions + 1, 1)), 0.0, 1.0))
        self.harm_minimization_score = float(1.0 / (1.0 + self.negative_actions))

        eval_obs = np.array([self.diversity_score, self.consistency_score, self.harm_minimization_score], dtype=np.float32)

        base_obs = np.concatenate([
            vision_obs,
            np.array([pos_x_norm, pos_y_norm, ubuntu_norm, pos_norm, neg_norm, dist_norm], dtype=np.float32),
            state_encoding,
            np.array([pos_norm, neg_norm], dtype=np.float32),  # compatibility
            eval_obs
        ])

        if self.enable_history:
            self.observation_history.append(base_obs)
            if len(self.observation_history) > self.history_length:
                self.observation_history.pop(0)
            history_padding = [np.zeros_like(base_obs) for _ in range(self.history_length - len(self.observation_history))]
            full_history = history_padding + self.observation_history
            history_obs = np.concatenate(full_history)
            obs = np.concatenate([base_obs, history_obs])
        else:
            obs = base_obs

        return obs.astype(np.float32)

    def step(self, action):
        # FIX: Convert action to Python int immediately
        # Stable Baselines3 sometimes passes actions as numpy arrays or numpy scalars
        if isinstance(action, np.ndarray):
            action = int(action.item())
        elif isinstance(action, (np.integer, np.int32, np.int64)):
            action = int(action)
        else:
            action = int(action)
        
        self.steps += 1
        reward = 0.0
        info = {}
        truncated = False
        terminated = False

        # Dynamic philosophy spawning
        if random.random() < self.dynamic_spawn_chance:
            self._spawn_dynamic_philosophy()

        action_names = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

        info['action_type'] = action_names[action]
        info['grid_pos'] = tuple(self.grid_pos)

        dx, dy = directions[action]
        new_x = self.grid_pos[0] + dx
        new_y = self.grid_pos[1] + dy

        # Track repeated moves
        if self.last_action is not None:
            opposite_actions = {0: 1, 1: 0, 2: 3, 3: 2}
            if action == opposite_actions.get(self.last_action, -1):
                self.repeated_moves += 1
            else:
                self.repeated_moves = 0

        self.position_history.append(tuple(self.grid_pos))
        if len(self.position_history) > 10:
            self.position_history.pop(0)

        unique_positions = set()
        if len(self.position_history) >= 10:
            unique_positions = set(self.position_history[-10:])
            if len(unique_positions) <= 4:
                self.stuck_counter += 1
            if self.grid_pos[0] >= self.grid_size - 2:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            old_pos = tuple(self.grid_pos)
            self.grid_pos = [new_x, new_y]
            new_pos = tuple(self.grid_pos)
            self.position = np.array([new_x * self.cell_size, new_y * self.cell_size], dtype=np.float32)

            if new_x > self.furthest_progress:
                self.furthest_progress = new_x
                reward += 2.0

            if self.repeated_moves >= 2:
                reward -= 1.5
            if self.grid_pos[0] >= self.grid_size - 2:
                reward -= 2.0

            stuck_threshold = 5 if self.grid_pos[0] < self.grid_size - 2 else 3
            if self.stuck_counter >= stuck_threshold:
                reward -= 3.0
                info['stuck'] = True

            if self.stuck_counter > 0 and len(unique_positions) > 4:
                reward += 1.0

            self.last_action = action
            self.last_position = list(old_pos)
            cell = self.grid[new_pos]

            # Record move
            self.decision_history.append({
                'from': old_pos,
                'to': new_pos,
                'action': action_names[action],
                'cell_type': cell['type'],
                'philosophy': cell['philosophy']
            })

            # Apply rewards based on cell type
            if not cell['visited']:
                cell['visited'] = True

                if cell['type'] == 'good':
                    reward += float(cell['reward']) * 2.0
                    self.ubuntu_score += float(cell['reward'])
                    self.positive_actions += 1
                    self.encountered_philosophies.add(cell['philosophy'])
                    info.update({
                        'cell_type': 'good',
                        'philosophy': cell['philosophy'],
                        'cell_reward': float(cell['reward'])
                    })
                    self._create_particles(self.position, 'green', 20)

                elif cell['type'] == 'bad':
                    reward += float(cell['reward'])
                    self.ubuntu_score += float(cell['reward'])
                    self.negative_actions += 1
                    self.encountered_philosophies.add(cell['philosophy'])
                    info.update({
                        'cell_type': 'bad',
                        'philosophy': cell['philosophy'],
                        'cell_reward': float(cell['reward'])
                    })
                    self._create_particles(self.position, 'red', 20)

                    if self.negative_actions >= self.violence_limit:
                        terminated = True
                        reward -= 200.0
                        info['reason'] = 'too_many_bad_choices'
                        self.ubuntu_state = 'ANTI_UBUNTU'

                elif cell['type'] == 'goal':
                    terminated = True
                    reward += float(cell['reward'])
                    self.composite_ubuntu_score = self._compute_composite_ubuntu_score()
                    diversity_met = len(self.encountered_philosophies) >= self.required_diversity

                    if self.ubuntu_score >= self.ubuntu_threshold and diversity_met:
                        reward += 300.0
                        info['reason'] = 'ubuntu_mastered'
                        self.ubuntu_state = 'UBUNTU_MASTER'
                    elif self.ubuntu_score >= self.ubuntu_threshold:
                        reward += 150.0
                        info['reason'] = 'goal_reached_insufficient_diversity'
                        self.ubuntu_state = 'UBUNTU_LEARNER'
                    else:
                        reward += 100.0
                        info['reason'] = 'goal_reached'
                        self.ubuntu_state = 'UBUNTU_LEARNER'

                    info['cell_type'] = 'goal'
                    self._create_particles(self.position, 'gold', 40)

                elif cell['type'] == 'empty':
                    reward -= 0.05
                    info['cell_type'] = 'empty'

                else:
                    reward -= 0.5
                    info['cell_type'] = cell['type']
                    info['visited'] = True

        else:
            reward -= 2.0
            info['wall_collision'] = True
            if self.grid_pos[0] >= self.grid_size - 2:
                reward -= 2.0
            self.stuck_counter += 2

        # Update ubuntu state
        if not terminated:
            if self.ubuntu_score >= self.ubuntu_threshold:
                self.ubuntu_state = 'UBUNTU_LEARNER'
                self.agent_color = [0.5, 1.0, 0.5, 1.0]
            elif self.ubuntu_score >= self.ubuntu_threshold // 2:
                self.ubuntu_state = 'UBUNTU_LEARNER'
                self.agent_color = [0.5, 1.0, 0.5, 1.0]
            elif self.ubuntu_score < 0:
                self.ubuntu_state = 'ANTI_UBUNTU'
                self.agent_color = [1.0, 0.0, 0.0, 1.0]
            else:
                self.ubuntu_state = 'LEARNING'
                self.agent_color = [1.0, 1.0, 1.0, 1.0]

        # Update particles
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['position'] = p['position'] + p['velocity']
            p['lifetime'] -= 1
            p['velocity'] *= 0.95

        # Distance-based rewards
        dist_to_goal = abs(self.grid_pos[0] - self.goal_pos[0]) + abs(self.grid_pos[1] - self.goal_pos[1])
        progress_ratio = self.grid_pos[0] / float(self.grid_size - 1)
        reward += progress_ratio * 1.0
        if dist_to_goal <= 3:
            reward += (4 - dist_to_goal) * 2.0
        if hasattr(self, 'last_dist_to_goal') and dist_to_goal < self.last_dist_to_goal:
            reward += 0.5
        self.last_dist_to_goal = dist_to_goal

        if self.stuck_counter >= 15:
            terminated = True
            reward -= 50.0
            info['reason'] = 'stuck_oscillating'

        if self.steps >= MAX_EPISODE_STEPS:
            terminated = True
            info['reason'] = 'max_steps'
            reward += 30.0 if self.ubuntu_score > 0 else -30.0

        # Add evaluation metrics
        info.update({
            'ubuntu_score': float(self.ubuntu_score),
            'ubuntu_state': self.ubuntu_state,
            'distance_to_goal': int(dist_to_goal),
            'diversity_score': float(self.diversity_score),
            'consistency_score': float(self.consistency_score),
            'harm_minimization_score': float(self.harm_minimization_score),
            'composite_ubuntu_score': float(self.composite_ubuntu_score),
            'philosophies_encountered': int(len(self.encountered_philosophies)),
            'diversity_met': bool(len(self.encountered_philosophies) >= self.required_diversity)
        })

        # SANITIZE INFO: Convert any remaining numpy scalars to Python types
        for k, v in list(info.items()):
            if isinstance(v, (np.float32, np.float64)):
                info[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                info[k] = int(v)
            elif isinstance(v, np.ndarray):
                info[k] = v.tolist()
            elif isinstance(v, tuple):
                # Sanitize tuples that might contain numpy types
                info[k] = tuple(int(x) if isinstance(x, (np.integer,)) else 
                               float(x) if isinstance(x, (np.floating,)) else x 
                               for x in v)

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    
    def _get_philosophy_reward_value(self, philosophy_type):
        """Get reward value for a philosophy type"""
        ubuntu_philosophies = {
            # Positive Ubuntu actions
            'help_needy': 15,
            'share_knowledge': 12,
            'show_compassion': 10,
            'build_community': 8,
            'forgive': 6,
            # Negative anti-Ubuntu actions
            'selfish_act': -12,
            'harm_other': -15,
            'hoard_resources': -10,
            'show_greed': -8,
            'violence': -50,
        }
        return ubuntu_philosophies.get(philosophy_type, 0)
    
    def _compute_composite_ubuntu_score(self):
        """COMPREHENSIVE EVALUATION: Compute composite Ubuntu mastery score."""
        # Weighted combination of multiple factors
        ubuntu_component = np.clip(self.ubuntu_score / 100.0, 0.0, 1.0) * 0.4
        diversity_component = self.diversity_score * 0.3
        consistency_component = self.consistency_score * 0.2
        harm_component = self.harm_minimization_score * 0.1
        
        composite = ubuntu_component + diversity_component + consistency_component + harm_component
        return composite
    
    def _spawn_dynamic_philosophy(self):
        """ROBUSTNESS: Spawn new philosophy encounter dynamically during episode."""
        # Find empty cells
        empty_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                       if self.grid[(x, y)]['type'] == 'empty' and (x, y) != tuple(self.grid_pos)]
        
        if empty_cells:
            x, y = random.choice(empty_cells)
            philosophy = random.choice(self.all_philosophies)
            phil_type = 'good' if philosophy in self.good_morals else 'bad'
            self.grid[(x, y)] = {
                'type': phil_type,
                'philosophy': philosophy,
                'reward': self._get_philosophy_reward_value(philosophy),
                'visited': False
            }
    
    def _create_particles(self, position, color, count):
        color_map = {'green': [0, 255, 0], 'red': [255, 0, 0], 'gold': [255, 215, 0]}
        rgb = color_map.get(color, [255, 255, 255])
        for _ in range(count):
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(0.05, 0.3)
            self.particles.append({
                'position': np.array(position, dtype=np.float32).copy(),
                'velocity': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'color': rgb,
                'lifetime': np.random.randint(10, 30)
            })

    def render(self):
        if self.render_mode is None:
            return None
            
        import pygame
        
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("Ubuntu Society - Journey to Ubuntu Mastery")
            else:
                self.window = pygame.Surface((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        
        for y in range(self.window_size):
            if y < self.window_size * 0.6:
                ratio = y / (self.window_size * 0.6)
                r = int(135 + (100 - 135) * ratio)
                g = int(206 + (150 - 206) * ratio)
                b = int(235 + (200 - 235) * ratio)
            else:
                ratio = (y - self.window_size * 0.6) / (self.window_size * 0.4)
                r = int(144 - 44 * ratio)
                g = int(238 - 68 * ratio)
                b = int(144 - 44 * ratio)
            pygame.draw.line(canvas, (r, g, b), (0, y), (self.window_size, y))
        
        cell_pixels = self.window_size // self.grid_size
        
        # Draw sun
        sun_x = self.window_size - 80
        sun_y = 60
        pygame.draw.circle(canvas, (255, 220, 100), (sun_x, sun_y), 35)
        pygame.draw.circle(canvas, (255, 240, 150), (sun_x, sun_y), 25)
        
        # Draw clouds
        cloud_positions = [(150, 80), (400, 100), (650, 70)]
        for cx, cy in cloud_positions:
            # Simple clouds using circles
            pygame.draw.circle(canvas, (255, 255, 255, 200), (cx, cy), 25)
            pygame.draw.circle(canvas, (255, 255, 255, 200), (cx + 20, cy), 30)
            pygame.draw.circle(canvas, (255, 255, 255, 200), (cx + 45, cy), 25)
            pygame.draw.circle(canvas, (255, 255, 255, 200), (cx + 30, cy - 15), 20)
        
        # Draw path/roads connecting the community
        path_color = (210, 180, 140)  # Tan path color
        for gx in range(self.grid_size):
            for gy in range(self.grid_size):
                px = gx * cell_pixels + cell_pixels // 2
                py = gy * cell_pixels + cell_pixels // 2
                
                # Draw paths to neighbors
                if gx < self.grid_size - 1:
                    next_px = (gx + 1) * cell_pixels + cell_pixels // 2
                    pygame.draw.line(canvas, path_color, (px, py), (next_px, py), 4)
                if gy < self.grid_size - 1:
                    next_py = (gy + 1) * cell_pixels + cell_pixels // 2
                    pygame.draw.line(canvas, path_color, (px, py), (px, next_py), 4)
        
        # Draw community buildings, trees, and people
        font_label = pygame.font.Font(None, 14)
        font_bold = pygame.font.Font(None, 18)
        
        for gx in range(self.grid_size):
            for gy in range(self.grid_size):
                cell = self.grid[(gx, gy)]
                
                # Calculate pixel position (center of cell)
                px = gx * cell_pixels + cell_pixels // 2
                py = gy * cell_pixels + cell_pixels // 2
                
                # Skip if this is the agent's current position (draw agent separately)
                if (gx, gy) == tuple(self.grid_pos):
                    continue
                
                # Draw visual elements based on cell type
                if cell['type'] == 'goal' and not cell['visited']:
                    # Ubuntu Mastery Temple - beautiful golden shrine
                    temple_size = 45
                    
                    # Shadow
                    shadow_surf = pygame.Surface((temple_size*2, temple_size*2), pygame.SRCALPHA)
                    pygame.draw.ellipse(shadow_surf, (0, 0, 0, 60), (10, temple_size + 20, temple_size*1.5, 15))
                    canvas.blit(shadow_surf, (px - temple_size, py - temple_size))
                    
                    # Temple base - 3D effect
                    pygame.draw.rect(canvas, (218, 165, 32), (px - temple_size//2, py, temple_size, temple_size//2))
                    pygame.draw.rect(canvas, (255, 215, 0), (px - temple_size//2 + 3, py + 3, temple_size - 6, temple_size//2 - 6))
                    
                    # Temple pillars
                    pillar_color = (255, 235, 150)
                    for pillar_x in [-temple_size//2 + 8, -temple_size//2 + 20, temple_size//2 - 20, temple_size//2 - 8]:
                        pygame.draw.rect(canvas, pillar_color, (px + pillar_x, py - temple_size//3, 4, temple_size//2 + 5))
                    
                    # Temple roof with layers
                    roof_points = [
                        (px - temple_size//2 - 8, py - temple_size//3),
                        (px + temple_size//2 + 8, py - temple_size//3),
                        (px + temple_size//2 - 2, py - temple_size//2),
                        (px, py - temple_size//2 - 8),
                        (px - temple_size//2 + 2, py - temple_size//2)
                    ]
                    pygame.draw.polygon(canvas, (184, 134, 11), roof_points)
                    
                    # Decorative elements on roof
                    roof_edge = [(px - temple_size//2 - 5, py - temple_size//3 - 2), 
                                 (px + temple_size//2 + 5, py - temple_size//3 - 2)]
                    pygame.draw.line(canvas, (255, 223, 0), roof_edge[0], roof_edge[1], 3)
                    
                    # Golden sphere/dome on top
                    pygame.draw.circle(canvas, (255, 215, 0), (px, py - temple_size//2 - 12), 8)
                    pygame.draw.circle(canvas, (255, 235, 150), (px - 2, py - temple_size//2 - 14), 3)
                    
                    # Radiating glow
                    for i in range(4):
                        alpha = 50 - i * 12
                        radius = temple_size//2 + i * 12
                        glow_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                        pygame.draw.circle(glow_surf, (255, 215, 0, alpha), (radius, radius), radius)
                        canvas.blit(glow_surf, (px - radius, py - radius))
                    
                    # Label with nice styling
                    label = font_bold.render('UBUNTU TEMPLE', True, (255, 255, 255))
                    label_shadow = font_bold.render('UBUNTU TEMPLE', True, (0, 0, 0))
                    label_rect = label.get_rect(center=(px, py + temple_size//2 + 12))
                    canvas.blit(label_shadow, (label_rect.x + 1, label_rect.y + 1))
                    canvas.blit(label, label_rect)
                
                elif cell['type'] == 'good' and not cell['visited']:
                    # Good philosophy - SQUARE/ROUNDED SHAPE (clear distinction)
                    visual_map = {
                        'help_needy': ('Help', (40, 180, 80)),
                        'share_knowledge': ('Teach', (70, 140, 220)),
                        'show_compassion': ('Care', (160, 100, 220)),
                        'build_community': ('Unite', (240, 160, 50)),
                        'forgive': ('Forgive', (220, 100, 160))
                    }
                    
                    label_text, main_color = visual_map.get(cell['philosophy'], ('Good', (40, 180, 80)))
                    
                    # Subtle glow
                    pulse = abs(math.sin(self.steps * 0.08)) * 2
                    glow_size = int(26 + pulse)
                    glow_surf = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
                    pygame.draw.rect(glow_surf, (*main_color, 25), (glow_size - 22, glow_size - 22, 44, 44), border_radius=8)
                    canvas.blit(glow_surf, (px - glow_size, py - glow_size))
                    
                    # ROUNDED SQUARE SHAPE - clear visual distinction from bad (triangle)
                    square_size = 20
                    light_color = tuple(min(c + 60, 255) for c in main_color)
                    
                    # Background square
                    pygame.draw.rect(canvas, light_color, 
                                   (px - square_size, py - square_size - 4, square_size*2, square_size*2), 
                                   border_radius=8)
                    # Border
                    pygame.draw.rect(canvas, main_color, 
                                   (px - square_size, py - square_size - 4, square_size*2, square_size*2), 
                                   3, border_radius=8)
                    
                    # Plus symbol inside (positive connotation)
                    plus_size = 8
                    pygame.draw.rect(canvas, (255, 255, 255), 
                                   (px - plus_size//2, py - plus_size - 4, plus_size//2, plus_size*2), 
                                   border_radius=1)
                    pygame.draw.rect(canvas, (255, 255, 255), 
                                   (px - plus_size, py - plus_size//2 - 4, plus_size*2, plus_size//2), 
                                   border_radius=1)
                    
                    # Minimal label
                    label = font_label.render(label_text, True, (255, 255, 255))
                    label_rect = label.get_rect(center=(px, py + 24))
                    pygame.draw.rect(canvas, (0, 0, 0, 120), label_rect.inflate(6, 2), border_radius=2)
                    canvas.blit(label, label_rect)
                
                elif cell['type'] == 'bad' and not cell['visited']:
                    # Bad philosophy - TRIANGLE/DIAMOND SHAPE (clear distinction from good)
                    visual_map = {
                        'selfish_act': ('Greed', (200, 70, 70)),
                        'harm_other': ('Harm', (180, 50, 50)),
                        'hoard_resources': ('Hoard', (160, 60, 60)),
                        'show_greed': ('Selfish', (140, 50, 50)),
                        'violence': ('DANGER', (220, 40, 40))
                    }
                    
                    label_text, color = visual_map.get(cell['philosophy'], ('Bad', (200, 70, 70)))
                    
                    # Warning pulse
                    pulse = abs(math.sin(self.steps * 0.12)) * 3
                    
                    # Warning glow
                    glow_size = int(28 + pulse)
                    glow_surf = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (*color, 30), (glow_size, glow_size), glow_size)
                    canvas.blit(glow_surf, (px - glow_size, py - glow_size))
                    
                    # TRIANGLE/DIAMOND SHAPE - clear visual distinction from good (square)
                    triangle_size = 22
                    
                    # Inverted triangle (warning symbol shape)
                    triangle_points = [
                        (px, py - triangle_size - 4),  # Top
                        (px - triangle_size, py + triangle_size//2 - 4),  # Bottom left
                        (px + triangle_size, py + triangle_size//2 - 4)   # Bottom right
                    ]
                    
                    # Background with darker color
                    dark_color = tuple(max(c - 40, 0) for c in color)
                    pygame.draw.polygon(canvas, dark_color, triangle_points)
                    
                    # Border
                    pygame.draw.polygon(canvas, color, triangle_points, 3)
                    
                    # Exclamation mark inside (warning)
                    pygame.draw.rect(canvas, (255, 255, 255), 
                                   (px - 2, py - 14, 4, 12), border_radius=1)
                    pygame.draw.circle(canvas, (255, 255, 255), (px, py + 4), 2)
                    
                    # Minimal label
                    label = font_label.render(label_text, True, (255, 200, 200))
                    label_rect = label.get_rect(center=(px, py + 24))
                    pygame.draw.rect(canvas, (0, 0, 0, 140), label_rect.inflate(6, 2), border_radius=2)
                    canvas.blit(label, label_rect)
                
                elif cell['type'] == 'empty' and not cell['visited']:
                    # Empty spaces - MINIMALISTIC nature elements (smaller, less detail)
                    nature_type = (gx * 7 + gy * 3) % 5
                    
                    if nature_type == 0:
                        # Simple small tree
                        trunk_color = (101, 67, 33)
                        pygame.draw.rect(canvas, trunk_color, (px - 3, py + 4, 6, 12), border_radius=1)
                        # Simple foliage circle
                        pygame.draw.circle(canvas, (60, 150, 60), (px, py - 2), 10)
                        pygame.draw.circle(canvas, (80, 180, 80), (px, py - 6), 8)
                        
                    elif nature_type == 1:
                        # Simple small bush
                        pygame.draw.ellipse(canvas, (60, 140, 60), (px - 10, py - 2, 20, 14))
                        pygame.draw.circle(canvas, (255, 150, 180), (px, py - 4), 3)
                    
                    elif nature_type == 2:
                        # Minimalistic house
                        house_color = (160, 100, 60)
                        # Simple house body (smaller)
                        pygame.draw.rect(canvas, house_color, (px - 8, py, 16, 12), border_radius=1)
                        # Simple roof triangle
                        roof_points = [(px - 10, py), (px + 10, py), (px, py - 10)]
                        pygame.draw.polygon(canvas, (120, 70, 40), roof_points)
                    
                    elif nature_type == 3:
                        # Simple rock - single ellipse
                        pygame.draw.ellipse(canvas, (120, 120, 120), (px - 8, py + 2, 16, 10))
                        pygame.draw.ellipse(canvas, (140, 140, 140), (px - 6, py + 3, 12, 8))
                    
                    else:
                        # Simple minimalistic flower - just one
                        pygame.draw.line(canvas, (60, 140, 60), (px, py + 4), (px, py - 4), 2)
                        pygame.draw.circle(canvas, (255, 150, 180), (px, py - 4), 4)
                        pygame.draw.circle(canvas, (255, 200, 100), (px, py - 4), 2)
                
                # Draw narrow trail for visited locations
                if cell['visited'] and cell['type'] != 'goal':
                    # Small dot trail in center
                    pygame.draw.circle(canvas, (150, 255, 150, 100), (px, py), 6)
                    pygame.draw.circle(canvas, (100, 200, 100, 60), (px, py), 3)
        
        # Draw particles (philosophy effects) - subtle visuals
        for p in self.particles:
            if p['lifetime'] > 0:
                px = int(p['position'][0] * cell_pixels + cell_pixels//2)
                py = int(p['position'][1] * cell_pixels + cell_pixels//2)
                alpha = int((p['lifetime'] / 30.0) * 180)
                size = int(4 + (1 - p['lifetime'] / 30.0) * 3)
                
                # Draw particle with subtle glow
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p['color'], alpha//3), (size, size), size)
                pygame.draw.circle(s, (*p['color'], alpha), (size, size), size//2)
                canvas.blit(s, (px - size, py - size))
        
        # Draw agent as AI Robot character
        agent_px = self.grid_pos[0] * cell_pixels + cell_pixels // 2
        agent_py = self.grid_pos[1] * cell_pixels + cell_pixels // 2
        
        # Robot appearance - metallic colors based on ubuntu state
        if self.ubuntu_state == 'UBUNTU_MASTER':
            primary_color = (220, 180, 40)  # Gold robot
            secondary_color = (255, 220, 80)
            accent_color = (255, 200, 0)
            eye_color = (0, 255, 200)  # Cyan glow
        elif self.ubuntu_state == 'UBUNTU_LEARNER':
            primary_color = (50, 150, 80)  # Green robot
            secondary_color = (80, 200, 120)
            accent_color = (100, 255, 150)
            eye_color = (100, 255, 255)  # Bright cyan
        elif self.ubuntu_state == 'ANTI_UBUNTU':
            primary_color = (180, 60, 60)  # Red robot
            secondary_color = (220, 80, 80)
            accent_color = (255, 100, 100)
            eye_color = (255, 50, 50)  # Red glow
        else:
            primary_color = (120, 130, 150)  # Gray robot
            secondary_color = (160, 170, 190)
            accent_color = (200, 210, 220)
            eye_color = (100, 200, 255)  # Blue glow
        
        # Robot shadow
        pygame.draw.ellipse(canvas, (0, 0, 0, 40), (agent_px - 17, agent_py + 13, 34, 8))
        
        # Energy field/aura
        glow_radius = 33
        aura_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(aura_surf, (*accent_color, 35), (glow_radius, glow_radius), glow_radius)
        canvas.blit(aura_surf, (agent_px - glow_radius, agent_py - glow_radius))
        
        # Robot head - rectangular with rounded corners
        head_rect = pygame.Rect(agent_px - 11, agent_py - 19, 22, 18)
        pygame.draw.rect(canvas, (80, 80, 90), head_rect, border_radius=4)
        pygame.draw.rect(canvas, primary_color, head_rect.inflate(-3, -3), border_radius=3)
        
        # Antenna on top
        pygame.draw.rect(canvas, secondary_color, (agent_px - 1, agent_py - 23, 2, 5))
        pygame.draw.circle(canvas, accent_color, (agent_px, agent_py - 23), 2)
        # Antenna light - subtle pulse
        pulse = abs(math.sin(self.steps * 0.15)) * 40 + 180
        pygame.draw.circle(canvas, (int(pulse), int(pulse), 220, 150), (agent_px, agent_py - 23), 1)
        
        # Visor/face panel - darker area
        visor_rect = pygame.Rect(agent_px - 8, agent_py - 16, 16, 10)
        pygame.draw.rect(canvas, (20, 25, 35), visor_rect, border_radius=2)
        
        # Glowing AI eyes - subtle animation
        eye_glow = abs(math.sin(self.steps * 0.15)) * 30 + 140
        # Left eye
        pygame.draw.rect(canvas, eye_color, (agent_px - 6, agent_py - 13, 3, 5), border_radius=1)
        eye_surf = pygame.Surface((7, 7), pygame.SRCALPHA)
        pygame.draw.circle(eye_surf, (*eye_color, int(eye_glow * 0.3)), (3, 3), 3)
        canvas.blit(eye_surf, (agent_px - 8, agent_py - 14))
        
        # Right eye
        pygame.draw.rect(canvas, eye_color, (agent_px + 3, agent_py - 13, 3, 5), border_radius=1)
        canvas.blit(eye_surf, (agent_px + 1, agent_py - 14))
        
        # Small display/mouth - digital line
        mouth_points = [(agent_px - 5, agent_py - 7), (agent_px - 2, agent_py - 6), 
                       (agent_px + 2, agent_py - 6), (agent_px + 5, agent_py - 7)]
        pygame.draw.lines(canvas, accent_color, False, mouth_points, 1)
        
        # Robot body - torso with panels
        body_rect = pygame.Rect(agent_px - 12, agent_py - 1, 24, 16)
        pygame.draw.rect(canvas, (80, 80, 90), body_rect, border_radius=2)
        pygame.draw.rect(canvas, primary_color, body_rect.inflate(-2, -2), border_radius=2)
        
        # Chest panel with circuit design
        panel_rect = pygame.Rect(agent_px - 6, agent_py + 1, 12, 10)
        pygame.draw.rect(canvas, secondary_color, panel_rect, border_radius=1)
        # Circuit lines
        pygame.draw.line(canvas, accent_color, (agent_px, agent_py + 2), (agent_px, agent_py + 10), 1)
        pygame.draw.line(canvas, accent_color, (agent_px - 4, agent_py + 6), (agent_px + 4, agent_py + 6), 1)
        # Core light
        pygame.draw.circle(canvas, eye_color, (agent_px, agent_py + 6), 2)
        
        # Shoulder joints
        pygame.draw.circle(canvas, (80, 80, 90), (agent_px - 12, agent_py + 1), 3)
        pygame.draw.circle(canvas, secondary_color, (agent_px - 12, agent_py + 1), 2)
        pygame.draw.circle(canvas, (80, 80, 90), (agent_px + 12, agent_py + 1), 3)
        pygame.draw.circle(canvas, secondary_color, (agent_px + 12, agent_py + 1), 2)
        
        # Robot arms - segmented
        # Left arm
        pygame.draw.rect(canvas, primary_color, (agent_px - 16, agent_py + 1, 5, 10), border_radius=1)
        pygame.draw.line(canvas, secondary_color, (agent_px - 14, agent_py + 4), (agent_px - 14, agent_py + 8), 1)
        # Left hand
        pygame.draw.rect(canvas, secondary_color, (agent_px - 17, agent_py + 10, 7, 4), border_radius=1)
        
        # Right arm
        pygame.draw.rect(canvas, primary_color, (agent_px + 11, agent_py + 1, 5, 10), border_radius=1)
        pygame.draw.line(canvas, secondary_color, (agent_px + 13, agent_py + 4), (agent_px + 13, agent_py + 8), 1)
        # Right hand
        pygame.draw.rect(canvas, secondary_color, (agent_px + 10, agent_py + 10, 7, 4), border_radius=1)
        
        # Master state - crown/halo
        if self.ubuntu_state == 'UBUNTU_MASTER':
            crown_y = agent_py - 26
            # Holographic crown
            for i in range(5):
                x_offset = (i - 2) * 4
                height = 5 if i % 2 == 0 else 7
                crown_rect = pygame.Rect(agent_px + x_offset - 1, crown_y - height, 2, height)
                crown_surf = pygame.Surface((2, height), pygame.SRCALPHA)
                pygame.draw.rect(crown_surf, (*accent_color, 180), (0, 0, 2, height))
                canvas.blit(crown_surf, crown_rect)
        
        # Clean "YOU" indicator with AI theme
        font_indicator = pygame.font.Font(None, 16)
        you_text = font_indicator.render('🤖 AI', True, (255, 255, 255))
        you_rect = you_text.get_rect(center=(agent_px, agent_py - 37))
        
        # Holographic background
        pygame.draw.rect(canvas, (0, 0, 0, 180), you_rect.inflate(8, 4), border_radius=2)
        pygame.draw.rect(canvas, primary_color, you_rect.inflate(8, 4), 1, border_radius=2)
        canvas.blit(you_text, you_rect)
        
        # Animated arrow pointing down
        arrow_bounce = abs(math.sin(self.steps * 0.15)) * 2
        arrow_y = agent_py - 31 + arrow_bounce
        pygame.draw.polygon(canvas, accent_color, [
            (agent_px, arrow_y),
            (agent_px - 3, arrow_y - 3),
            (agent_px + 3, arrow_y - 3)
        ])
        
        # Draw minimal info panel at TOP-RIGHT (moved from blocking view)
        panel_width = 220
        panel_height = 85
        panel_x = self.window_size - panel_width - 10
        panel_y = 10  # TOP instead of bottom
        
        # Panel background with high transparency
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, (15, 20, 30, 160), (0, 0, panel_width, panel_height), border_radius=8)
        
        # Thin border
        pygame.draw.rect(panel_surface, (255, 255, 255, 100), (0, 0, panel_width, panel_height), 2, border_radius=8)
        canvas.blit(panel_surface, (panel_x, panel_y))
        
        # Info text - COMPACT
        font_info = pygame.font.Font(None, 20)
        
        # Get display color based on state
        if self.ubuntu_state == 'UBUNTU_MASTER':
            display_color = (220, 180, 40)
            state_short = '👑 MASTER'
        elif self.ubuntu_state == 'UBUNTU_LEARNER':
            display_color = (50, 150, 80)
            state_short = '📈 LEARNER'
        elif self.ubuntu_state == 'ANTI_UBUNTU':
            display_color = (180, 60, 60)
            state_short = '⚠️ ANTI'
        else:
            display_color = (120, 130, 150)
            state_short = '🎓 LEARNING'
        
        # Compact layout - everything in 3 lines
        y_offset = panel_y + 12
        
        # Line 1: State
        state_text = font_info.render(state_short, True, display_color)
        canvas.blit(state_text, (panel_x + 12, y_offset))
        
        # Line 2: Score with mini progress bar
        y_offset += 24
        score_text = font_info.render(f'Score: {int(self.ubuntu_score)}/{self.ubuntu_threshold}', True, (255, 255, 255))
        canvas.blit(score_text, (panel_x + 12, y_offset))
        
        # Mini progress bar next to score
        bar_width = 80
        bar_height = 8
        bar_x = panel_x + 140
        bar_y = y_offset + 6
        pygame.draw.rect(canvas, (40, 40, 50), (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        progress = min(self.ubuntu_score / self.ubuntu_threshold, 1.0)
        if progress > 0:
            progress_width = int(bar_width * progress)
            pygame.draw.rect(canvas, display_color, (bar_x, bar_y, progress_width, bar_height), border_radius=4)
        
        # Line 3: Actions and Steps
        y_offset += 24
        actions_text = font_info.render(f'💚{self.positive_actions} 💔{self.negative_actions} • {self.steps} steps', True, (200, 200, 200))
        canvas.blit(actions_text, (panel_x + 12, y_offset))
        
        if self.render_mode == "human":
            # Copy to display window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
