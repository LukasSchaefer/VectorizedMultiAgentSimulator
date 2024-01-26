from time import perf_counter
from typing import Dict, List, Optional, Union

import torch


class EpisodeStatisticsRecorder:
    """Record statistics for each episode."""

    def __init__(self, n_agents:int, n_envs: int, device: torch.device):
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.device = device
        self.t0 = [perf_counter() for _ in range(n_envs)]
        self.episode_reward = torch.zeros((n_agents, n_envs), dtype=torch.float32, device=device)
        self.episode_length = torch.zeros(n_envs, dtype=torch.float32, device=device)

    def new_episode(self, index: Optional[int] = None):
        if index is None:
            self.episode_reward.zero_()
            self.episode_length.zero_()
            self.t0 = [perf_counter() for _ in range(self.n_envs)]
        else:
            self.episode_reward[:, index].zero_()
            self.episode_length[index].zero_()
            self.t0[index] = perf_counter()

    def process_step(
            self, reward: Union[List[torch.tensor], Dict[str, torch.tensor]],
            done: torch.tensor,
            info: List[Dict[str, torch.tensor]],
        ):
        """
        Process a step of the environment and add entries to info dict.
        :param reward: Rewards as list of len n_agents of tensors of shape (n_envs) or dict mapping agent names
            to tensors of shape (n_envs)
        :param done: Done as tensor of shape (n_envs)
        :param info: Info as list of len n_envs containing dicts mapping metric names to tensors
        """
        if isinstance(reward, dict):
            reward = list(reward.values())
        # accumulate rewards and lengths for each env
        self.episode_reward += torch.stack(reward)
        self.episode_length += 1
        if any(done):
            # if any episode terminated --> add respective values to info and reset episode statistics
            for env_index, (env_done, env_info) in enumerate(zip(done, info)):
                if env_done:
                    env_info["episode_reward"] = self.episode_reward[:, env_index].clone()
                    env_info["episode_length"] = self.episode_length[env_index].clone()
                    env_info["episode_time"] = perf_counter() - self.t0[env_index]
                    for i in range(self.n_agents):
                        env_info[f"agent{i}/episode_reward"] = self.episode_reward[i, env_index].clone()