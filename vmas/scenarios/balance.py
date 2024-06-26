#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Line, Box
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, Y


IMMUTABLES = ["n_agents", "agent_radius", "line_length"]

class Scenario(BaseScenario):
    def init_params(self, **kwargs):
        self.n_agents = kwargs.get("n_agents", 3)
        assert self.n_agents > 1
        self.observe_other_agents = kwargs.get("observe_other_agents", False)

        self.package_mass = kwargs.get("package_mass", 5)
        self.random_package_pos_on_line = kwargs.get("random_package_pos_on_line", True)

        self.line_length = kwargs.get("line_length", 0.8)
        self.line_mass = kwargs.get("line_mass", 5)

        self.agent_radius = kwargs.get("agent_radius", 0.03)

        self.world_gravity = kwargs.get("world_gravity", (0.0, -0.05))

        self.shaping_factor = kwargs.get("shaping_factor", 100)
        self.fall_reward = kwargs.get("fall_reward", -10)
        self.rew_on_goal = kwargs.get("rew_on_goal", 0)
        self.terminate_on_goal = kwargs.get("terminate_on_goal", True)

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.init_params(**kwargs)

        # Make world
        world = World(batch_dim, device, gravity=self.world_gravity, y_semidim=1)
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}", shape=Sphere(self.agent_radius), u_multiplier=0.7
            )
            world.add_agent(agent)

        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        self.package = Landmark(
            name="package",
            collide=True,
            movable=True,
            shape=Sphere(),
            mass=self.package_mass,
            color=Color.RED,
        )
        self.package.goal = goal
        world.add_landmark(self.package)
        # Add landmarks

        self.line = Landmark(
            name="line",
            shape=Line(length=self.line_length),
            collide=True,
            movable=True,
            rotatable=True,
            mass=self.line_mass,
            color=Color.BLACK,
        )
        world.add_landmark(self.line)

        self.floor = Landmark(
            name="floor",
            collide=True,
            shape=Box(length=10, width=1),
            color=Color.WHITE,
        )
        world.add_landmark(self.floor)

        return world
    
    def update_arguments(self, **kwargs):
        super().update_arguments(IMMUTABLES, **kwargs)
        
        # arguments that require changes to the world
        if "world_gravity" in kwargs:
            self.world._gravity = self.world_gravity
        
        # arguments that require changes the package
        if "package_mass" in kwargs:
            self.package.mass = self.package_mass
        
        # arguments that require changes of the line
        if "line_mass" in kwargs:
            self.line.mass = self.line_mass
    
    def get_mutable_arguments(self):
        return [
            "world_gravity",
            "package_mass",
            "line_mass",
            # "random_package_pos_on_line",
            # "shaping_factor",
            # "fall_reward",
        ]
        
    def reset_world_at(self, env_index: int = None):
        goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    0.0,
                    self.world.y_semidim,
                ),
            ],
            dim=1,
        )
        line_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0 + self.line_length / 2,
                    1.0 - self.line_length / 2,
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    -self.world.y_semidim + self.agent_radius * 2,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )
        package_rel_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.line_length / 2 + self.package.shape.radius
                    if self.random_package_pos_on_line
                    else 0.0,
                    self.line_length / 2 - self.package.shape.radius
                    if self.random_package_pos_on_line
                    else 0.0,
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    self.package.shape.radius,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )

        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                line_pos
                + torch.tensor(
                    [
                        -(self.line_length - agent.shape.radius) / 2
                        + i
                        * (self.line_length - agent.shape.radius)
                        / (self.n_agents - 1),
                        -self.agent_radius * 2,
                    ],
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )

        self.line.set_pos(
            line_pos,
            batch_index=env_index,
        )
        self.package.goal.set_pos(
            goal_pos,
            batch_index=env_index,
        )
        self.line.set_rot(
            torch.zeros(1, device=self.world.device, dtype=torch.float32),
            batch_index=env_index,
        )
        self.package.set_pos(
            line_pos + package_rel_pos,
            batch_index=env_index,
        )

        self.floor.set_pos(
            torch.tensor(
                [
                    0,
                    -self.world.y_semidim
                    - self.floor.shape.width / 2
                    - self.agent_radius,
                ],
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        self.compute_on_the_ground()
        if env_index is None:
            self.global_shaping = (
                torch.linalg.vector_norm(
                    self.package.state.pos - self.package.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )
        else:
            self.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.package.state.pos[env_index]
                    - self.package.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )

    def compute_on_the_ground(self):
        self.on_the_ground = self.world.is_overlapping(
            self.line, self.floor
        ) + self.world.is_overlapping(self.package, self.floor)

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )

            # on goal reward
            self.package.on_goal = self.world.is_overlapping(self.package, self.package.goal)
            self.package.color = torch.tensor(
                Color.RED.value, device=self.world.device, dtype=torch.float32
            ).repeat(self.world.batch_dim, 1)
            self.package.color[self.package.on_goal] = torch.tensor(
                Color.GREEN.value, device=self.world.device, dtype=torch.float32
            )
            if torch.is_tensor(self.rew_on_goal):
                assert self.rew_on_goal.shape[0] == self.world.batch_dim
                self.rew[self.package.on_goal] += self.rew_on_goal[self.package.on_goal]
            else:
                self.rew[self.package.on_goal] += self.rew_on_goal

            # fall / touch ground reward
            self.compute_on_the_ground()
            self.rew[self.on_the_ground] += self.fall_reward

            # distance reward
            self.package_dist = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )
            global_shaping = self.package_dist * self.shaping_factor
            self.rew[~self.package.on_goal] += (self.global_shaping[~self.package.on_goal] - global_shaping[~self.package.on_goal])
            self.global_shaping = global_shaping

        return self.rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        other_agent_obs = []
        if self.observe_other_agents:
            # observe other agent pos and vel
            for other_agent in [other_agent for other_agent in self.world.agents if other_agent != agent]:
                other_agent_obs.extend([other_agent.state.pos - agent.state.pos, other_agent.state.vel])

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - self.package.state.pos,
                agent.state.pos - self.line.state.pos,
                self.package.state.pos - self.package.goal.state.pos,
                self.package.state.vel,
                self.line.state.vel,
                self.line.state.ang_vel,
                self.line.state.rot % torch.pi,
                *other_agent_obs
            ],
            dim=-1,
        )

    def done(self):
        if torch.is_tensor(self.terminate_on_goal):
            assert all(self.terminate_on_goal) or not any(self.terminate_on_goal), "terminate_on_goal must be either True or False for all environments"

        if (not torch.is_tensor(self.terminate_on_goal) and self.terminate_on_goal) or (
            torch.is_tensor(self.terminate_on_goal) and all(self.terminate_on_goal)):
            return torch.logical_or(self.package.on_goal, self.on_the_ground)
        else:
            return self.on_the_ground


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        batch_dim = observation.shape[0]

        index_package_goal_pos = 8
        dist_package_goal = observation[
            :, index_package_goal_pos : index_package_goal_pos + 2
        ]
        y_distance_ge_0 = dist_package_goal[:, Y] >= 0

        if self.continuous_actions:
            action_agent = torch.clamp(
                torch.stack(
                    [
                        torch.zeros(batch_dim, device=observation.device),
                        -dist_package_goal[:, Y],
                    ],
                    dim=1,
                ),
                min=-u_range,
                max=u_range,
            )
            action_agent[:, Y][y_distance_ge_0] = 0
        else:
            action_agent = torch.full((batch_dim,), 4, device=observation.device)
            action_agent[y_distance_ge_0] = 0
        return action_agent


if __name__ == "__main__":
    render_interactively(
        __file__,
        n_agents=2,
        package_mass=0.5,
        line_mass=1.0,
        world_gravity=(0.0, -0.01),
        random_package_pos_on_line=True,
        control_two_agents=True,
        terminate_on_goal=False,
        rew_on_goal=2,
    )
