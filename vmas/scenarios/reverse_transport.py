#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, DRAG


IMMUTABLES = [
    "n_agents",
    "observe_other_agents",
    "agent_radius",
    "goal_radius",
    "package_length",
    "package_width",
    "collision_force",
]

class Scenario(BaseScenario):
    def init_params(self, **kwargs):
        self.n_agents = kwargs.get("n_agents", 4)
        self.package_width = kwargs.get("package_width", 0.6)
        self.package_length = kwargs.get("package_length", 0.6)
        self.package_mass = kwargs.get("package_mass", 50)
        self.observe_other_agents = kwargs.get("observe_other_agents", False)

        self.agent_radius = kwargs.get("agent_radius", 0.03)
        self.goal_radius = kwargs.get("goal_radius", 0.09)

        self.collision_force = kwargs.get("collision_force", 500)
        self.world_drag = kwargs.get("world_drag", DRAG)

        self.rew_dist_shaping_factor = kwargs.get("rew_dist_shaping_factor", 100)
        self.rew_on_goal = kwargs.get("rew_on_goal", 0)
        self.terminate_on_goal = kwargs.get("terminate_on_goal", True)

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.init_params(**kwargs)

        for rew_factor in ["rew_dist_shaping_factor", "rew_on_goal"]:
            if rew_factor in kwargs:
                value = getattr(self, rew_factor)
                if torch.is_tensor(value):
                    assert value.shape[0] == batch_dim
                    if value.dim() > 1:
                        setattr(self, rew_factor, value.reshape(batch_dim))

        # Make world
        world = World(
            batch_dim, device, contact_margin=6e-3, substeps=5, collision_force=self.collision_force, drag=self.world_drag,
        )
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(name=f"agent_{i}", shape=Sphere(self.agent_radius), u_multiplier=0.5)
            world.add_agent(agent)
        # Add landmarks
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=self.goal_radius),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)

        self.package = Landmark(
            name=f"package {i}",
            collide=True,
            movable=True,
            mass=self.package_mass,
            shape=Box(
                length=self.package_length, width=self.package_width, hollow=True
            ),
            color=Color.RED,
        )
        self.package.goal = goal
        world.add_landmark(self.package)

        return world
    
    def update_arguments(self, **kwargs):
        super().update_arguments(**kwargs)

        if any(k in kwargs for k in IMMUTABLES):
            raise ValueError(f"Cannot change {IMMUTABLES} after initialization")
        
        if "package_mass" in kwargs:
            self.package.mass = self.package_mass
        
        if "world_drag" in kwargs:
            self.world.drag = self.world_drag
        
        for rew_factor in ["rew_dist_shaping_factor", "rew_on_goal"]:
            if rew_factor in kwargs:
                value = getattr(self, rew_factor)
                if torch.is_tensor(value):
                    assert value.shape[0] == self.world.batch_dim
                    if value.dim() > 1:
                        setattr(self, rew_factor, value.reshape(self.world.batch_dim))
        
    def get_mutable_arguments(self):
        return ["package_mass", "world_drag"]

    def reset_world_at(self, env_index: int = None):
        package_pos = torch.zeros(
            (1, self.world.dim_p)
            if env_index is not None
            else (self.world.batch_dim, self.world.dim_p),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(
            -1.0,
            1.0,
        )

        self.package.set_pos(
            package_pos,
            batch_index=env_index,
        )
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                torch.cat(
                    [
                        torch.zeros(
                            (1, 1)
                            if env_index is not None
                            else (self.world.batch_dim, 1),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            -self.package_length / 2 + agent.shape.radius,
                            self.package_length / 2 - agent.shape.radius,
                        ),
                        torch.zeros(
                            (1, 1)
                            if env_index is not None
                            else (self.world.batch_dim, 1),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            -self.package_width / 2 + agent.shape.radius,
                            self.package_width / 2 - agent.shape.radius,
                        ),
                    ],
                    dim=1,
                )
                + package_pos,
                batch_index=env_index,
            )

        self.package.goal.set_pos(
            torch.zeros(
                (1, self.world.dim_p)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(
                -1.0,
                1.0,
            ),
            batch_index=env_index,
        )

        if env_index is None:
            self.package.global_shaping = (
                torch.linalg.vector_norm(
                    self.package.state.pos - self.package.goal.state.pos, dim=1
                )
                * self.rew_dist_shaping_factor
            )
            self.package.on_goal = torch.zeros(
                self.world.batch_dim, dtype=torch.bool, device=self.world.device
            )
        else:
            self.package.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.package.state.pos[env_index]
                    - self.package.goal.state.pos[env_index]
                )
                * self.rew_dist_shaping_factor
            )
            self.package.on_goal[env_index] = False

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )

            self.package.dist_to_goal = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )
            self.package.on_goal = self.world.is_overlapping(
                self.package, self.package.goal
            )
            self.package.color = torch.tensor(
                Color.RED.value, device=self.world.device, dtype=torch.float32
            ).repeat(self.world.batch_dim, 1)
            self.package.color[self.package.on_goal] = torch.tensor(
                Color.GREEN.value, device=self.world.device, dtype=torch.float32
            )

            package_shaping = self.package.dist_to_goal * self.rew_dist_shaping_factor
            self.rew[~self.package.on_goal] += (
                self.package.global_shaping[~self.package.on_goal]
                - package_shaping[~self.package.on_goal]
            )
            self.package.global_shaping = package_shaping

            if torch.is_tensor(self.rew_on_goal):
                assert self.rew_on_goal.shape[0] == self.world.batch_dim
                self.rew[self.package.on_goal] += self.rew_on_goal[self.package.on_goal]
            else:
                self.rew[self.package.on_goal] += self.rew_on_goal

        return self.rew

    def observation(self, agent: Agent):
        if self.observe_other_agents:
            # observe own pos and vel
            observe_data = [agent.state.pos, agent.state.vel]
            # observe other agent pos and vel
            for other_agent in [other_agent for other_agent in self.world.agents if other_agent != agent]:
                observe_data.extend([other_agent.state.pos, other_agent.state.vel])
            # observe package vel and relative pos
            observe_data.extend([self.package.state.vel, self.package.state.pos - agent.state.pos])
            # observe package goal relative pos
            observe_data.append(self.package.state.pos - self.package.goal.state.pos)
            return torch.cat(observe_data, dim=-1)
        else:
            # observe own pos and vel, package vel and relative pos, package goal relative pos
            # (don't observe other agent pos and vel)
            return torch.cat(
                [
                    agent.state.pos,
                    agent.state.vel,
                    self.package.state.vel,
                    self.package.state.pos - agent.state.pos,
                    self.package.state.pos - self.package.goal.state.pos,
                ],
                dim=-1,
            )

    def done(self):
        if torch.is_tensor(self.terminate_on_goal):
            assert all(self.terminate_on_goal) or not any(self.terminate_on_goal), "terminate_on_goal must be either True or False for all environments"

        if (not torch.is_tensor(self.terminate_on_goal) and self.terminate_on_goal) or (
            torch.is_tensor(self.terminate_on_goal) and all(self.terminate_on_goal)):
            return self.package.on_goal
        else:
            return torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)

if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_agents=4,
        package_width=0.6,
        package_length=0.6,
        # terminate_on_goal=False,
        # package_mass=0.5,
        # rew_on_goal=5,
        # world_drag=DRAG / 2,
    )
