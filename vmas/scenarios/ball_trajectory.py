#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.joints import Joint
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import X, Color, JOINT_FORCE


IMMUTABLES = ["n_agents", "joints", "desired_radius", "agent_radius", "ball_radius"]

class Scenario(BaseScenario):

    def init_params(self, **kwargs):
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 0)
        self.speed_shaping_factor = kwargs.get("speed_shaping_factor", 1)
        self.dist_shaping_factor = kwargs.get("dist_shaping_factor", 0)
        self.joints = kwargs.get("joints", True)

        self.world_drag = kwargs.get("world_drag", 0.0)

        self.n_agents = 2

        self.desired_speed = kwargs.get("desired_speed", 1)
        self.desired_radius = kwargs.get("desired_radius", 0.5)

        self.agent_spacing = kwargs.get("agent_spacing", 0.4)
        self.agent_radius = kwargs.get("agent_radius", 0.03)
        self.agent_drag = kwargs.get("agent_drag", 0.25)

        self.ball_radius = kwargs.get("ball_radius", 2 * self.agent_radius)
        self.ball_friction = kwargs.get("ball_friction", 0.04)

        self.joint_mass = kwargs.get("joint_mass", 1)

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.init_params(**kwargs)

        # Make world
        world = World(
            batch_dim,
            device,
            substeps=15 if self.joints else 5,
            joint_force=900 if self.joints else JOINT_FORCE,
            collision_force=1500 if self.joints else 400,
            drag=self.world_drag,
        )
        # Add agents
        agent = Agent(name="agent_0", shape=Sphere(self.agent_radius), drag=self.agent_drag)
        world.add_agent(agent)
        agent = Agent(name="agent_1", shape=Sphere(self.agent_radius), drag=self.agent_drag)
        world.add_agent(agent)

        self.ball = Landmark(
            name="ball",
            shape=Sphere(radius=self.ball_radius),
            collide=True,
            movable=True,
            linear_friction=self.ball_friction,
        )
        world.add_landmark(self.ball)

        if self.joints:
            self.joints = []
            for i in range(self.n_agents):
                self.joints.append(
                    Joint(
                        world.agents[i],
                        self.ball,
                        anchor_a=(0, 0),
                        anchor_b=(0, 0),
                        dist=self.agent_spacing / 2,
                        rotate_a=True,
                        rotate_b=True,
                        collidable=False,
                        width=0,
                        mass=self.joint_mass,
                    )
                )
                world.add_joint(self.joints[i])

        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.speed_rew = self.pos_rew.clone()
        self.dist_rew = self.pos_rew.clone()

        return world
    
    def update_arguments(self, **kwargs):
        super().update_arguments(IMMUTABLES, **kwargs)

        # arguments that require changes of agents
        if "agent_drag" in kwargs:
            for agent in self.world.agents:
                agent.drag = self.agent_drag

        # arguments that require changes the ball
        if "ball_friction" in kwargs:
            self.ball.linear_friction = self.ball_friction
        
        # arguments that require changes of joints
        if self.joints and "joint_mass" in kwargs:
            for joint in self.joints:
                if joint.landmark is not None:
                    joint.landmark.mass = self.joint_mass
        
        # arguments that require changes of the world
        if "world_drag" in kwargs:
            self.world._drag = self.world_drag
    
    def get_mutable_arguments(self):
        return [
            "pos_shaping_factor",
            "speed_shaping_factor",
            "dist_shaping_factor",
            "world_drag",
            "desired_speed",
            "agent_spacing",
            "agent_drag",
            "ball_friction",
            "joint_mass",
        ]

    def reset_world_at(self, env_index: int = None):
        ball_pos = torch.zeros(
            (1, self.world.dim_p)
            if env_index is not None
            else (self.world.batch_dim, self.world.dim_p),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(
            -self.desired_radius,
            self.desired_radius,
        )

        self.ball.set_pos(
            ball_pos,
            batch_index=env_index,
        )

        order = torch.randperm(self.n_agents).tolist()
        agents = [self.world.agents[i] for i in order]
        for i, agent in enumerate(agents):
            agent_pos = ball_pos.clone()
            agent_pos[:, X] += (self.agent_spacing / 2) * (-1 if i == 0 else 1)
            agent.set_pos(
                agent_pos,
                batch_index=env_index,
            )

        if env_index is None:
            self.pos_shaping = (
                torch.linalg.vector_norm(
                    self.ball.state.pos
                    - self.get_closest_point_circle(self.ball.state.pos),
                    dim=1,
                )
                ** 0.5
                * self.pos_shaping_factor
            )

            self.speed_shaping = (
                self.desired_speed
                - torch.linalg.vector_norm(self.ball.state.vel, dim=1)
            ).abs() * self.speed_shaping_factor

            self.dist_shaping = (
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            a.state.pos - self.ball.state.pos, dim=1
                        )
                        for a in self.world.agents
                    ],
                    dim=1,
                ).sum(dim=1)
                * self.dist_shaping_factor
            )
        else:
            self.pos_shaping = (
                torch.linalg.vector_norm(
                    self.ball.state.pos[env_index]
                    - self.get_closest_point_circle(self.ball.state.pos)[env_index]
                )
                ** 0.5
                * self.pos_shaping_factor
            )

            self.speed_shaping[env_index] = (
                self.desired_speed
                - torch.linalg.vector_norm(self.ball.state.vel[env_index])
            ).abs() * self.speed_shaping_factor

            self.dist_shaping[env_index] = (
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            a.state.pos[env_index] - self.ball.state.pos[env_index],
                        ).unsqueeze(-1)
                        for a in self.world.agents
                    ],
                    dim=1,
                ).sum(dim=1)
                * self.dist_shaping_factor
            )

    def reward(self, agent: Agent):
        pos_shaping = (
            torch.linalg.vector_norm(
                self.ball.state.pos
                - self.get_closest_point_circle(self.ball.state.pos),
                dim=1,
            )
            ** 0.5
            * self.pos_shaping_factor
        )
        self.pos_rew = self.pos_shaping - pos_shaping
        self.pos_shaping = pos_shaping

        speed = torch.linalg.vector_norm(self.ball.state.vel, dim=1)
        speed_shaping = (self.desired_speed - speed).abs() * self.speed_shaping_factor
        self.speed_rew = self.speed_shaping - speed_shaping
        self.speed_shaping = speed_shaping

        dist_shaping = (
            torch.stack(
                [
                    torch.linalg.vector_norm(a.state.pos - self.ball.state.pos, dim=1)
                    for a in self.world.agents
                ],
                dim=1,
            ).sum(dim=1)
            * self.dist_shaping_factor
        )
        self.dist_rew = self.dist_shaping - dist_shaping
        self.dist_shaping = dist_shaping

        return self.pos_rew + self.speed_rew + self.dist_rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - self.ball.state.pos,
                agent.state.pos,
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew,
            "speed_rew": self.speed_rew,
            "dist_rew": self.dist_rew,
        }

    def get_closest_point_circle(self, pos: Tensor):
        pos_norm = torch.linalg.vector_norm(pos, dim=1)
        agent_pos_normalized = pos / pos_norm.unsqueeze(-1)

        agent_pos_normalized *= self.desired_radius

        return torch.nan_to_num(agent_pos_normalized)

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []

        # Trajectory goal circle
        color = Color.BLACK.value
        circle = rendering.make_circle(self.desired_radius, filled=False)
        xform = rendering.Transform()
        circle.add_attr(xform)
        xform.set_translation(0, 0)
        circle.set_color(*color)
        geoms.append(circle)

        return geoms


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
