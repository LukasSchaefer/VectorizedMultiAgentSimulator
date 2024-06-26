#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import warnings

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


IMMUTABLES = [
    "n_agents",
    "n_packages",
    "randomise_n_packages",
    "goal_radius",
    "agent_radius",
    "package_length",
    "package_width",
    "observe_other_agents",
    "partial_observability",
    "observation_range",
    "observe_n_packages",
]


class Scenario(BaseScenario):
    def init_params(self, **kwargs):
        self.n_agents = kwargs.get("n_agents", 4)
        self.n_packages = kwargs.get("n_packages", 1)
        self.randomise_n_packages = kwargs.get("randomise_n_packages", False)
        if self.n_packages == 1 and self.randomise_n_packages:
            # If only one package, randomisation has no effect
            warnings.warn(
                "randomise_n_packages has no effect when n_packages=1. Set n_packages > 1 to use randomise_n_packages."
            )
            self.randomise_n_packages = False
        self.package_width = kwargs.get("package_width", 0.15)
        self.package_length = kwargs.get("package_length", 0.15)
        self.package_mass = kwargs.get("package_mass", 50)

        self.observe_other_agents = kwargs.get("observe_other_agents", False)
        self.partial_observability = kwargs.get("partial_observability", False)
        self.observation_range = kwargs.get("observation_range", 0.75)
        # Can set to number > n_packages to have "empty" packages in observation
        # for training on different number of packages
        self.observe_n_packages = kwargs.get("observe_n_packages", self.n_packages)

        self.goal_radius = kwargs.get("goal_radius", 0.15)

        self.agent_radius = kwargs.get("agent_radius", 0.03)

        self.rew_package_on_goal = kwargs.get("rew_package_on_goal", 0)
        self.rew_all_packages_on_goal = kwargs.get("rew_all_packages_on_goal", 0)
        self.terminate_on_goal = kwargs.get("terminate_on_goal", True)

    def __get_active_packages(self, world, env_index=None):
        if env_index is None:
            active_packages = torch.randint(
                0,
                2,
                (world.batch_dim, self.n_packages),
                device=world.device,
                dtype=torch.bool,
            )
            # always have at least one package
            random_indices = torch.randint(
                0, self.n_packages, (world.batch_dim,), device=world.device
            )
            active_packages[torch.arange(world.batch_dim), random_indices] = 1
        else:
            active_packages = torch.randint(
                0, 2, (self.n_packages,), device=world.device, dtype=torch.bool
            )
            # always have at least one package
            random_index = torch.randint(0, self.n_packages, (1,), device=world.device)
            active_packages[random_index] = 1
        return active_packages

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.init_params(**kwargs)
        self.shaping_factor = 100
        self.world_semidim = 1

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.world_semidim
            + 2 * self.agent_radius
            + max(self.package_length, self.package_width),
            y_semidim=self.world_semidim
            + 2 * self.agent_radius
            + max(self.package_length, self.package_width),
        )
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}", shape=Sphere(self.agent_radius), u_multiplier=0.6
            )
            world.add_agent(agent)
        # Add landmarks
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=self.goal_radius),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        self.packages = []

        for i in range(self.n_packages):
            if (
                self.n_packages > 1
                and torch.is_tensor(self.package_mass)
                and self.package_mass.size(-1) == self.n_packages
            ):
                # individual mass for each package
                package_mass = self.package_mass[..., i].unsqueeze(-1)
            else:
                # same mass for all packages
                package_mass = self.package_mass
            package = Landmark(
                name=f"package {i}",
                collide=True,
                movable=True,
                mass=package_mass,
                shape=Box(length=self.package_length, width=self.package_width),
                color=Color.RED,
            )
            package.goal = goal
            package.color = torch.tensor(
                package.color, device=world.device, dtype=torch.float32
            ).repeat(batch_dim, 1)
            self.packages.append(package)
            world.add_landmark(package)

        if self.randomise_n_packages:
            self.active_packages = self.__get_active_packages(world)
        else:
            self.active_packages = torch.ones(
                (world.batch_dim, self.n_packages),
                device=world.device,
                dtype=torch.bool,
            )

        return world

    def update_arguments(self, **kwargs):
        super().update_arguments(IMMUTABLES, **kwargs)

        if "package_mass" in kwargs:
            if (
                self.n_packages > 1
                and torch.is_tensor(self.package_mass)
                and self.package_mass.size(-1) == self.n_packages
            ):
                # individiual mass for each package
                for i, package in enumerate(self.packages):
                    package.mass = self.package_mass[..., i].unsqueeze(-1)
            else:
                # same mass for all packages
                for package in self.packages:
                    package.mass = self.package_mass

    def get_mutable_arguments(self):
        return ["package_mass"]

    def reset_world_at(self, env_index: int = None):
        # Random pos between -1 and 1
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=self.agent_radius * 2,
            x_bounds=(
                -self.world_semidim,
                self.world_semidim,
            ),
            y_bounds=(
                -self.world_semidim,
                self.world_semidim,
            ),
        )
        agent_occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            agent_occupied_positions = agent_occupied_positions[env_index].unsqueeze(0)

        goal = self.world.landmarks[0]
        ScenarioUtils.spawn_entities_randomly(
            [goal] + self.packages,
            self.world,
            env_index,
            min_dist_between_entities=max(
                package.shape.circumscribed_radius() + goal.shape.radius + 0.01
                for package in self.packages
            ),
            x_bounds=(
                -self.world_semidim,
                self.world_semidim,
            ),
            y_bounds=(
                -self.world_semidim,
                self.world_semidim,
            ),
            occupied_positions=agent_occupied_positions,
        )

        if self.randomise_n_packages:
            if env_index is None:
                self.active_packages = self.__get_active_packages(self.world)
            else:
                self.active_packages[env_index] = self.__get_active_packages(
                    self.world, env_index
                )

            # move packages that are not active out of the world
            to_reset_env_indices = (
                [env_index] if env_index is not None else range(self.world.batch_dim)
            )
            for i, package in enumerate(self.packages):
                for reset_env_index in to_reset_env_indices:
                    if self.active_packages[reset_env_index, i] == 0:
                        package.set_pos(
                            torch.tensor([1e6, 1e6]), batch_index=reset_env_index
                        )
                        package._collide[reset_env_index] = False
                        package.color[reset_env_index] = torch.tensor(Color.GRAY.value)
                    else:
                        package._collide[reset_env_index] = True
                        package.color[reset_env_index] = torch.tensor(Color.RED.value)

            assert torch.all(
                self.active_packages.sum(dim=-1) >= 1
            ), "At least one package must be active for each parallel env"

        for i, package in enumerate(self.packages):
            package.on_goal = self.world.is_overlapping(package, package.goal)

            if env_index is None:
                package.global_shaping = (
                    torch.linalg.vector_norm(
                        package.state.pos - package.goal.state.pos, dim=1
                    )
                    * self.shaping_factor
                )
            else:
                package.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        package.state.pos[env_index] - package.goal.state.pos[env_index]
                    )
                    * self.shaping_factor
                )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )

            for i, package in enumerate(self.packages):
                active_package_mask = self.active_packages[:, i] == 1
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                package.on_goal = self.world.is_overlapping(package, package.goal)
                package.color[package.on_goal & active_package_mask] = torch.tensor(
                    Color.GREEN.value, device=self.world.device, dtype=torch.float32
                )
                package.color[~package.on_goal & active_package_mask] = torch.tensor(
                    Color.RED.value, device=self.world.device, dtype=torch.float32
                )

                package_shaping = package.dist_to_goal * self.shaping_factor
                self.rew[~package.on_goal & active_package_mask] += (
                    package.global_shaping[~package.on_goal & active_package_mask]
                    - package_shaping[~package.on_goal & active_package_mask]
                )
                package.global_shaping = package_shaping

                if torch.is_tensor(self.rew_package_on_goal):
                    assert self.rew_package_on_goal.shape[0] == self.world.batch_dim
                    self.rew[package.on_goal & active_package_mask] += (
                        self.rew_package_on_goal[package.on_goal & active_package_mask]
                    )
                else:
                    self.rew[package.on_goal & active_package_mask] += (
                        self.rew_package_on_goal
                    )

            # Check for each batch size if all packages are on goal
            all_packages_on_goal = torch.stack(
                [
                    package.on_goal | (self.active_packages[:, i].logical_not())
                    for i, package in enumerate(self.packages)
                ],
                dim=-1,
            ).all(dim=-1)
            if any(all_packages_on_goal):
                if torch.is_tensor(self.rew_all_packages_on_goal):
                    assert (
                        self.rew_all_packages_on_goal.shape[0] == self.world.batch_dim
                    )
                    self.rew[all_packages_on_goal] += self.rew_all_packages_on_goal[
                        all_packages_on_goal
                    ]
                else:
                    self.rew[all_packages_on_goal] += self.rew_all_packages_on_goal

        return self.rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        assert (
            not torch.is_tensor(self.observe_n_packages)
            or len(self.observe_n_packages) == 1
        ), "observe_n_packages must be a scalar"
        all_packages_obs = torch.zeros(
            self.world.batch_dim, 7 * self.observe_n_packages, device=self.world.device
        )
        for i, package in enumerate(self.packages):
            package_obs = []
            package_dist_to_agent = torch.linalg.vector_norm(
                package.state.pos - agent.state.pos, dim=1
            )
            package_observed = torch.ones_like(package_dist_to_agent)
            if self.randomise_n_packages:
                # only observe packages that are active
                package_observed[~self.active_packages[:, i]] = 0
            if self.partial_observability:
                # only observe packages within observation range
                package_observed[package_dist_to_agent > self.observation_range] = 0
            package_obs.append(package.state.pos - package.goal.state.pos)
            package_obs.append(package.state.pos - agent.state.pos)
            package_obs.append(package.state.vel)
            package_obs.append(package.on_goal.unsqueeze(-1))
            package_obs = torch.cat(package_obs, dim=-1)
            all_packages_obs[:, i * 7 : (i + 1) * 7] = (
                package_obs * package_observed.unsqueeze(-1)
            )

        other_agent_obs = []
        if self.observe_other_agents:
            # observe other agent pos and vel
            for other_agent in [
                other_agent for other_agent in self.world.agents if other_agent != agent
            ]:
                other_agent_dist_to_agent = torch.linalg.vector_norm(
                    other_agent.state.pos - agent.state.pos, dim=1
                )
                if self.partial_observability:
                    other_agent_in_obs_range = (
                        other_agent_dist_to_agent <= self.observation_range
                    ).unsqueeze(-1)
                else:
                    other_agent_in_obs_range = torch.ones_like(
                        other_agent_dist_to_agent
                    ).unsqueeze(-1)
                values = [
                    (other_agent.state.pos - agent.state.pos)
                    * other_agent_in_obs_range,
                    other_agent.state.vel * other_agent_in_obs_range,
                ]
                other_agent_obs.extend(values)

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                all_packages_obs,
                *other_agent_obs,
            ],
            dim=-1,
        )

    def done(self):
        if torch.is_tensor(self.terminate_on_goal):
            assert all(self.terminate_on_goal) or not any(
                self.terminate_on_goal
            ), "terminate_on_goal must be either True or False for all environments"

        if (not torch.is_tensor(self.terminate_on_goal) and self.terminate_on_goal) or (
            torch.is_tensor(self.terminate_on_goal) and all(self.terminate_on_goal)
        ):
            return torch.all(
                torch.stack(
                    [package.on_goal for package in self.packages],
                    dim=1,
                ),
                dim=-1,
            )
        else:
            return torch.zeros(
                self.world.batch_dim, dtype=torch.bool, device=self.world.device
            )


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookahead = 0.0  # evaluate u at this value along the spline
        self.start_vel_dist_from_target_ratio = (
            0.5  # distance away from the target for the start_vel to point
        )
        self.start_vel_behind_ratio = 0.5  # component of start vel pointing directly behind target (other component is normal)
        self.start_vel_mag = 1.0  # magnitude of start_vel (determines speed along the whole trajectory, as spline is recalculated continuously)
        self.hit_vel_mag = 1.0
        self.package_radius = 0.15 / 2
        self.agent_radius = -0.02
        self.dribble_slowdown_dist = 0.0
        self.speed = 0.95

    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        self.n_env = observation.shape[0]
        self.device = observation.device
        agent_pos = observation[:, :2]
        agent_vel = observation[:, 2:4]
        package_pos = observation[:, 6:8] + agent_pos
        goal_pos = -observation[:, 4:6] + package_pos
        # control = self.get_action(goal_pos, curr_pos=agent_pos, curr_vel=agent_vel)
        control = self.dribble(agent_pos, package_pos, goal_pos)
        control *= self.speed * u_range
        return torch.clamp(control, -u_range, u_range)

    def dribble(self, agent_pos, package_pos, goal_pos, agent_vel=None):
        package_disp = goal_pos - package_pos
        ball_dist = package_disp.norm(dim=-1)
        direction = package_disp / ball_dist[:, None]
        hit_pos = package_pos - direction * (self.package_radius + self.agent_radius)
        hit_vel = direction * self.hit_vel_mag
        start_vel = self.get_start_vel(
            hit_pos, hit_vel, agent_pos, self.start_vel_mag * 2
        )
        slowdown_mask = ball_dist <= self.dribble_slowdown_dist
        hit_vel[slowdown_mask, :] *= (
            ball_dist[slowdown_mask, None] / self.dribble_slowdown_dist
        )
        return self.get_action(
            target_pos=hit_pos,
            target_vel=hit_vel,
            curr_pos=agent_pos,
            curr_vel=agent_vel,
            start_vel=start_vel,
        )

    def hermite(self, p0, p1, p0dot, p1dot, u=0.0, deriv=0):
        # Formatting
        u = u.reshape((-1,))

        # Calculation
        U = torch.stack(
            [
                self.nPr(3, deriv) * (u ** max(0, 3 - deriv)),
                self.nPr(2, deriv) * (u ** max(0, 2 - deriv)),
                self.nPr(1, deriv) * (u ** max(0, 1 - deriv)),
                self.nPr(0, deriv) * (u**0),
            ],
            dim=1,
        ).float()
        A = torch.tensor(
            [
                [2.0, -2.0, 1.0, 1.0],
                [-3.0, 3.0, -2.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            device=U.device,
        )
        P = torch.stack([p0, p1, p0dot, p1dot], dim=1)
        ans = U[:, None, :] @ A[None, :, :] @ P
        ans = ans.squeeze(1)
        return ans

    def nPr(self, n, r):
        if r > n:
            return 0
        ans = 1
        for k in range(n, max(1, n - r), -1):
            ans = ans * k
        return ans

    def get_start_vel(self, pos, vel, start_pos, start_vel_mag):
        start_vel_mag = torch.as_tensor(start_vel_mag, device=self.device).view(-1)
        goal_disp = pos - start_pos
        goal_dist = goal_disp.norm(dim=-1)
        vel_mag = vel.norm(dim=-1)
        vel_dir = vel.clone()
        vel_dir[vel_mag > 0] /= vel_mag[vel_mag > 0, None]
        goal_dir = goal_disp / goal_dist[:, None]

        vel_dir_normal = torch.stack([-vel_dir[:, 1], vel_dir[:, 0]], dim=1)
        dot_prod = (goal_dir * vel_dir_normal).sum(dim=1)
        vel_dir_normal[dot_prod > 0, :] *= -1

        dist_behind_target = self.start_vel_dist_from_target_ratio * goal_dist
        point_dir = -vel_dir * self.start_vel_behind_ratio + vel_dir_normal * (
            1 - self.start_vel_behind_ratio
        )

        target_pos = pos + point_dir * dist_behind_target[:, None]
        target_disp = target_pos - start_pos
        target_dist = target_disp.norm(dim=1)
        start_vel_aug_dir = target_disp
        start_vel_aug_dir[target_dist > 0] /= target_dist[target_dist > 0, None]
        start_vel = start_vel_aug_dir * start_vel_mag[:, None]
        return start_vel

    def get_action(
        self,
        target_pos,
        target_vel=None,
        start_pos=None,
        start_vel=None,
        curr_pos=None,
        curr_vel=None,
    ):
        if curr_pos is None:  # If None, target_pos is assumed to be a relative position
            curr_pos = torch.zeros(target_pos.shape, device=self.device)
        if curr_vel is None:  # If None, curr_vel is assumed to be 0
            curr_vel = torch.zeros(target_pos.shape, device=self.device)
        if (
            start_pos is None
        ):  # If None, start_pos is assumed to be the same as curr_pos
            start_pos = curr_pos
        if target_vel is None:  # If None, target_vel is assumed to be 0
            target_vel = torch.zeros(target_pos.shape, device=self.device)
        if start_vel is None:  # If None, start_vel is calculated with get_start_vel
            start_vel = self.get_start_vel(
                target_pos, target_vel, start_pos, self.start_vel_mag * 2
            )

        u_start = torch.ones(curr_pos.shape[0], device=self.device) * self.lookahead
        des_curr_pos = self.hermite(
            start_pos,
            target_pos,
            start_vel,
            target_vel,
            u=u_start,
            deriv=0,
        )
        des_curr_vel = self.hermite(
            start_pos,
            target_pos,
            start_vel,
            target_vel,
            u=u_start,
            deriv=1,
        )
        des_curr_pos = torch.as_tensor(des_curr_pos, device=self.device)
        des_curr_vel = torch.as_tensor(des_curr_vel, device=self.device)
        control = 0.5 * (des_curr_pos - curr_pos) + 0.5 * (des_curr_vel - curr_vel)
        return control


if __name__ == "__main__":
    render_interactively(
        __file__,
        n_packages=3,
        randomise_n_packages=True,
        control_two_agents=True,
        rew_package_on_goal=1,
        rew_all_packages_on_goal=5,
        terminate_on_goal=False,
        package_mass=2.0,
        observe_other_agents=True,
        # partial_observability=True,
    )
