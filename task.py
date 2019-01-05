import numpy as np
from physics_sim import PhysicsSim
import math


class Task:
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None,
                 action_size=4):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            action_size: number of actions
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        self.duration_step = 0
        self.state_size = self.action_repeat * 6
        self.action_low = 300
        self.action_high = 900
        self.action_size = action_size
        self.previous_pos = self.sim.init_pose[:3]

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # *** FLY_UP REWARD ***
        # punish if the rotors velocity distance (in our case all rotors should have the same value)
        # rotors_speed_delta = (np.average(rotor_speeds) - np.max(rotor_speeds)) / 100

        reward = 0
        # Give Reward if we achieve target/goal
        if self.sim.pose[2] >= self.target_pos[2]:
            reward += 5

        # Punish if we are losing altitude
        # altitude_gain_reward = -20 if self.sim.init_pose[2] >= self.sim.pose[2] else 0.5

        # Reward if the altitude is increasing
        altitude_gain_reward = 5 if self.previous_pos[2] <= self.sim.pose[2] else -3
        reward += altitude_gain_reward
        # print('self.previous_pos[2]:', self.previous_pos[2], "self.sim.pose[2]:", self.sim.pose[2], "reward:", reward)
        # target_distance = self.sim.init_pose[2] - self.target_pos[2]
        # z_distance_offset = (self.sim.pose[2] - self.target_pos[2])
        # z_delta = z_distance_offset - (self.target_pos[2] - self.sim.pose[2])
        # reward += z_delta
        # Reward current-target position
        target_distance_reward = 1. - .3*(abs(self.sim.pose[2] - self.target_pos[2])).sum()
        # reward = altitude_gain_reward + z_distance_offset # + target_distance_reward  # + rotors_speed_delta
        reward += target_distance_reward
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        self.duration_step += 1
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.duration_step = 0
        self.previous_pos = self.sim.init_pose[:3]
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
