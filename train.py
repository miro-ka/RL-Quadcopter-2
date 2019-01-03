import time
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from agents.dqn_agent import DQNAgent
from task import Task


OUTPUT_FOLDER = 'out/'
logging.basicConfig(level=logging.INFO)
num_episodes = 1000
score_buffer = []
sim_pose = []
sim_velocity = []


def initialize_task(init_pose, init_velocities, target_pos, init_angle_velocities):
    """
    Initializes task
    """
    task = Task(init_pose=init_pose, target_pos=target_pos, init_velocities=init_velocities,
                init_angle_velocities=init_angle_velocities)
    return task

def train():
    """
    Trains model
    """
    logging.info("Starting simulation...")

    # Initialization
    task = initialize_task(init_pose=np.array([0., 0., 0., 0., 0., 0.]),
                           target_pos=np.array([0., 0., 10.]),
                           init_velocities = np.array([0., 0., 0.]),
                           init_angle_velocities = np.array([0., 0., 0.]))
    agent = DQNAgent(task)

    for i_episode in range(1, num_episodes + 1):
        # Start a new episode
        state = agent.reset_episode()
        # We initialize velocity 4 x 0's vector, representing velocity for each rotor
        rotor_velocity = [0.0, 0.0, 0.0, 0.0]
        while True:
            action = agent.act(state)
            rotor_velocity = agent.actions_to_rotor_velocity(rotor_velocity, action)
            next_state, reward, done = task.step(action)
            # print(rotor_velocity, action.tolist().index(1), reward, agent.score)
            agent.step(reward, done)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # Save simulation states
            sim_pose.append(task.sim.pose)
            sim_velocity.append(task.sim.v)
            if done:
                score_buffer.append((i_episode, agent.score))
                print("\rEpisode ={:4d}, score ={:7.5f}".format(i_episode, agent.score), end="")
                # logging.info("Episode: " + str(i_episode) + ", agent_score: " + str(agent.score))
                break

    # plot score
    df = pd.DataFrame(score_buffer, columns=['episode_idx', 'score'])
    plot = df.plot(x='episode_idx', y='score', title="Agent's score")
    plot_file = OUTPUT_FOLDER + 'score_dqn.png'
    plot.get_figure().savefig(plot_file)


if __name__ == "__main__":
    sim_start_time = time.time()
    train()
    sim_end_time = time.time()
    logging.info("Training done in (sec): " + str(int(sim_end_time-sim_start_time)))
