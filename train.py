import time
import sys
import numpy as np
import pandas as pd
import matplotlib
from task import Task
from agents.agent import Agent
matplotlib.use('TkAgg')


OUTPUT_FOLDER = 'out/'
score_buffer = []


def train():
    """
    Trains model
    """
    print("Starting simulation...")

    num_episodes = 1000
    task = Task(init_pose=np.array([0., 0., 10., 0., 0., 0.]),
                target_pos=np.array([0., 0., 20.]),
                init_velocities=np.array([0., 0., 0.]),
                init_angle_velocities=np.array([0., 0., 0.]))

    agent = Agent(task)

    for i_episode in range(1, num_episodes + 1):
        state = agent.reset_episode()  # start a new episode
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            if done:
                score_buffer.append((i_episode, reward))
                print("\rEpisode = {:4d}, score = {:7.3f}".format(
                    i_episode, reward), end="")
                break
        sys.stdout.flush()

    # plot score
    df = pd.DataFrame(score_buffer, columns=['episode_idx', 'score'])
    plot = df.plot(x='episode_idx', y='score', title="Agent's score")
    plot_file = OUTPUT_FOLDER + 'score_ddpg.png'
    plot.get_figure().savefig(plot_file)


if __name__ == "__main__":
    sim_start_time = time.time()
    train()
    sim_end_time = time.time()
    print("Training done in (sec):" + str(int(sim_end_time-sim_start_time)))
