import sys
import time
import logging
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


OUTPUT_FOLDER = 'out/'
logging.basicConfig(level=logging.INFO)

import pandas as pd
from agents.policy_search import PolicySearch_Agent
from task import Task



def main(agent_name='policy-search'):
    logging.info("Starting simulation with agent - '" + agent_name + "'")

    num_episodes = 1000
    target_pos = np.array([0., 0., 10.])
    task = Task(target_pos=target_pos)
    if agent_name == 'policy-search':
        agent = PolicySearch_Agent(task)
    elif agent_name == 'q-learning':
        agent = None
    else:
        logging.warning("Invalid model selected")
        return

    score_buffer = []
    for i_episode in range(1, num_episodes + 1):
        state = agent.reset_episode()  # start a new episode
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(reward, done)
            state = next_state
            if done:
                score_buffer.append((i_episode, agent.score))
                logging.info("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                    i_episode, agent.score, agent.best_score, agent.noise_scale))  # [debug]
                break
        sys.stdout.flush()

    # plot score
    df = pd.DataFrame(score_buffer, columns=['episode_idx', 'score'])
    plot = df.plot(x='episode_idx', y='score', title="Agent's score")
    plot_file = OUTPUT_FOLDER + 'score_' + agent_name + '.png'
    plot.get_figure().savefig(plot_file)
    return score_buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quadcopter reinforcement gym simulation')
    parser.add_argument('-a', '--agent', help="Simulation agent. Valid values (policy-search)", required=True)
    args = vars(parser.parse_args())
    agent_name = args['agent']
    sim_start_time = time.time()
    main(agent_name)
    sim_end_time = time.time()
    logging.info("Simulation done in (sec): " + str(int(sim_end_time-sim_start_time)))
