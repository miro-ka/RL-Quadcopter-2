import time
import logging
import numpy as np
import pandas as pd
import matplotlib
import tensorflow as tf
matplotlib.use('TkAgg')
from agents.ddpg_agent import ActorNetwork, OrnsteinUhlenbeckActionNoise, CriticNetwork, build_summaries
from task import Task
from replay_buffer import ReplayBuffer


OUTPUT_FOLDER = 'out/'

score_buffer = []
random_seed = 2222
actors_learning_rate = 0.01
critics_learning_rate = 0.01
soft_target_update = 0.001
minibatch_size = 64
gamma = 0.99  # discount factor for critic updates
buffer_size = 1000000
max_episodes = 10000
max_episode_len = 1000


def train():
    """
    Trains model
    """
    print("Starting simulation...")

    with tf.Session() as sess:
        # Initialization
        task = Task(init_pose=np.array([0., 0., 0., 0., 0., 0.]),
                    target_pos=np.array([0., 0., 10.]),
                    init_velocities=np.array([0., 0., 0.]),
                    init_angle_velocities=np.array([0., 0., 0.]),
                    action_size=4)

        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

        state_dim = task.state_size
        action_dim = task.action_size
        action_bound = task.action_high

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             actors_learning_rate, soft_target_update,
                             minibatch_size)

        critic = CriticNetwork(sess, state_dim, action_dim,
                               critics_learning_rate, soft_target_update,
                               gamma,
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        # Start training
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('summary_dir', sess.graph)

        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()

        # Initialize replay memory
        replay_buffer = ReplayBuffer(buffer_size, random_seed)

        for i in range(max_episodes):

            s = task.reset()

            ep_reward = 0
            ep_ave_max_q = 0

            for j in range(max_episode_len):

                # Added exploration noise
                # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
                action = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

                next_state, reward, done = task.step(action[0])

                replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward,
                                  done, np.reshape(next_state, (actor.s_dim,)))

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if replay_buffer.size() > minibatch_size:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(minibatch_size)

                    # Calculate targets
                    target_q = critic.predict_target(
                        s2_batch, actor.predict_target(s2_batch))

                    y_i = []
                    for k in range(minibatch_size):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + critic.gamma * target_q[k])

                    # Update the critic given the targets
                    predicted_q_value, _ = critic.train(
                        s_batch, a_batch, np.reshape(y_i, (minibatch_size, 1)))

                    ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)
                    actor.train(s_batch, grads[0])

                    # Update target networks
                    actor.update_target_network()
                    critic.update_target_network()

                s = next_state
                ep_reward += reward

                if done and j > 0:
                    score_buffer.append((i, ep_reward))

                    print('\r| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward),
                                                                                   i, (ep_ave_max_q / float(j))), end="")
                    break

    # plot score
    df = pd.DataFrame(score_buffer, columns=['episode_idx', 'score'])
    plot = df.plot(x='episode_idx', y='score', title="Agent's score")
    plot_file = OUTPUT_FOLDER + 'score_ddpg.png'
    plot.get_figure().savefig(plot_file)


if __name__ == "__main__":
    sim_start_time = time.time()
    train()
    sim_end_time = time.time()
    print("Training done in (sec): " + str(int(sim_end_time-sim_start_time)))
