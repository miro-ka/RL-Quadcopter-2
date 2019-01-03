import numpy as np
import tensorflow as tf
from nets.qnet import QNet
from nets.memory import Memory
import logging


EPISODES = 1000

class DQNAgent:

    session = None

    def __init__(self, task):
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.hidden_size = 64
        self.count = 0
        self.score = 0.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory_size = 2000
        self.memory = Memory(max_size=self.memory_size)
        self.task = task
        self.model = self._build_model()

    def _build_model(self):
        tf.reset_default_graph()
        self.session = tf.Session()
        qnet = QNet(name='main',
                    hidden_size=self.hidden_size,
                    learning_rate=self.learning_rate,
                    action_size=self.action_size)
        self.session.run(tf.global_variables_initializer())
        return qnet

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state):
        """
        Returns action based on the given state
        """
        # Explore - return random states
        if np.random.rand() <= self.epsilon:
            actions = np.zeros(self.action_size)
            actions[np.random.randint(self.action_size)] = 1
            return actions
        # Exploit - Get action from Q-network
        feed = {self.model.inputs_: state.reshape((1, *state.shape))}
        Qs = self.session.run(self.model.output, feed_dict=feed)
        action_idx = np.argmax(Qs)
        actions = np.zeros(self.action_size)
        actions[action_idx] = 1
        return actions

    def retrain(self, batch_size):
        batch = self.memory.sample(batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])

        # Train network
        target_Qs = self.session.run(self.model.output, feed_dict={self.model.inputs_: next_states})

        # Set target_Qs to 0 for states where episode ends
        episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
        target_Qs[episode_ends] = (0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0)

        targets = rewards + self.gamma * np.max(target_Qs, axis=1)

        loss, _ = self.session.run([self.model.loss, self.model.opt],
                                   feed_dict={self.model.inputs_: states,
                                   self.model.targetQs_: targets,
                                   self.model.actions_: actions})

        # logging.info('loss: ' + str(loss))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def reset_episode(self):
        self.count = 0
        self.score = 0.0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.score += reward
        self.count += 1

        if self.memory.size() > self.batch_size:
            self.retrain(self.batch_size)

    @staticmethod
    def actions_to_rotor_velocity(rotor_velocity, action, delta_change=0.05):
        """
        Updates and returns rotor_velocity based on the input actions
        """
        action_index = action.tolist().index(1)
        # Rotor 1
        if action_index == 0:
            rotor_velocity[0] += delta_change
        elif action_index == 1:
            rotor_velocity[0] -= delta_change
        # Rotor 2
        elif action_index == 2:
            rotor_velocity[1] += delta_change
        elif action_index == 3:
            rotor_velocity[1] -= delta_change
        # Rotor 3
        elif action_index == 4:
            rotor_velocity[2] += delta_change
        elif action_index == 5:
            rotor_velocity[2] -= delta_change
        # Rotor 4
        elif action_index == 6:
            rotor_velocity[3] += delta_change
        elif action_index == 7:
            rotor_velocity[3] -= delta_change
        # Rotor 1234
        elif action_index == 8:
            rotor_velocity[0] += delta_change
            rotor_velocity[1] += delta_change
            rotor_velocity[2] += delta_change
            rotor_velocity[3] += delta_change
        elif action_index == 9:
            rotor_velocity[0] -= delta_change
            rotor_velocity[1] -= delta_change
            rotor_velocity[2] -= delta_change
            rotor_velocity[3] -= delta_change
        # Rotor 12
        elif action_index == 10:
            rotor_velocity[0] += delta_change
            rotor_velocity[1] += delta_change
        # Rotor 12
        elif action_index == 11:
            rotor_velocity[0] -= delta_change
            rotor_velocity[1] -= delta_change
        # Rotor 23
        elif action_index == 12:
            rotor_velocity[1] += delta_change
            rotor_velocity[2] += delta_change
        # Rotor 23
        elif action_index == 13:
            rotor_velocity[1] -= delta_change
            rotor_velocity[2] -= delta_change
        # Rotor 34
        elif action_index == 14:
            rotor_velocity[2] += delta_change
            rotor_velocity[3] += delta_change
        # Rotor 34
        elif action_index == 15:
            rotor_velocity[2] -= delta_change
            rotor_velocity[3] -= delta_change

        # If one of the velocity is negative just set it to zero
        #for idx, velocity in enumerate(rotor_velocity):
        #    if velocity < 0.0:
        #        rotor_velocity[idx] = 0.0
        return rotor_velocity