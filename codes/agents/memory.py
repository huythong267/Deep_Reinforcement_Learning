"""Defines a memory class to store the data of RL exploration.
"""
from collections import defaultdict, deque
import numpy as np
import tensorflow as tf

BUFFER_SIZE = 50000
BATCH_SIZE = 64
GAMMA = 0.99


class Memory:
    def __init__(self, 
                buffer_size=BUFFER_SIZE, 
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                compute_discounted_rewards=False,
                normalize_discounted_rewards=True,
                dtype=tf.float32):
        """Memorizes the experiences in the training.

        The parameters we want to restore are:
            states, actions, rewards, next_states, dones, actionProbs
        We include the actionProbs for use in the ppo method.

        Args:
            buffer_size: the size of the memory deque
            batch_size: the size of a batch when sampled from memory
            gamma: discounted factor in the Bellman equation
            compute_discounted_rewards: True to compute dicounted rewards when converting to np arrays
            normalized_discounted_rewards: True to normalize the discounted rewards
            dtype: data type when sampled a batch from memory
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.compute_discounted_rewards = compute_discounted_rewards
        self.normalize_discounted_rewards = normalize_discounted_rewards
        self.dtype = dtype
        self.gamma = gamma

        self.reset()

    def reset(self):
        """Resets all the parameters.
        """
        self.states = deque(maxlen=self.buffer_size)
        self.actions = deque(maxlen=self.buffer_size)
        self.rewards = deque(maxlen=self.buffer_size)
        self.next_states = deque(maxlen=self.buffer_size)
        self.dones = deque(maxlen=self.buffer_size)
        self.actionProbs = deque(maxlen=self.buffer_size)

    def add(self, one_step):
        """Adds the experience from one_step to memory.

        Args:
            one_step: a tuple of len 6 with
                    state, action, reward, next_state, done, actionProb
        """
        assert len(one_step) == 6
        state, action, reward, next_state, done, actionProb = one_step
            # = [list(v) for v in one_step]

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.actionProbs.append(actionProb)

    def compute_discounted_rewards_arr(self):
        """Computes discounted rewards.

        Follows the Bellman equation:
            V(s) = Reward + gamma * V(next_s)
        """
        self.discounted_rewards  = []

        discounted_reward = 0
        for reward, done in zip(list(self.rewards)[::-1], list(self.dones)[::-1]):
            discounted_reward = reward if done else reward + (self.gamma * discounted_reward)
            self.discounted_rewards.insert(0, discounted_reward)

        self.discounted_rewards_arr = np.array(self.discounted_rewards)
        self.discounted_rewards_arr = self.discounted_rewards_arr.reshape(-1, 1)

        if self.normalize_discounted_rewards:
            mean = self.discounted_rewards_arr.mean()
            std = self.discounted_rewards_arr.std() + 1e-5
            self.discounted_rewards_arr = (self.discounted_rewards_arr - mean) / std

    def to_numpy_arrays(self):
        """Converts from the deque to numpy arr.
        """
        self.states_arr = np.array(self.states)
        self.actions_arr = np.array(self.actions)
        self.rewards_arr = np.array(self.rewards).reshape(-1, 1)
        self.next_states_arr = np.array(self.next_states)
        self.dones_arr = np.array(self.dones).reshape(-1, 1)
        self.actionProbs_arr = np.array(self.actionProbs)

        if self.compute_discounted_rewards:
            self.compute_discounted_rewards_arr()

    def sample(self):
        """Samples experiences from the memory.

        Use 5 basic experiences: states, actions, rewards, next_states, dones
        """
        if self.__len__() < self.batch_size * 2:
            return None

        indices_batch = np.random.choice(self.__len__(), self.batch_size)

        states_batch = tf.convert_to_tensor(np.array(self.states)[indices_batch], dtype=self.dtype)
        actions_batch = tf.convert_to_tensor(np.array(self.actions)[indices_batch], dtype=self.dtype)
        rewards_batch = tf.convert_to_tensor(np.array(self.rewards)[indices_batch].reshape(-1, 1), dtype=self.dtype)
        next_states_batch = tf.convert_to_tensor(np.array(self.next_states)[indices_batch], dtype=self.dtype)
        dones_batch = tf.convert_to_tensor(np.array(self.dones)[indices_batch].reshape(-1, 1), dtype=self.dtype)

        sample_experiences = (states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)
        return sample_experiences

    def __len__(self):
        return len(self.states)