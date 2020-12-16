"""Loads the environment and stores the training history.
"""
from collections import defaultdict

import gym
import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, 
                env_name,
                solved_reward,
                aggregate_every,
                print_every,
                render_every,):
        """Initializes the environment.
        
        Args:
            evn_name: [Text] the name of the OpenAI environment
            solved_reward: [Int] the env is solved when rewards over aggregate_every epochs > solved_reward
            aggregate_every: [Int] aggregate the running rewards for aggregate_every epochs to compute rewards
            print_every: [Int] print the result after print_every epochs
            render_every: [Int] render the play after render_every epochs
        """
        self.env_name = env_name
        self.print_every = print_every
        self.render_every = render_every
        self.aggregate_every = aggregate_every
        self.solved_reward = solved_reward

        self.env = gym.make(self.env_name)
        self.get_env_params()

        self.episode = 0
        self.running_reward = defaultdict(int)
        self.average_length = defaultdict(int)


    def get_env_params(self):
        """Gets state_dim, action_dim, and upper_bound for the environment.
        """
        self.state_dim = self.env.observation_space.shape[0]
        try:
            # Continuous environments
            self.action_dim = self.env.action_space.shape[0]
            self.upper_bound = self.env.action_space.high[0]
        except:
            # Discrete environments
            self.action_dim = self.env.action_space.n

        print('Environment: %s \t State_dim = %2d \t Action_dim = %2d'
            %(self.env_name, self.state_dim, self.action_dim))

    def start_episode(self):
        """Starts an episode.
        """
        self.episode += 1
        return self.env.reset()

    def update_history_metrics(self, reward):
        """Updates history metrics after an action.
        """
        self.running_reward[self.episode] += reward
        self.average_length[self.episode] += 1

        if self.episode % self.render_every == 0:
            self.env.render()

    def is_finished(self):
        """Checks if we got the solved rewards over aggregate_every episodes.
        """
        self.env.close()
        evaluting_episodes = range(self.episode + 1)[- self.aggregate_every:]
        running_reward = np.mean([self.running_reward[e] for e in evaluting_episodes])
        average_length = np.mean([self.average_length[e] for e in evaluting_episodes])

        if self.episode % self.print_every == 0:
            print('[Episode %4d] average_length: %4d \tRewards: %4d' 
                    %(self.episode, int(average_length), int(running_reward)))

        if self.episode > self.aggregate_every and running_reward > self.solved_reward:
            print("########## Solved! ##########")
            print('[Episode %4d] average_length: %4d \tRewards: %4d' 
                    %(self.episode, int(average_length), int(running_reward)))
            return True
        return False

    def plot_history_metric(self):
        """Plots the training history when the training is finished.
        """
        rewards = [self.running_reward[e] for e in range(1, self.episode + 1)]
        plt.plot(rewards, 'k', linewidth=2)
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel('Running Rewards', fontsize=12)
        plt.title('Training History', fontsize=12)
        plt.show()