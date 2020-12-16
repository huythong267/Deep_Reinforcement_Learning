"""Develops a DQN agent to solve the discrete tasks.
"""
import os
import numpy as np
import tensorflow as tf

import codes.agents.memory as memory
import codes.agents.models as models

# GAMMA = 0.99
# LR_CRITIC = 0.0005
# TAU_CRITIC = 0.001
# LEARN_EVERY = 4

# EPS_DECAY = 0.995
# EPS_MIN = 0.01

GAMMA = 0.99
LR_CRITIC = 0.001
TAU_CRITIC = 0.01
LEARN_EVERY = 2

EPS_DECAY = 0.99
EPS_MIN = 0.01


def softmax(x):
    """ applies softmax to an input x"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Agent(models.Agents):
    def __init__(self, 
                state_dim,
                action_dim,
                gamma=GAMMA,
                lr_critic=LR_CRITIC,
                tau_critic=TAU_CRITIC,
                learn_every=LEARN_EVERY,
                eps_decay=EPS_DECAY):
        """Initializes a DQN agent.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.lr_critic = lr_critic
        self.tau_critic = tau_critic
        self.learn_every = learn_every
        self.algo_name = 'dqn'
        self.eps = 1.0
        self.eps_decay = eps_decay
        
        self._build_critic()
        self.memory = memory.Memory(gamma=self.gamma)

        self._print_hyper_parameters()

    def get_action_and_actionProb(self, state, training=True):
        """Gets action and actionProb when given a state.

        Notes: actionProb is reversed for the PPO algorithm.
        """
        Q_values = self.critic.predict(state.reshape(1, -1)).flatten()
        actionProb = softmax(Q_values)
        if training:
            if np.random.random() < self.eps:
                action = np.random.choice(self.action_dim, p=actionProb)
            else:
                action = np.argmax(Q_values)
        else:
            action = np.argmax(Q_values)
        return action, actionProb

    def step(self, one_step):
        """Updates the agent after a step of action.

        Args:
            one_step: a tuple of length = 6 with
                    state, action, reward, next_state, done, actionProb
        """
        self.memory.add(one_step)

        if np.random.randint(0, self.learn_every) % self.learn_every == 0:
            experiences = self.memory.sample()
            if not experiences:
                return
            states, actions, rewards, next_states, dones = experiences
            self._update_weights(states, actions, rewards, next_states, dones)
            self._transfer_weights(self.critic_target, self.critic, self.tau_critic)

    def update_after_an_episode(self):
        """Update the agent after playing an episode.
        """
        self.eps = max(self.eps * self.eps_decay, EPS_MIN)

    @tf.function
    def _update_weights(self, states, actions, rewards, next_states, dones):
        """Updates the trainable weights of the agent.
        """
        # Trains self.critic
        with tf.GradientTape() as tape:
            predicted_next_values = self.critic_target(next_states) # (N, 4)
            best_predicted_next_values = tf.math.reduce_max(predicted_next_values, axis=-1, keepdims=True)

            bellman_values = rewards + self.gamma * best_predicted_next_values * (1 - dones)

            Q_values = self.critic(states, training=True)
            actions_onehot = tf.one_hot(tf.cast(actions, tf.int32), self.action_dim)
            predicted_values = tf.math.reduce_sum(Q_values * actions_onehot, axis = -1, keepdims=True)

            critic_loss = tf.math.reduce_mean(tf.math.square(bellman_values - predicted_values))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic_optim.apply_gradients(zip(critic_grad, self.critic.trainable_weights))

    #######################################################################################################
    #   Follow the function in Save & Load of the agent
    #######################################################################################################
    def _get_a_dictionary_of_name_to_model(self, env_name):
        return {
            '%s_%s_critic.h5' %(self.algo_name, env_name): self.critic,
            '%s_%s_critic_target.h5' %(self.algo_name, env_name): self.critic_target,
        }

    #######################################################################################################
    #   Build the models for the agent
    #######################################################################################################
    def _build_critic(self,):
        """Builds the critic models.
        """
        critic_settings = {
            'input_size': self.state_dim,
            'output_size': self.action_dim,
            'output_activation': 'linear',
        }
        self.critic = models.build_dnn_models(**critic_settings)
        self.critic_target = models.build_dnn_models(**critic_settings)
        self.critic_optim = tf.keras.optimizers.Adam(lr=self.lr_critic)

    def _print_hyper_parameters(self,):
        """Prints hyper parameters.
        """
        print('DQN agent with decay')
        print('learning_rate: ', self.lr_critic)
        print('weight_transfer tau: ', self.tau_critic)
        print('learn_every: ', self.learn_every)
        print('eps_decay : ', self.eps_decay)
        print()