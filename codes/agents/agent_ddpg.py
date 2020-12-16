"""Develops a DDPG agent to solve the continuous tasks.

Reference: [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import codes.agents.memory as memory
import codes.agents.models as models


GAMMA = 0.99
LR_ACTOR = 0.001
LR_CRITIC = 0.002
TAU_ACTOR = 0.005
TAU_CRITIC = 0.005
LEARN_EVERY = 2
SIGMA_NOISE = 0.01


class OUAactionNoise:
    """Adds noise to continous actions.
    """
    def __init__(self, size=1, mu=0, sigma=0.01, theta=0.015):
        self.theta = theta
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class Agent(models.Agents):
    def __init__(self, 
                state_dim,
                action_dim,
                output_scale,
                gamma=GAMMA,
                lr_actor=LR_ACTOR, lr_critic=LR_CRITIC,
                tau_actor=TAU_ACTOR, tau_critic=TAU_CRITIC,
                learn_every=LEARN_EVERY,
                sigma_noise=SIGMA_NOISE,
                ):
        """Initializes a DDPG agent.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_scale = output_scale

        self.gamma = gamma
        self.lr_actor, self.lr_critic = lr_actor, lr_critic
        self.tau_actor, self.tau_critic = tau_actor, tau_critic
        self.learn_every = learn_every
        self.sigma_noise = sigma_noise * self.output_scale
        self.algo_name = 'ddpg'

        self._build_actor()
        self._build_critic()
        self.memory = memory.Memory(gamma=self.gamma)
        self.noise = OUAactionNoise(size=self.action_dim, sigma=self.sigma_noise)
        self._print_hyper_parameters()

    def get_action_and_actionProb(self, state, training=True):
        """Gets action and actionProb when given a state.

        Notes: actionProb is reversed for the PPO algorithm.
        """
        action = self.actor.predict(state.reshape(1, -1)).flatten()
        actionProb = action
        if training:
            action += self.noise.sample()
            action = np.clip(action, -self.output_scale, self.output_scale)
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
            self._update_weights(experiences)
            self._transfer_weights(self.actor_target, self.actor, self.tau_actor)
            self._transfer_weights(self.critic_target, self.critic, self.tau_critic)

    def update_after_an_episode(self):
        """Updates the agent after a step of action.

        Args:
            one_step: a tuple of len 6 with
                    state, action, reward, next_state, done, actionProb
        """
        self.noise.reset()

    @tf.function
    def _update_weights(self, experiences):
        """Updates the trainable weights of actors and critics.
        """
        assert len(experiences) == 5
        states, actions, rewards, next_states, dones = experiences

        # Trains self.critic
        with tf.GradientTape() as tape:
            predicted_next_actions = self.actor_target(next_states)
            predicted_next_values = self.critic_target([next_states, predicted_next_actions])

            bellman_values = rewards + self.gamma * predicted_next_values * (1 - dones)
            predicted_values = self.critic([states, actions], training=True)

            critic_loss = tf.math.reduce_mean(tf.math.square(bellman_values - predicted_values))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic_optim.apply_gradients(zip(critic_grad, self.critic.trainable_weights))

        # Trains self.actor
        with tf.GradientTape() as tape:
            predicted_actions = self.actor(states, training=True)
            predicted_values = self.critic([states, predicted_actions])
            actor_loss = - tf.math.reduce_mean(predicted_values)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_optim.apply_gradients(zip(actor_grad, self.actor.trainable_weights))

    #######################################################################################################
    #   Save & Load - Refer to models.Agents
    #######################################################################################################
    def _get_a_dictionary_of_name_to_model(self, env_name):
        return {
            '%s_%s_actor.h5' %(self.algo_name, env_name): self.actor,
            '%s_%s_actor_target.h5' %(self.algo_name, env_name): self.actor_target,
            '%s_%s_critic.h5' %(self.algo_name, env_name): self.critic,
            '%s_%s_critic_target.h5' %(self.algo_name, env_name): self.critic_target,
        }

    #######################################################################################################
    #   Build the models for the agent
    #######################################################################################################
    def _build_actor(self,):
        """Builds the actor models.
        """
        actor_settings = {
            'input_size': self.state_dim,
            'output_size': self.action_dim,
            'output_activation': 'tanh',
            'output_scale': self.output_scale,
        }
        self.actor = models.build_dnn_models(**actor_settings)
        self.actor_target = models.build_dnn_models(**actor_settings)
        self.actor_target.set_weights(self.actor.get_weights())

        self.actor_optim = tf.keras.optimizers.Adam(lr=self.lr_actor)

    def _build_critic(self,):
        """Builds the critic models.
        """
        critic_settings = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }
        self.critic = models.build_Q_models(**critic_settings)
        self.critic_target = models.build_Q_models(**critic_settings)
        self.critic_target.set_weights(self.critic.get_weights())

        self.critic_optim = tf.keras.optimizers.Adam(lr=self.lr_critic)

    def _print_hyper_parameters(self,):
        """Prints hyper parameters.
        """
        print('DDPG agent')
        print('output_scale: ', self.output_scale)
        print('learning_rate actor = %s, critic = %s' %(self.lr_actor, self.lr_critic))
        print('weight_transfer tau actor = %s, critic = %s' %(self.tau_actor, self.tau_critic))
        print('learn_every: ', self.learn_every)
        print()
        print('sigma_noise: ', self.sigma_noise)
        print('Examples of OUAactionNoise:')
        noise = []
        for _ in range(10000):
            noise.append(self.noise.sample())
        plt.plot(noise); plt.show()
        self.noise.reset()
        print()