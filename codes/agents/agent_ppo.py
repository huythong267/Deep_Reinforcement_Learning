"""Develops a PPO agent to solve the discrete tasks.

Reference: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
"""
import os
import numpy as np
import tensorflow as tf

import codes.agents.memory as memory
import codes.agents.models as models

GAMMA = 0.99
LR_ACTOR = 0.002
LR_CRITIC = 0.002

PPO_UPDATE_TIMESTEP = 2000
PPO_EPSILON = 0.2
PPO_EPOCHS = 8


class Agent(models.Agents):
    def __init__(self, 
                state_dim,
                action_dim,
                gamma=GAMMA,
                lr_actor=LR_ACTOR, lr_critic=LR_CRITIC,
                ppo_epsilon=PPO_EPSILON,
                ppo_update_timestep=PPO_UPDATE_TIMESTEP,
                ppo_epochs=PPO_EPOCHS
                ):
        """Initializes a PPO agent.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.lr_actor, self.lr_critic = lr_actor, lr_critic
        self.ppo_epsilon = ppo_epsilon
        self.ppo_update_timestep = ppo_update_timestep
        self.ppo_epochs = ppo_epochs

        self.algo_name = 'ppo'

        self._build_actor()
        self._build_critic()
        self.memory = memory.Memory(
            gamma=self.gamma, 
            compute_discounted_rewards=True
        )
        self._print_hyper_parameters()

    def get_action_and_actionProb(self, state, training=True):
        """Gets action and actionProb when given a state.

        Notes: actionProb is reversed for the PPO algorithm.
        """
        actionProb = self.actor.predict(state.reshape(1, -1)).flatten()
        if training:
            action = np.random.choice(self.action_dim, p=actionProb)
        else:
            action = np.argmax(actionProb)
        return action, actionProb

    def step(self, one_step):
        """Updates the agent after a step of action.

        Args:
            one_step: a tuple of length = 6 with
                    state, action, reward, next_state, done, actionProb
        """
        self.memory.add(one_step)

    def update_after_an_episode(self):
        """Update the agent after playing an episode.
        """
        if len(self.memory) < self.ppo_update_timestep:
            return

        self.memory.to_numpy_arrays()
        states = tf.convert_to_tensor(self.memory.states_arr)
        actions_onehot = tf.one_hot(self.memory.actions_arr, self.action_dim)
        values = tf.convert_to_tensor(self.memory.discounted_rewards_arr, dtype=tf.float32)
        old_actionProbs = tf.convert_to_tensor(self.memory.actionProbs_arr)
        old_probs = tf.math.reduce_sum(actions_onehot * old_actionProbs, axis=-1) + 1e-5 # (N,)

        # experiences = states, actions_onehot, values, old_probs
        for _ in range(self.ppo_epochs):
            self._update_weights(states, actions_onehot, values, old_probs)

        self.memory.reset()

    @tf.function
    def _update_weights(self, states, actions_onehot, values, old_probs):
        """Updates the trainable weights of the agent.
        """
        # Train critic
        with tf.GradientTape() as tape:
            predicted_values = self.critic(states, training=True)
            value_loss = tf.math.reduce_mean(tf.math.square(values - predicted_values))

        critic_grad = tape.gradient(value_loss, self.critic.trainable_weights)
        self.critic_optim.apply_gradients(zip(critic_grad, self.critic.trainable_weights))

        # Train actor
        with tf.GradientTape() as tape:
            actionProbs = self.actor(states, training=True) # (N, 4)

            this_prob = tf.math.reduce_sum(actionProbs * actions_onehot, axis=-1) # (N, 4) * (N, 4) --> axis=-1 --> (N,) 

            ratio = tf.reshape(this_prob / old_probs, (-1, 1)) # (N, 1)
            advantages = values - predicted_values # (N, 1)

            surr_loss1 = ratio * advantages  # (N, 1)
            surr_loss2 = tf.clip_by_value(
                ratio, 
                clip_value_min = 1 - self.ppo_epsilon, 
                clip_value_max = 1 + self.ppo_epsilon) * advantages
            surr_loss = - tf.math.reduce_mean(tf.math.minimum(surr_loss1, surr_loss2))
            entropy_loss = - tf.math.reduce_mean( - actionProbs * tf.math.log(actionProbs + 1e-5) )

            actor_loss = surr_loss + 0.01 * entropy_loss

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_optim.apply_gradients(zip(actor_grad, self.actor.trainable_weights))

    #######################################################################################################
    #   Follow the function in Save & Load of the agent
    #######################################################################################################
    def _get_a_dictionary_of_name_to_model(self, env_name):
        return {
            '%s_%s_actor.h5' %(self.algo_name, env_name): self.actor,
            '%s_%s_critic.h5' %(self.algo_name, env_name): self.critic,
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
            'output_activation': 'softmax',
        }
        self.actor = models.build_dnn_models(**actor_settings)
        self.actor_optim = tf.keras.optimizers.Adam(lr=self.lr_actor)


    def _build_critic(self,):
        """Builds the critic models.
        """
        critic_settings = {
            'input_size': self.state_dim,
            'output_size': 1,
            'output_activation': 'linear',
        }
        self.critic = models.build_dnn_models(**critic_settings)
        self.critic_optim = tf.keras.optimizers.Adam(lr=self.lr_critic)

    def _print_hyper_parameters(self,):
        """Prints hyper parameters.
        """
        print('PPO agent')
        print('learning_rate actor = %s, critic = %s' %(self.lr_actor, self.lr_critic))
        print('PPO ppo_epsilon', self.ppo_epsilon)
        print('PPO update timestep: ', self.ppo_update_timestep)
        print('PPO ppo_epochs', self.ppo_epochs)
        print()