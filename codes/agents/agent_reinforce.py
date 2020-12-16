"""Develops a REINFORCE agent to solve the discrete tasks.
"""
import os
import numpy as np
import tensorflow as tf

import codes.agents.memory as memory
import codes.agents.models as models


GAMMA = 0.99
LR_ACTOR = 0.002
LR_CRITIC = 0.002

REINFORCE_UPDATE_TIMESTEP = 2000
PPO_EPOCHS = 8


class Agent(models.Agents):
    def __init__(self, 
                state_dim,
                action_dim,
                gamma=GAMMA,
                lr_actor=LR_ACTOR,
                reinforce_update_timestep=REINFORCE_UPDATE_TIMESTEP,
                ):
        """Initializes a REINFORCE agent.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.lr_actor = lr_actor

        self.reinforce_update_timestep = reinforce_update_timestep

        self.algo_name = 'reinforce'

        self._build_actor()
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
        if len(self.memory) < self.reinforce_update_timestep:
            return

        self.memory.to_numpy_arrays()
        states = tf.convert_to_tensor(self.memory.states_arr)
        actions_onehot = tf.one_hot(self.memory.actions_arr, self.action_dim)
        values = tf.convert_to_tensor(self.memory.discounted_rewards_arr, dtype=tf.float32)
        
        self._update_weights(states, actions_onehot, values)

        self.memory.reset()

    @tf.function
    def _update_weights(self, states, actions_onehot, values):
        """Updates the trainable weights of the agent.
        """
        # Train actor
        with tf.GradientTape() as tape:
            actionProbs = self.actor(states, training=True) # (N, action_dim)
            this_prob = tf.math.reduce_sum(actionProbs * actions_onehot, axis=-1, keepdims=True) + 1e-5 # (N, 1) 
            actor_loss = - values * tf.math.log(this_prob)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_optim.apply_gradients(zip(actor_grad, self.actor.trainable_weights))

    #######################################################################################################
    #   Follow the function in Save & Load of the agent
    #######################################################################################################
    def _get_a_dictionary_of_name_to_model(self, env_name):
        return {
            '%s_%s_actor.h5' %(self.algo_name, env_name): self.actor,
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

    def _print_hyper_parameters(self,):
        """Prints hyper parameters.
        """
        print('REINFORCE agent')
        print('learning_rate actor = %s' %(self.lr_actor))
        print('REINFORCE update timestep: ', self.reinforce_update_timestep)
        print()