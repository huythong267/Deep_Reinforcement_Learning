"""Initializes models with trainable weights.
"""
import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

HIDDEN_DIM = 64


def build_dnn_models(input_size, 
                    output_size, 
                    output_activation, 
                    hidden_activation='relu',
                    output_scale=1,
                    hidden_dim=HIDDEN_DIM):
    """Defines a model for either actors or critics.
    """
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    states = Input(shape=(input_size,))
    net = Dense(hidden_dim, activation=hidden_activation)(states)
    net = Dense(hidden_dim, activation=hidden_activation)(net)
    outs = Dense(output_size, activation=output_activation, kernel_initializer=last_init)(net)
    if output_scale != 1:
        outs = Activation(lambda x: x * output_scale)(outs)
    return Model(inputs=states, outputs=outs)


def build_Q_models(state_dim, action_dim, hidden_dim=HIDDEN_DIM):
    """Predicts the values of the states-actions pairs.
    """
    states = Input(shape=(state_dim,))
    states_out = Dense(hidden_dim, activation='relu')(states)
    states_out = Dense(hidden_dim, activation='relu')(states_out)

    actions = Input(shape=(action_dim,))
    actions_out = Dense(hidden_dim, activation='relu')(actions)

    out = Concatenate()([states_out, actions_out])

    out = Dense(hidden_dim, activation='relu')(out)
    out = Dense(hidden_dim, activation='relu')(out)
    outputs = Dense(1)(out)

    return Model([states, actions], outputs)


class Agents:
    #######################################################################################################
    #   Transfer weights
    #######################################################################################################
    @tf.function
    def _transfer_weights(self, target_model, model, tau):
        """Updates gradually the weights of the target model with those of the model.
        """
        for target_weight, weight in zip(target_model.variables, model.variables):
            target_weight.assign((1 - tau) * target_weight + tau * weight)

    #######################################################################################################
    #   Save & Load
    #######################################################################################################
    def save_weights(self, env_name):
        model_dir = os.path.join(os.curdir, 'pre_trained_models')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        for model_name, model in self._get_a_dictionary_of_name_to_model(env_name).items():
            model.save_weights(os.path.join(model_dir, model_name))

    def load_weights(self, env_name):
        model_dir = os.path.join(os.curdir, 'pre_trained_models')

        for model_name, model in self._get_a_dictionary_of_name_to_model(env_name).items():
            model_path = os.path.join(model_dir, model_name)
            if os.path.exists(model_path):
                model.load_weights(model_path)