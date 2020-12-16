"""Defines a trainer that trains agents to master the environments.
"""
from codes.agents import agent_ddpg
from codes.agents import agent_dqn
from codes.agents import agent_ppo
from codes.agents import agent_reinforce
import codes.environment as environment

PRINT_EVERY = 20
RENDER_EVERY = 50

AGGREGATE_EVERY = 50

MAX_EPISODES = 5000
MAX_TIMESTEPS = 1000


def train(env_name,
        solved_reward,
        algo_name,
        aggregate_every=AGGREGATE_EVERY,
        print_every=PRINT_EVERY,
        render_every=RENDER_EVERY,
        max_episodes = MAX_EPISODES,
        max_timesteps = MAX_TIMESTEPS,
        **kwargs):
    """Trains an agent to master an environment.

    Args:
        env_name: [Text] the name of the enviroment
        algo_name: [Text] 'reinforce'/'ppo'/'dqn'/'ddpg'
        solved_reward: the env is solved when rewards over aggregate_every epochs > solved_reward
        aggregate_every: [Int] aggregate the running rewards for aggregate_every epochs to compute rewards
        print_every: [Int] print the result after print_every epochs
        render_every: [Int] render the play after render_every epochs
        max_episodes: [Int] the maximum number of episodes
        max_timesteps: [Int] the maximum number of steps per an episode
    """
    game = environment.Environment(
        env_name=env_name, 
        print_every=print_every,
        render_every=render_every,
        aggregate_every=aggregate_every,
        solved_reward=solved_reward
    )

    # Initializes agent
    if algo_name == 'ddpg':
        agent = agent_ddpg.Agent(
            state_dim=game.state_dim,
            action_dim=game.action_dim,
            output_scale=game.upper_bound,
            **kwargs,
        )
    elif algo_name == 'dqn':
        agent = agent_dqn.Agent(
            state_dim=game.state_dim,
            action_dim=game.action_dim,
            **kwargs,
        )
    elif algo_name == 'ppo':
        agent = agent_ppo.Agent(
            state_dim=game.state_dim,
            action_dim=game.action_dim,
            **kwargs,
        )
    elif algo_name == 'reinforce':
        agent = agent_reinforce.Agent(
            state_dim=game.state_dim,
            action_dim=game.action_dim,
            **kwargs,
        )
    else:
        raise NotImplementedError

    # Let the agent interacting with the environment
    for episode in range(1, max_episodes + 1):
        # Plays the episode
        state = game.start_episode()

        for _ in range(max_timesteps):
            action, actionProb = agent.get_action_and_actionProb(state)
            next_state, reward, done, _ = game.env.step(action)

            agent.step([state, action, reward, next_state, done, actionProb])
            game.update_history_metrics(reward)

            state = next_state
            if done:
                break

        agent.update_after_an_episode()
        if game.is_finished():
            agent.save_weights(env_name=env_name)
            break

    game.plot_history_metric()