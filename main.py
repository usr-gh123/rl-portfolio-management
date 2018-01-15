from rl_portfolio_management.environments.portfolio import PortfolioEnv
from rl_portfolio_management.wrappers import SoftmaxActions, ConcatStates
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.agents import VPGAgent
from tensorforce.execution import Runner
from tensorforce.core.optimizers import NaturalGradient
import time
import numpy as np

class TFOpenAIGymCust(OpenAIGym):
    def __init__(self, gym_id, gym):
        self.gym_id = gym_id
        self.gym = gym


env = PortfolioEnv(
    [],
    steps=200,
    scale=True,
    trading_cost=0.0003,
    window_length = 30,
    output_mode='mlp',
)

class TFOpenAIGymCust(OpenAIGym):
    def __init__(self, gym_id, gym):
        self.gym_id = gym_id
        self.gym = gym
        self.visualize = False


env = ConcatStates(env)
environment = TFOpenAIGymCust('CryptoPortfolioEIIE-v0', env)

env.seed(0)
state = environment.reset()
state, done, reward=environment.execute(env.action_space.sample())

network_spec = [
    dict(type='dense', size=16),
    dict(type='dense', size=10)
]

agent = VPGAgent(
    states_spec=environment.states,
    actions_spec=environment.actions,
    batch_size=20,
    network_spec=network_spec,
    discount=0.8,
    optimizer=dict(
        type='adam',
        learning_rate=1e-4
    )
)

runner = Runner(
    agent=agent,
    environment=environment,
    repeat_actions=1
)

report_episodes = 100

print("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))


pvs = []
def episode_finished(r):
    if r.episode % report_episodes == 0:
        steps_per_second = r.timestep / (time.time() - r.start_time)
        print("Finished episode {} after {} timesteps. Steps Per Second {}".format(
            r.agent.episode, r.episode_timestep, steps_per_second
        ))
        #fpv = np.exp(r.episode_rewards[-1])
        fpv = r.episode_rewards[-1]
        print("Episode reward: {}".format(fpv))
        pvs.append(fpv)
        print(
            "Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
        print(
            "Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
    return True


runner.run(
    episodes=50000,
    episode_finished=episode_finished
)

print("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))
import pandas as pd

pd.DataFrame(pvs).to_csv("mdp.csv")