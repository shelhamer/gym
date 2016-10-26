import numpy as np
import click

import caffe
import gym

from policies import NetworkPolicy, EpsilonGreedyPolicy, RepeatPolicy
from atari_envs import DQNObsEnv, BufferedObsEnv, NoopResetEnv

@click.command()
@click.argument('env')
@click.argument('arch', type=click.Path(exists=True))
@click.argument('weights', type=click.Path(exists=True))
@click.option('--num_episodes', default=100)
@click.option('--max_episode_time', default=60 * 60 * 5)
@click.option('--seed', default=0)
@click.option('--gpu', default=0)
def eval_policy(env, arch, weights, num_episodes, max_episode_time, seed, gpu):
    # set seeds
    np.random.seed(seed)

    # environment: fully deterministic Atari from pixels with
    # - random no-op restarts
    # - DQN-style pre-processing of frames
    # - DQN-style frame stacking
    click.secho('scoring {} for {} episodes (with time limit {})'.format(env, num_episodes, max_episode_time), fg='blue')
    env = gym.make('{}NoFrameskip-v3'.format(env))
    env.seed(seed)
    env = NoopResetEnv(env)
    env = DQNObsEnv(env)
    env = BufferedObsEnv(env, skip=4)

    # agent: network policy w/ epsilon-greedy behavior that acts at 15 hz
    caffe.set_device(gpu)
    caffe.set_mode_gpu()
    agent = NetworkPolicy(arch, weights=weights)
    agent = EpsilonGreedyPolicy(agent, epsilon_start=0.05, epsilon_end=0.05)
    agent = RepeatPolicy(agent, action_repeat=4)

    # record episode returns and lengths
    returns, times = np.zeros((num_episodes)), np.zeros((num_episodes))
    with click.progressbar(range(num_episodes)) as bar:
        for i in bar:
            done = False
            obs = env.reset()
            # rollout until end or time limit
            while not done:
                action = agent.act(obs)
                obs, reward, done, _ = env.step(action)
                returns[i] += reward
                times[i] += 1
                done = done or times[i] == max_episode_time

    for i, r, t in zip(range(num_episodes), returns, times):
        click.echo('episode {:3}: return {:10.2f} time {:10.0f}'.format(i, r, t))
    click.echo('returns: {:10.2f} +/- {:10.2f}'.format(returns.mean(), returns.std()))
    click.echo('times:   {:10.2f} +/- {:10.2f}'.format(times.mean(), times.std()))

if __name__ == '__main__':
    eval_policy()
