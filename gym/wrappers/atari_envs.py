import numpy as np

from collections import deque
from PIL import Image

import gym


# TODO(shelhamer) make compatible with >1 channel
class BufferedObsEnv(gym.ObservationWrapper):
    """Buffer observations and stack e.g. for frame skipping.

    n is the length of the buffer, and number of observations stacked.
    skip is the number of steps between buffered observations.
    """
    def __init__(self, env=None, n=4, skip=4):
        super(BufferedObsEnv, self).__init__(env)
        self.n = n
        self.skip = skip
        self.buffer = deque(maxlen=self.n)
        self.counter = 0  # init and reset should agree on this

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def _observation(self, obs):
        if self.counter % self.skip == 0:
            self.buffer.append(obs)
        self.counter += 1
        return np.stack(self.buffer)

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        obs = self.env.reset()
        self.clear(obs)
        return np.stack(self.buffer)

    # observation(), clear() are not for the env, but preprocessing other data

    def observation(self, obs):
        return self._observation(obs)

    def clear(self, obs):
        self.buffer.clear()
        self.counter = 0
        for i in range(self.n):
            self.buffer.append(obs)
        return np.stack(self.buffer)


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset.

    No-op is assumed to be action 0.
    """
    def __init__(self, env=None, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)
        return obs


class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing."""
    def __init__(self, env=None, noop_max=30):
        super(FireResetEnv, self).__init__(env)

    # TODO(shelhamer) whitelist envs that require this by name by env.spec.id
    def _reset(self):
        """Take fire action exactly once after reset."""
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over.

    Done by DeepMind for the DQN and co. since it helps value estimation.
    """
    def __init__(self, env=None):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = self.env.unwrapped.ale.lives()

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.

        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.lives == 0:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class DQNObsEnv(gym.ObservationWrapper):
    """Preprocess ALE frames in the DeepMind DQN style.

    1. take max across past and current frame to reduce flicker
    2. convert to greyscale
    3. resize to 84x84

    n.b. The DQN states buffer these observations into a stack of 4.
    """
    def __init__(self, env=None, shape=(84, 84)):
        super(DQNObsEnv, self).__init__(env)
        self.obs_shape = shape
        # most recent raw observations (for max pooling across time steps)
        self.obs_buffer = deque(maxlen=2)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def _observation(self, obs):
        """Take obs from env and preprocess by DeepMind rules."""
        self.obs_buffer.append(obs)
        max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        intensity_frame = self._rgb2y(max_frame).astype(np.uint8)
        small_frame = np.array(Image.fromarray(intensity_frame).resize(
            self.obs_shape, resample=Image.BILINEAR), dtype=np.uint8)
        return small_frame

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self.obs_buffer.clear()
        return self._observation(self.env.reset())

    def _rgb2y(self, im):
        """Converts an RGB image to a Y image (as in YUV).
        These coefficients are taken from the torch/image library.
        """
        return np.sum(im * [0.299, 0.587, 0.114], axis=2)

    # observation(), clear() are not for the env, but preprocessing other data

    def observation(self, obs):
        return self._observation(obs)

    def clear(self, obs):
        self.obs_buffer.clear()
        return self._observation(obs)
