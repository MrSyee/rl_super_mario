
'''
reference : https://github.com/chris-chris/mario-rl-tutorial (kakao)
'''

import cv2
import gym
import numpy as np
from gym import spaces

# 학습속도를 높이기 위해 gray_scale로 전처리 + size도 줄인다.
class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(img): # [224, 256, 3]
        # img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        x_t = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_t = cv2.resize(x_t, (84, 84), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(x_t, (84, 84, 1))
        x_t = np.nan_to_num(x_t) # Replace nan with zero and inf with finite numbers.
        return (np.float32(x_t.astype(np.uint8) / 255.))


class MarioActionSpaceWrapper(gym.Wrapper):
    """
      Wrapper to convert MultiDiscrete action space to Discrete
      Only supports one config, which maps to the most logical discrete space possible
    """
    mapping = {
    0: [0, 0, 0, 0, 0, 0],  # NOOP
    1: [1, 0, 0, 0, 0, 0],  # Up
    2: [0, 0, 1, 0, 0, 0],  # Down
    3: [0, 1, 0, 0, 0, 0],  # Left
    4: [0, 1, 0, 0, 1, 0],  # Left + A
    5: [0, 1, 0, 0, 0, 1],  # Left + B
    6: [0, 1, 0, 0, 1, 1],  # Left + A + B
    7: [0, 0, 0, 1, 0, 0],  # Right
    8: [0, 0, 0, 1, 1, 0],  # Right + A
    9: [0, 0, 0, 1, 0, 1],  # Right + B
    10: [0, 0, 0, 1, 1, 1],  # Right + A + B
    11: [0, 0, 0, 0, 1, 0],  # A
    12: [0, 0, 0, 0, 0, 1],  # B
    13: [0, 0, 0, 0, 1, 1],  # A + B
    }

    def __init__(self, env):
        super(MarioActionSpaceWrapper, self).__init__(env)
        self.action_space = spaces.Discrete(14)

    def _action(self, action):
        return self.mapping.get(action)

    def _reverse_action(self, action):
        for k in self.mapping.keys():
          if(self.mapping[k] == action):
            return self.mapping[k]
        return 0

class SetPlayingModeWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SetPlayingModeWrapper, self).__init__(env)
        if target_mode not in ['algo', 'human']:
            raise gym.error.Error('Error - The mode "{}" is not supported. Supported options are "algo" or "human"'.format(target_mode))
        self.unwrapped.mode = target_mode
