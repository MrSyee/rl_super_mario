
# Modu Flipped School RL Class Kim Kyung Hwan
# SuperMario Agent Train

import gym
# import gym_pull
import ppaquette_gym_super_mario

import numpyt as np

def train_mario():
    # Create gym environment (Mario)
    # gym_pull.pull('github.com/ppaquette/gym-super-mario')


def main():
    # train_mario()
    env = gym.make("ppaquette/SuperMarioBros-1-1-v0")
    state = env.reset() # [224, 256, 3]
    height = np.shape(state)[1]/2
    state = state[:, :height, :] # [224, 128, 3]

    for _ in range(100000):
        env.render()
        env.step(env.action_space.sample()) # random action

if __name__ == '__main__':
    main()
