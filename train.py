
# Modu Flipped School RL Class Kim Kyung Hwan
# SuperMario Agent Train

import gym
import gym_pull
import ppaquette_gym_super_mario

def train_mario():
    # Create gym environment (Mario)
    # gym_pull.pull('github.com/ppaquette/gym-super-mario')
    env = gym.make("ppaquette/SuperMarioBros-1-1-v0")

def main():
    train_mario()

if __name__ == '__main__':
    main()
