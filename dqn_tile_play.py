from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
import os
import gym
import time
# import gym_pull
import ppaquette_gym_super_mario

from wrappers import MarioActionSpaceWrapper
from wrappers import ProcessFrame84

EPISODES = 1
savefile_name = "supermario_dqn_v_tile_1_2.h5"

if not os.path.isdir('./save_model/'):
    os.mkdir("./save_model/")

# SuperMario DQN Agent
class DQNAgent:
    def __init__(self, n_action=5):
        self.render = False
        self.load_model = True
        # 상태와 행동의 크기 정의
        self.state_size = (13, 16, 4) # 84, 84 화면이 4장
        # 마리오는 224, 256 -> resize 할것
        self.n_action = n_action
        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = self.build_model()

        if self.load_model:
            self.model.load_weights("./save_model/%s" % savefile_name)

        # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential() # conv filter output size = (N-F/Stride) + 1
        model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu',   # 13 x 16 -> 7x8 (padding=valid(default) or same)
                         input_shape=self.state_size, padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu')) # 7x7x64= 3136 -> 512
        model.add(Dense(self.n_action))       # 512 -> n_action
        model.summary()
        return model

    # 학습된 신경망을 불러옴
    def load_model(self, filename):
        self.model.load_weights(filename)

    # select action
    def get_action(self, history):
        if np.random.rand() <= 0:
            return random.randrange(self.n_action) # 탐험시 점프가 포함된 명령+ right
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

if __name__ == "__main__":
    # 환경 종류 : SuperMarioBros-1-1-v0([224, 226, 3]), SuperMarioBros-1-1-Tiles-v0([13, 16, 1])
    #           meta-SuperMarioBros-Tiles-v0 (클리어 시 다음 스테이지로)
    env = gym.make("ppaquette/SuperMarioBros-1-1-Tiles-v0")
    # Apply action space wrapper
    env = MarioActionSpaceWrapper(env)
    # Apply observation space wrapper to reduce input size
    # env = ProcessFrame84(env)

    agent = DQNAgent(n_action=9)
    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset() # [224, 256, 3] -> [84, 84, 1]

        # breakout 예제에서 처음에 구석으로 몰리는 것을 방지하기 위함
        #for _ in range(random.randint(1, agent.no_op_steps)): # 1 ~ np_op_steps(30) 까지의 수중 하나를 고른다. 그 후 그 수만큼 for문 돌림.
        #    observe, _, _, _ = env.step(1)

        # state = pre_processing(observe)
        history = np.stack((observe, observe, observe, observe), axis=2) # 맨처음엔 같은 화면 4개를 history로
        history = np.reshape([history], (1, 13, 16, 4))

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1
            '''
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
            '''
            # 바로 전 4개의 상태로 행동을 선택
            action = agent.get_action(history)
            #
            if action == 0:
                real_action = [0, 0, 0, 1, 1, 1]  # Right + A + B
                action_name = "Right + A + B"
            elif action == 1:
                real_action = [0, 0, 0, 0, 1, 0]  # A
                action_name = "A"
            elif action == 2:
                real_action = [0, 0, 0, 1, 1, 0]  # Right + A
                action_name = "Right + A"
            elif action == 3:
                real_action = [0, 0, 0, 1, 0, 0]  # Right
                action_name = "Right"
            elif action == 4:
                real_action = [0, 0, 0, 1, 0, 1]  # Right + B
                action_name = "Right + B"

            elif action == 5:
                real_action = [0, 1, 0, 0, 1, 0]  # Left + A
                action_name = "Left + A"
            elif action == 6:
                real_action = [0, 1, 0, 0, 0, 0]  # Left
                action_name = "Left"
            '''
            elif action == 7:
                real_action = [0, 0, 0, 0, 0, 0]  # NOOP
                action_name = "NOOP"
            elif action == 8:
                real_action = [0, 1, 0, 0, 1, 1]  # Left + A + B
                action_name = "Left + A + B"
            '''

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_observe, reward, done, info = env.step(real_action)
            time.sleep(0.02)

            # 각 타임스텝마다 상태 전처리
            # next_state = pre_processing(next_observe)
            next_state = np.reshape([next_observe], (1, 13, 16, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            score = info['distance']
            history = next_history
            print ("step : ", step, " score : ", score, " action :", action, " real_action : ", action_name)

            if done:
                print("episode:", e, "  score:", score)
