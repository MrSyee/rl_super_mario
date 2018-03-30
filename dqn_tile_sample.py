from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
from collections import deque
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
import os
import h5py
import gym
# import gym_pull
import ppaquette_gym_super_mario

from wrappers import MarioActionSpaceWrapper
from wrappers import ProcessFrame84

EPISODES = 3000
savefile_name = "supermario_dqn_v_tile_1_1.h5"
summary_name = 'summary/mario_dqn_v_tile_1_1'

if not os.path.isdir('./save_model/'):
    os.mkdir("./save_model/")

# SuperMario DQN Agent
class DQNAgent:
    def __init__(self, n_action=5):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = (13, 16, 4) # 84, 84 화면이 4장
        # 마리오는 224, 256 -> resize 할것
        self.n_action = n_action
        # DQN 하이퍼파라미터
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000
        # 탐험을 얼마나 할것인가. epsilon 크기가 계속 줄어든다
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 32
        self.train_start = 5000
        self.update_target_rate = 1000
        self.discount_factor = 0.99
        self.epoch = 1
        # 리플레이 메모리, 최대 크기 400000
        self.memory = deque(maxlen=40000)
        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            summary_name, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save_model/%s" % savefile_name)

        # supermario_dqn.h5 : action_size = 4
        # supermario_dqn2.h5 : action_size = 5 more jump, and when exploring do jump

        # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential() # conv filter output size = (N-F/Stride) + 1
        '''
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',   # 84x84 -> 20x20 (padding=valid(default) or same)
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))  # 20x20 -> 9x9
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))  # 9x9 -> 7x7
        '''
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

    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # select action
    def get_action(self, history):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_action) # 탐험시 점프가 포함된 명령+ right
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        for _ in range(self.epoch):
            if self.epsilon > self.epsilon_end:
                self.epsilon -= self.epsilon_decay_step

            mini_batch = random.sample(self.memory, self.batch_size) # replay memory에서 batch_size만큼 뽑아옴

            history = np.zeros((self.batch_size, self.state_size[0],      # [batch_size(32), 84, 84, 4]
                                self.state_size[1], self.state_size[2]))
            next_history = np.zeros((self.batch_size, self.state_size[0], # [batch_size(32), 84, 84, 4]
                                     self.state_size[1], self.state_size[2]))
            target = np.zeros((self.batch_size,))
            action, reward, dead = [], [], []

            for i in range(self.batch_size):
                history[i] = np.float32(mini_batch[i][0])
                next_history[i] = np.float32(mini_batch[i][3])
                action.append(mini_batch[i][1])
                reward.append(mini_batch[i][2])
                dead.append(mini_batch[i][4])

            target_value = self.target_model.predict(next_history)

            for i in range(self.batch_size):
                if dead[i]:
                    target[i] = reward[i]
                else:
                    target[i] = reward[i] + self.discount_factor * \
                                            np.amax(target_value[i])

            loss = self.optimizer([history, action, target])
            self.avg_loss += loss[0]

    # Huber Loss를 이용하기 위해 최적화 함수를 직접 정의
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.n_action)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

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
    # 환경과 DQN 에이전트 생성
    # 환경 종류 : SuperMarioBros-1-1-v0([224, 226, 3]), SuperMarioBros-1-1-Tiles-v0([13, 16, 1])
    #           meta-SuperMarioBros-Tiles-v0 (클리어 시 다음 스테이지로)
    env = gym.make("ppaquette/SuperMarioBros-1-1-Tiles-v0")
    # Apply action space wrapper
    env = MarioActionSpaceWrapper(env)
    # Apply observation space wrapper to reduce input size [224, 256, 3] -> [84, 84, 1]
    # env = ProcessFrame84(env)

    n_action = 9
    agent = DQNAgent(n_action=n_action)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset() # [13, 16, 1]

        # breakout 예제에서 처음에 구석으로 몰리는 것을 방지하기 위함
        #for _ in range(random.randint(1, agent.no_op_steps)): # 1 ~ np_op_steps(30) 까지의 수중 하나를 고른다. 그 후 그 수만큼 for문 돌림.
        #    observe, _, _, _ = env.step(1)

        # state = pre_processing(observe)
        history = np.stack((observe, observe, observe, observe), axis=2) # 맨처음엔 같은 화면 4개를 history로
        history = np.reshape([history], (1, 13, 16, 4))

        reward_count = 0 # 계속 같은 곳에 머물 경우를 체크

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1
            '''
                0: [0, 0, 0, 0, 0, 0]  # NOOP
                1: [1, 0, 0, 0, 0, 0]  # Up
                2: [0, 0, 1, 0, 0, 0]  # Down
                3: [0, 1, 0, 0, 0, 0]  # Left
                4: [0, 1, 0, 0, 1, 0]  # Left + A
                5: [0, 1, 0, 0, 0, 1]  # Left + B
                6: [0, 1, 0, 0, 1, 1]  # Left + A + B
                7: [0, 0, 0, 1, 0, 0]  # Right
                8: [0, 0, 0, 1, 1, 0]  # Right + A
                9: [0, 0, 0, 1, 0, 1]  # Right + B
                10: [0, 0, 0, 1, 1, 1]  # Right + A + B
                11: [0, 0, 0, 0, 1, 0]  # A
                12: [0, 0, 0, 0, 0, 1]  # B
                13: [0, 0, 0, 0, 1, 1]  # A + B
            '''
            # 바로 전 4개의 상태로 행동을 선택
            action = agent.get_action(history)


            # action
            if action == 0:
                real_action = [0, 0, 0, 1, 1, 1]  # Right + A + B
            elif action == 1:
                real_action = [0, 0, 0, 0, 1, 0]  # A
            elif action == 2:
                real_action = [0, 0, 0, 1, 1, 0]  # Right + A
            elif action == 3:
                real_action = [0, 0, 0, 1, 0, 1]  # Right + B
            elif action == 4:
                real_action = [0, 0, 0, 0, 1, 1]  # A + B

            elif action == 5:
                real_action = [0, 1, 0, 0, 1, 0]  # Left + A
            elif action == 6:
                real_action = [0, 1, 0, 0, 1, 1]  # Left + A + B
            elif action == 7:
                real_action = [0, 0, 0, 0, 0, 0]  # NOOP
            elif action == 8:
                real_action = [0, 1, 0, 0, 0, 0]  # Left


            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_observe, reward, done, info = env.step(real_action)

            # 각 타임스텝마다 상태 전처리
            # next_state = pre_processing(next_observe)
            next_state = np.reshape([next_observe], (1, 13, 16, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history))[0])

            reward = np.clip(reward, -1., 1.) # reward를 -1 ~ 1 사이의 값으로 만듬

            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, dead)
            # print ("global_step : " ,global_step)

            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward
            history = next_history

            # 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            if done:
                # 각 에피소드 당 학습 정보를 기록
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)

                print("episode:", e, "  reward:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0

        # 1000 에피소드마다 모델 저장
        if e % 50 == 0:
            agent.model.save_weights("./save_model/%s" % savefile_name)
            print ("save model %d step" % e)
