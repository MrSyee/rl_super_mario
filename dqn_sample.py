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
import gym
# import gym_pull
import ppaquette_gym_super_mario

from wrappers import MarioActionSpaceWrapper
from wrappers import ProcessFrame84

EPISODES = 1000000

# SuperMario DQN Agent
class DQNAgent:
    def __init__(self, n_action):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4) # 84, 84 화면이 4장
        # 마리오는 224, 256 -> resize 할것
        self.n_action = n_action
        # DQN 하이퍼파라미터
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000. # 탐험을 얼마나 할것인가. epsilon 크기가 계속 줄어든다
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        # 리플레이 메모리, 최대 크기 400000
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
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
            'summary/breakout_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save_model/breakout_dqn.h5")

        # 상태가 입력, 큐함수가 출력인 인공신경망 생성
        def build_model(self):
            model = Sequential() # conv filter output size = (N-F/Stride) + 1
            model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',   # 84x84 -> 20x20 (padding=valid(default) or same)
                             input_shape=self.state_size))
            model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))  # 20x20 -> 9x9
            model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))  # 9x9 -> 7x7
            model.add(Flatten())
            model.add(Dense(512, activation='relu')) # 7x7x64= 3136 -> 512
            model.add(Dense(self.n_action))       # 512 -> n_action
            model.summary()
            return model

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

        def get_action(self, history):
            history = np.float32(history / 255.0)
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.n_action)
            else:
                q_value = self.model.predict(history)
                return np.argmax(q_value[0])

# 학습속도를 높이기 위해 흑백화면으로 전처리 + size도 줄인다.
def pre_processing(observe):
    # gray_img = cv2.cvtColor(observe, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.resize(gray_img, (84, 84), interpolation=cv2.INTER_AREA)
    # gray_img = gray_img.astype(np.unit8)

    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

def process(img):
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    x_t = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
    x_t = np.reshape(x_t, (84, 84, 1))
    x_t = np.nan_to_num(x_t)
    return x_t.astype(np.uint8)


if __name__ == "__main__":
    # 환경과 DQN 에이전트 생성
    env = gym.make("ppaquette/SuperMarioBros-1-1-v0")
    # Apply action space wrapper
    env = MarioActionSpaceWrapper(env)
    # Apply observation space wrapper to reduce input size
    env = ProcessFrame84(env)

    agent = DQNAgent(n_action=4)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset() # [224, 256, 3]

        for _ in range(random.randint(1, agent.no_op_steps)): # 1 ~ np_op_steps(30) 까지의 수중 하나를 고른다. 그 후 그 수만큼 for문 돌림.
            observe, _, _, _ = env.step(1)

        # state = pre_processing(observe)
        history = np.stack((observe, observe, observe, observe), axis=2) # 맨처음엔 같은 화면 4개를 history로
        history = np.reshape([history], (1, 84, 84, 4))

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
                real_action = 10,  # Right + A + B
            elif action == 1:
                real_action = 9,  # Right + B
            elif action == 2:
                real_action = 8,  # Right + A
            else:
                real_action = 7,  # Right

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observe, reward, done, info = env.step(real_action)
            # 각 타임스텝마다 상태 전처리
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1.)
            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, dead)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

            if dead:
                dead = False
            else:
                history = next_history

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

                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0

        # 1000 에피소드마다 모델 저장
        if e % 1000 == 0:
            agent.model.save_weights("./save_model/breakout_dqn.h5")
