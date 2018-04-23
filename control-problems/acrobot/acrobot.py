import gym
import random
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
from keras.utils import plot_model

ENV_NAME = "Acrobot-v1"
DEQUE_SIZE = 2000
LEARNING_RATE = 0.001
GAMMA = 0.85
EXPLORATION_RATE = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.998 #exp_rate = .5 after 342 episodes; .1 after 1140; .01 after 3400
SAMPLE_BATCH_SIZE = 32
EPISODES = 1000
SCORE_TO_BEAT = -100
PLOT = True
START_FRESH = False

class Agent():
	def __init__(self, state_size, action_size):
		self.weight_backup      = ENV_NAME + "_weight.h5"
		self.state_size         = state_size
		self.action_size        = action_size
		self.memory             = deque(maxlen=DEQUE_SIZE)
		self.learning_rate      = LEARNING_RATE
		self.gamma              = GAMMA
		self.exploration_rate   = EXPLORATION_RATE
		self.exploration_min    = EXPLORATION_MIN
		self.exploration_decay  = EXPLORATION_DECAY
		self.model              = self.build_model()

	def build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

		if not START_FRESH and os.path.isfile(self.weight_backup):
			model.load_weights(self.weight_backup)
			self.exploration_rate = self.exploration_min

		return model

	def build_explore_model(self):
		explore_model = Sequential()
		explore_model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		explore_model.add(Dense(24, activation='relu'))
		explore_model.add(Dense(1, activation='softmax'))
		explore_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		if not START_FRESH and os.path.isfile(self.explore_weight_backup):
			explore_model.load_weights(self.explore_weight_backup)

	def save_model(self):
			self.model.save(self.weight_backup)

	def act(self, state, score):
		act_values = self.model.predict(state)
		q = np.amax(act_values[0])
		if np.random.rand() <= self.exploration_rate:
			return random.randrange(self.action_size)

		return np.argmax(act_values[0])

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def replay(self, sample_batch_size):
		if len(self.memory) < sample_batch_size:
			return
		sample_batch = random.sample(self.memory, sample_batch_size)
		for state, action, reward, next_state, done in sample_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.exploration_rate > self.exploration_min:
			self.exploration_rate *= self.exploration_decay
		print("Exploration rate:            ", self.exploration_rate)

class CartPole:
	def __init__(self):
		self.sample_batch_size = SAMPLE_BATCH_SIZE
		self.episodes          = EPISODES
		self.env               = gym.make(ENV_NAME)
		self.state_size        = self.env.observation_space.shape[0]
		self.action_size       = self.env.action_space.n
		self.agent             = Agent(self.state_size, self.action_size)
		self.score_to_beat     = SCORE_TO_BEAT
		self.scores            = []
		self.mean_plot         = []

	def run(self):
		last_100_scores = deque(maxlen=100)

		for episode in range(self.episodes):
			state = self.env.reset()
			state = np.reshape(state, [1, self.state_size])

			done = False
			score = 0
			while not done:
				#self.env.render()
				action = self.agent.act(state, score)
				next_state, reward, done, _ = self.env.step(action)
				next_state = np.reshape(next_state, [1, self.state_size])
				self.agent.remember(state, action, reward, next_state, done)
				state = next_state
				score += reward

			print("Episode {}# Score: {}".format(episode, score))

			last_100_scores.append(score)
			mean_score = np.mean(last_100_scores)

			if mean_score >= self.score_to_beat and episode >= 100:
				print('Ran {} episodes. Solved after {} trials with mean score {}'.format(episode, episode - 100, mean_score))
				return episode - 100
			if episode % 100 == 0 and episode > 0:
				print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(episode, mean_score))

			if PLOT:
				self.scores.append(score)
				self.mean_plot.append(mean_score)

			self.agent.replay(self.sample_batch_size)

		print('Did not solve after {} episodes'.format(episode))
		self.agent.save_model()
		if PLOT:
			self.plot(np.arange(0, len(self.scores), 1), self.scores)

	def plot(self, x, y):
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
		print("Slope: ", slope)
		print("Intercept: ", intercept)
		print("R_value: ", r_value)
		print("P_value: ", p_value)
		print("Std_err: ", std_err)
		plt.scatter(x, y, s=10)
		plt.plot(x, intercept + slope*x, 'r', label='fitted line', alpha=0.4)
		plt.title("Scores as Model Learns")
		plt.ylabel('Score')
		plt.xlabel('Episode')
		plt.show()

if __name__ == "__main__":
	cartpole = CartPole()
	cartpole.run()
