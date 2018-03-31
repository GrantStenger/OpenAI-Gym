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

ENV_NAME = "CartPole-v0"
DEQUE_SIZE = 2000
LEARNING_RATE = 0.001
GAMMA = 0.95
EXPLORATION_RATE = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
SAMPLE_BATCH_SIZE = 32
EPISODES = 2000

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
		self.model              = self._build_model()

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

		if os.path.isfile(self.weight_backup):
			model.load_weights(self.weight_backup)
			self.exploration_rate = self.exploration_min

		return model

	def save_model(self):
			self.model.save(self.weight_backup)

	def act(self, state):
		if np.random.rand() <= self.exploration_rate:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
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

class CartPole:
	def __init__(self):
		self.sample_batch_size = SAMPLE_BATCH_SIZE
		self.episodes          = EPISODES
		self.env               = gym.make(ENV_NAME)
		self.state_size        = self.env.observation_space.shape[0]
		self.action_size       = self.env.action_space.n
		self.agent             = Agent(self.state_size, self.action_size)
		self.scores            = []

	def run(self):
		try:
			for index_episode in range(self.episodes):
				state = self.env.reset()
				state = np.reshape(state, [1, self.state_size])

				done = False
				index = 0
				while not done:
					#self.env.render()

					action = self.agent.act(state)

					next_state, reward, done, _ = self.env.step(action)
					next_state = np.reshape(next_state, [1, self.state_size])
					self.agent.remember(state, action, reward, next_state, done)
					state = next_state
					index += 1
				print("Episode {}# Score: {}".format(index_episode, index))

				self.scores.append(index)

				self.agent.replay(self.sample_batch_size)
		finally:
			self.agent.save_model()
			self.plot()

	def plot(self):
		x_axis = np.arange(0, len(self.scores), 1)
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_axis, self.scores)
		print("Slope: ", slope)
		print("Intercept: ", intercept)
		print("R_value: ", r_value)
		print("P_value: ", p_value)
		print("Std_err: ", std_err)
		plt.scatter(x_axis, self.scores, s=10)
		plt.plot(x_axis, intercept + slope*x_axis, 'r', label='fitted line', alpha=0.4)
		plt.title("Scores as Model Learns")
		plt.ylabel('Score')
		plt.xlabel('Episode')
		plt.show()

if __name__ == "__main__":
	cartpole = CartPole()
	cartpole.run()