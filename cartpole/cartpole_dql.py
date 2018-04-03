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

ENV_NAME = "CartPole-v0"
DEQUE_SIZE = 2000
LEARNING_RATE = 0.001
GAMMA = 0.85
EXPLORATION_RATE = 1.0
EXPLORATION_MIN = 0.01
#EXPLORATION_DECAY = 0.998 #exp_rate = .5 after 342 episodes; .1 after 1140; .01 after 3400
EXPLORATION_DECAY = 0.995 #exp_rate = .5 after 138 episodes; .1 after 460; .01 after 914
#EXPLORATION_DECAY = 0.99 #exp_rate = .5 after 69 episodes; .1 after 228; .01 after 455
#EXPLORATION_DECAY = 0.95 #exp_rate = .5 after 13 episodes; .1 after 45; .01 after 89
SAMPLE_BATCH_SIZE = 32
EPISODES = 10
SCORE_TO_BEAT = 195
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

		#self.explore_model      = self.build_explore_model()
		#self.explore_weight_backup = ENV_NAME + "_explore_weight.h5"
		self.q_plot             = []

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
			#self.model.save(self.explore_weight_backup)

	def act(self, state, score):
		act_values = self.model.predict(state)
		q = np.amax(act_values[0])
		if score == 0:
			self.q_plot.append(q)
		if np.random.rand() <= self.exploration_rate:
			return random.randrange(self.action_size)
		
		#print(act_values)
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
				#print(GAMMA*(200-score))
				next_state, reward, done, _ = self.env.step(action)
				next_state = np.reshape(next_state, [1, self.state_size])
				self.agent.remember(state, action, reward, next_state, done)
				state = next_state
				score += 1

			#print("Episode {}# Score: {}".format(episode, score))

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
			plot_model(self.agent.model)
			#self.plot(self.scores)
			#self.plot(self.mean_plot)
			#self.plot(self.agent.q_plot)
			"""
			x_axis = np.arange(0, len(self.scores), 1)
			plt.scatter(x_axis, self.scores, s=10)

			x_axis = np.arange(0, len(self.agent.q_plot), 1)
			plt.scatter(x_axis, self.agent.q_plot, s=10)

			plt.title("Scores as Model Learns")
			plt.ylabel('Score')
			plt.xlabel('Episode')
			plt.show()
			"""

			df = pd.DataFrame({'scores':self.scores, 'qs':self.agent.q_plot})
			df = df.sort_values(by=['qs'])
			self.plot(df['qs'], df['scores'])

			"""
			x_values1=np.arange(0, len(self.scores), 1)
			y_values1=df['scores']

			x_values2=np.arange(0, len(self.agent.q_plot), 1)
			y_values2=df['qs']

			fig=plt.figure()
			ax=fig.add_subplot(111, label="1")
			ax2=fig.add_subplot(111, label="2", frame_on=False)

			ax.scatter(x_values1, y_values1, color="C0", s=10)
			ax.set_xlabel("Episode", color="C0")
			ax.set_ylabel("Score", color="C0")
			ax.tick_params(axis='x', colors="C0")
			ax.tick_params(axis='y', colors="C0")

			ax2.scatter(x_values2, y_values2, color="C1", s=10)
			ax2.xaxis.tick_top()
			ax2.yaxis.tick_right()
			ax2.set_xlabel('Step', color="C1") 
			ax2.set_ylabel('Q value', color="C1")       
			ax2.xaxis.set_label_position('top') 
			ax2.yaxis.set_label_position('right') 
			ax2.tick_params(axis='x', colors="C1")
			ax2.tick_params(axis='y', colors="C1")

			plt.show()
			"""

	def plot(self, x, y):
		#x_axis = np.arange(0, len(scores), 1)
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