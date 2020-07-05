import gym
import random
import torch.nn as nn
from torch.optim import Adam
import torch
import json
from collections import deque

class ReplayBuffer:
	def __init__(self, max_size):
		self.transitions = deque(maxlen=max_size)

	def add(self, transition):
		self.transitions.append(transition)

	def sample(self, num_of_samples):
		samples = []
		for i in range(0, num_of_samples):
			samples.append(random.choice(self.transitions))

		return samples

class Agent:
	def __init__(self, state_dimension, num_actions, lr, start_exploration_rate):
		self.state_dimension = state_dimension
		self.num_actions = num_actions
		self.lr = lr
		self.exploration_rate = start_exploration_rate
		self.Q_values_model = nn.Sequential(
			nn.Linear(state_dimension, 32),
			nn.ReLU(),
			nn.Linear(32, num_actions)
		)
		self.optimiser = Adam(self.Q_values_model.parameters(), lr=lr)
		self.replay_buffer = ReplayBuffer(10000)


	def pick_action(self, state, should_learn):
		state = torch.tensor(state).float().unsqueeze(0)
		if random.random() < self.exploration_rate and should_learn:
			return random.randint(0, self.num_actions - 1)

		Q_vals_for_state = self.Q_values_model(state)[0]
		best_Q_val = float("-inf")
		best_action = 0
		for i in range(self.num_actions):
			if best_Q_val < Q_vals_for_state[i]:
				best_Q_val = Q_vals_for_state[i]
				best_action = i
		return best_action

	def learn_from_transitions(self, states, actions, reward_for_actions, new_states, dones):
		actions = actions.unsqueeze(1)
		states = states.float()
		new_states = new_states.float()

		self.optimiser.zero_grad()

		val_for_next_state = self.Q_values_model(new_states).max(dim=1).values.detach()
		estimate_q_vals = reward_for_actions + val_for_next_state * 0.95 * (1 - dones)

		current_Q_values = self.Q_values_model(states).gather(1, actions)
		errors = (estimate_q_vals - current_Q_values) ** 2
		error = errors.mean()

		error.backward()
		self.optimiser.step()

	def learn(self):
		transitions = self.replay_buffer.sample(100)
		states = []
		actions = []
		reward_for_actions = []
		new_states = []
		dones = []
		for state, action, reward_for_action, new_state, done in transitions:
			states.append(state)
			actions.append(action)
			reward_for_actions.append(reward_for_action)
			new_states.append(new_state)
			dones.append(done)

		self.learn_from_transitions(torch.tensor(states), torch.tensor(actions),
									torch.tensor(reward_for_actions), torch.tensor(new_states),
									torch.tensor(dones)
									)

def run_episode(should_learn):
	steps_taken = 0
	total_reward = 0
	done = False
	state = env.reset()

	while not done:
		action = agent.pick_action(state, should_learn)

		next_state, reward, done, _ = env.step(action)
		agent.replay_buffer.add((state, action, reward, next_state, int(done)))

		steps_taken += 1
		total_reward += reward
		if should_learn:
			agent.learn()
		else:
			pass
			#env.render()
		state = next_state

	return (total_reward, steps_taken)

def save_agent(dir, agent:Agent):
	q_network_state = agent.Q_values_model.state_dict()
	torch.save(q_network_state, dir + "/q_network_state.ckpt")

	agent_data = {"exploration_rate" : agent.exploration_rate, "num_actions" : agent.num_actions,
				  "lr" : agent.lr, "state_dimension" : agent.state_dimension}

	with open(dir + "/agent.json", "w") as f:
		json.dump(agent_data, f)




def load_agent(dir):
	with open(dir + "/agent.json") as f:
		agent_data = json.load(f)
	q_network_state = torch.load(dir + "/q_network_state.ckpt")
	agent = Agent(agent_data["state_dimension"], agent_data["num_actions"],
				  agent_data["lr"], agent_data["exploration_rate"])

	agent.Q_values_model.load_state_dict(q_network_state)

	return agent


env = gym.make("CartPole-v1")
state = env.reset()

start_exploration_rate = 1
end_exploration_rate = 0.05

exploration_change = (start_exploration_rate - end_exploration_rate) / 100000
deadline_for_exploartion_change = 100000

episode = 0

value = input("Load agent?")
if value == "Y":
	dir = "/Users/coding/Desktop/Developer/Python"
	agent = load_agent(dir)
else:
	agent = Agent(env.observation_space.shape[0], env.action_space.n, 0.05, start_exploration_rate)

while True:
	if episode % 100 == 0:
		sum_of_reward = 0
		sum_of_steps = 0
		for i in range(0, 2):
			reward, steps_taken = run_episode(should_learn=False)
			sum_of_reward += reward
			sum_of_steps += steps_taken

		average_reward = sum_of_reward / 2
		average_steps = sum_of_steps / 2
		print(f"Evaluation result: ")
		print(f"Reward: {average_reward}   Steps taken: {average_steps}, Episode: {episode}")
		print("---------------------------------")
		save_agent("/Users/coding/Desktop/Developer/Python", agent)

	run_episode(should_learn=True)
	episode += 1

	if episode < deadline_for_exploartion_change:
		agent.exploration_rate -= exploration_change




		#runningAverage = 0.99 * runningAverage + 0.01 * totalReward

		#print(f"RunningAverage: {runningAverage:.4f} Episode: {episode} Reward: {totalReward} Steps: {stepsTaken}")
