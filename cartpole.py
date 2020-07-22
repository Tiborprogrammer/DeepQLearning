import gym
from pathlib import Path

from Agent import Agent, run_episode, load_agent, evaluate_agent

env = gym.make("CartPole-v1")
state = env.reset()

start_exploration_rate = 1
end_exploration_rate = 0.05

exploration_change = (start_exploration_rate - end_exploration_rate) / 500
deadline_for_exploartion_change = 500

episode = 0
training = False
agent_dir = "/Users/coding/Desktop/Developer/Python/CartPole-v1"
Path(agent_dir).mkdir(parents=True, exist_ok=True)

value = input("Load agent y/n?")
if value.lower() == "y":
	agent = load_agent(agent_dir)
	print("You loaded the agent!")
else:
	agent = Agent(env.observation_space.shape[0], env.action_space.n, 0.05, start_exploration_rate, "CartPole-v1")
	print("Agent not loaded!")

while True:
	if training:
		if episode % 100 == 0:
			evaluate_agent(env, agent, agent_dir, 20, episode)

		run_episode(env, agent, should_learn=True)

		if episode < deadline_for_exploartion_change:
			agent.exploration_rate -= exploration_change
	else:
		evaluate_agent(env, agent, agent_dir, 20, episode, should_render=True)

	episode += 1